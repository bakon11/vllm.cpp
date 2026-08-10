// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE (GPT-2 bytes_to_unicode bijection + merge-ranked pair merging).
#include "vllm/tokenizer/bpe.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vllm/tokenizer/unicode_data.h"

namespace vllm::tok {
namespace {

constexpr bool IsPrintableByte(int b) {
  return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) ||
         (b >= 0xAE && b <= 0xFF);
}

// GPT-2 bytes_to_unicode: printable bytes keep their codepoint; the 68
// remaining bytes get 0x100 + n in increasing byte order, so every mapped
// codepoint is < 0x144.
constexpr uint32_t kMappedEnd = 0x144;

struct Tables {
  std::array<uint32_t, 256> byte_to_cp{};
  std::array<int16_t, kMappedEnd> cp_to_byte{};

  Tables() {
    cp_to_byte.fill(-1);
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
      const uint32_t cp =
          IsPrintableByte(b) ? static_cast<uint32_t>(b) : 0x100 + n++;
      byte_to_cp[static_cast<size_t>(b)] = cp;
      cp_to_byte[cp] = static_cast<int16_t>(b);
    }
  }
};

const Tables& GetTables() {
  static const Tables t;
  return t;
}

}  // namespace

uint32_t ByteToUnicode(uint8_t b) { return GetTables().byte_to_cp[b]; }

int32_t UnicodeToByte(uint32_t cp) {
  if (cp >= kMappedEnd) return -1;
  return GetTables().cp_to_byte[cp];
}

std::string MapBytesToUnicode(std::string_view raw) {
  std::string out;
  out.reserve(raw.size() * 2);
  for (const char c : raw) {
    EncodeUtf8(ByteToUnicode(static_cast<uint8_t>(c)), out);
  }
  return out;
}

std::string UnmapUnicodeToBytes(std::string_view mapped) {
  std::string out;
  out.reserve(mapped.size());
  size_t pos = 0;
  while (pos < mapped.size()) {
    const uint32_t cp = DecodeUtf8(mapped, pos);
    const int32_t b = UnicodeToByte(cp);
    if (b < 0) {
      throw std::runtime_error(
          "tokenizer: codepoint U+" + std::to_string(cp) +
          " is not in the byte-level alphabet (not a byte-level BPE token "
          "string)");
    }
    out.push_back(static_cast<char>(b));
  }
  return out;
}

std::string MergeKey(std::string_view left, std::string_view right) {
  std::string key;
  key.reserve(left.size() + right.size() + 1);
  key.append(left);
  key.push_back(' ');  // never occurs inside a mapped-alphabet symbol
  key.append(right);
  return key;
}

void BpeMerge(std::vector<std::string>& symbols, const MergeRanks& ranks) {
  // Heap + doubly-linked list BPE. Exact HF semantics (lowest rank, leftmost
  // on ties via stable index) but O(n log n) instead of the old O(n^2) scan +
  // vector erase. Required for Gemma-4: metaspace_split=false feeds the whole
  // prompt as one piece — naive merge was ~60s for 20k tokens / ~2 t/s class
  // wall on Hermes SOUL+tools tokenize.
  const size_t n = symbols.size();
  if (n < 2) return;

  std::vector<int32_t> prev(n), next(n);
  std::vector<uint8_t> alive(n, 1);
  for (size_t i = 0; i < n; ++i) {
    prev[i] = static_cast<int32_t>(i) - 1;
    next[i] = (i + 1 < n) ? static_cast<int32_t>(i + 1) : -1;
  }

  // min-heap of (rank, left_index). left_index breaks ties leftmost-first.
  using Node = std::pair<int32_t, int32_t>;  // rank, left_idx
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;

  auto consider = [&](int32_t i) {
    if (i < 0) return;
    const int32_t j = next[static_cast<size_t>(i)];
    if (j < 0 || !alive[static_cast<size_t>(i)] ||
        !alive[static_cast<size_t>(j)]) {
      return;
    }
    const auto it =
        ranks.find(MergeKey(symbols[static_cast<size_t>(i)],
                            symbols[static_cast<size_t>(j)]));
    if (it == ranks.end()) return;
    heap.emplace(it->second, i);
  };

  for (size_t i = 0; i + 1 < n; ++i) consider(static_cast<int32_t>(i));

  size_t alive_count = n;
  while (!heap.empty() && alive_count >= 2) {
    const auto [rank, i] = heap.top();
    heap.pop();
    if (i < 0 || !alive[static_cast<size_t>(i)]) continue;
    const int32_t j = next[static_cast<size_t>(i)];
    if (j < 0 || !alive[static_cast<size_t>(j)]) continue;
    // Stale heap entry: pair rank may have changed after neighbor merges.
    const auto it =
        ranks.find(MergeKey(symbols[static_cast<size_t>(i)],
                            symbols[static_cast<size_t>(j)]));
    if (it == ranks.end() || it->second != rank) continue;

    // Merge j into i.
    symbols[static_cast<size_t>(i)] += symbols[static_cast<size_t>(j)];
    alive[static_cast<size_t>(j)] = 0;
    --alive_count;
    const int32_t k = next[static_cast<size_t>(j)];
    next[static_cast<size_t>(i)] = k;
    if (k >= 0) prev[static_cast<size_t>(k)] = i;

    // New pairs involving i.
    consider(prev[static_cast<size_t>(i)]);
    consider(i);
  }

  // Compact survivors in order.
  std::vector<std::string> out;
  out.reserve(alive_count);
  int32_t cur = -1;
  for (size_t t = 0; t < n; ++t) {
    if (alive[t]) {
      cur = static_cast<int32_t>(t);
      break;
    }
  }
  while (cur >= 0) {
    out.push_back(std::move(symbols[static_cast<size_t>(cur)]));
    cur = next[static_cast<size_t>(cur)];
  }
  symbols.swap(out);
}

std::vector<std::string> BpeSplit(std::string_view mapped_pretoken,
                                  const MergeRanks& ranks) {
  // Start from single-codepoint symbols.
  std::vector<std::string> symbols;
  size_t pos = 0;
  while (pos < mapped_pretoken.size()) {
    const size_t begin = pos;
    (void)DecodeUtf8(mapped_pretoken, pos);
    symbols.emplace_back(mapped_pretoken.substr(begin, pos - begin));
  }
  BpeMerge(symbols, ranks);
  return symbols;
}

}  // namespace vllm::tok
