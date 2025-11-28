# Byte-Pair Encoding (BPE) Tokenizer: A From-Scratch Implementation

## Abstract

This repository presents a from-scratch implementation of the Byte-Pair Encoding (BPE) algorithm, a subword tokenization technique foundational to modern large language models (LLMs). We implement the complete BPE pipeline—vocabulary initialization, iterative pair merging, encoding, and decoding—and provide empirical comparison against OpenAI's GPT-2 tokenizer. Our implementation demonstrates the core mechanics of how neural language models transform raw text into learnable numerical representations.

## 1. Introduction

### 1.1 The Tokenization Problem

Neural networks operate on numerical data, yet natural language exists as sequences of characters. Tokenization bridges this gap by segmenting text into discrete units (tokens) that can be mapped to numerical indices. The choice of tokenization strategy profoundly impacts model performance, vocabulary size, and computational efficiency.

### 1.2 Tokenization Approaches

Three primary paradigms exist:

| Approach | Unit | Vocabulary Size | Semantic Meaning | OOV Handling |
|----------|------|-----------------|------------------|--------------|
| Word-based | Words | ~500,000+ | High | Poor (requires `[UNK]`) |
| Character-based | Characters | ~256 | Low | Perfect |
| Subword (BPE) | Variable | ~30,000-50,000 | Medium-High | Excellent |

Word-based tokenization produces semantically meaningful tokens but suffers from vocabulary explosion and out-of-vocabulary (OOV) failures. Character-based tokenization eliminates OOV issues but loses semantic coherence and increases sequence length. Subword tokenization, pioneered by BPE, achieves an optimal trade-off.

### 1.3 Byte-Pair Encoding

BPE was originally developed as a data compression algorithm (Gage, 1994) and later adapted for neural machine translation (Sennrich et al., 2016). The algorithm iteratively merges the most frequent adjacent symbol pairs, progressively building a vocabulary of subword units. This enables:

- **Compact vocabularies**: Typically 30,000-50,000 tokens
- **Zero OOV tokens**: Any word decomposes into known subwords
- **Morphological awareness**: Common affixes (e.g., "-est", "-er", "-ing") emerge as tokens

## 2. Algorithm

### 2.1 Training Phase

The BPE training algorithm proceeds as follows:
```
ALGORITHM: BPE Training
INPUT: Corpus C, number of merges N
OUTPUT: Merge operations M, final vocabulary V

1. Initialize vocabulary V with all unique characters in C
2. Tokenize each word in C into characters, append end-of-word marker </w>
3. FOR i = 1 to N:
   a. Count frequency of all adjacent token pairs
   b. Identify most frequent pair (p₁, p₂)
   c. Merge all occurrences: (p₁, p₂) → p₁p₂
   d. Add merged token to V
   e. Record merge operation in M
4. RETURN M, V
```

### 2.2 Encoding Phase

Given learned merge operations M, encoding new text follows:
```
ALGORITHM: BPE Encoding
INPUT: Text T, merge operations M
OUTPUT: Token sequence S

1. Split T into words
2. FOR each word w:
   a. Initialize tokens as character sequence + </w>
   b. FOR each merge (p₁, p₂) in M (in learned order):
      - Replace all adjacent (p₁, p₂) with merged token p₁p₂
   c. Append resulting tokens to S
3. RETURN S
```

### 2.3 Decoding Phase

Decoding simply concatenates tokens and removes boundary markers:
```
ALGORITHM: BPE Decoding
INPUT: Token sequence S
OUTPUT: Reconstructed text T

1. Concatenate all tokens in S
2. Replace </w> with space
3. Strip trailing whitespace
4. RETURN T
```

## 3. Implementation

### 3.1 Corpus

We train on a minimal corpus to demonstrate algorithmic behavior:
```python
corpus = [
    "low lower newest",
    "widest wider low",
    "newest lowest wider"
]
```

Total unique words: 6 (`low`, `lower`, `newest`, `widest`, `wider`, `lowest`)

### 3.2 Core Functions

| Function | Purpose |
|----------|---------|
| `get_vocab(corpus)` | Tokenizes corpus into character sequences with frequency counts |
| `get_pairs(vocab)` | Counts adjacent token pair frequencies across vocabulary |
| `merge_vocab(pair, vocab)` | Applies single merge operation to entire vocabulary |
| `encode(text, merges)` | Tokenizes new text using learned merge operations |
| `decode(tokens)` | Reconstructs original text from token sequence |

### 3.3 End-of-Word Marker

We append `</w>` to each word's final character. This critical design choice:

1. Distinguishes word-final tokens from word-internal tokens
2. Enables proper word boundary reconstruction during decoding
3. Allows different treatment of identical substrings in different positions

Example: `"low"` → `['l', 'o', 'w', '</w>']`

## 4. Results

### 4.1 Learned Merge Operations

Training with `N=10` merges produces:

| Merge # | Operation | Frequency | Linguistic Interpretation |
|---------|-----------|-----------|---------------------------|
| 1 | `l + o → lo` | 4 | Common bigram |
| 2 | `lo + w → low` | 4 | Root word formation |
| 3 | `e + s → es` | 4 | Suffix component |
| 4 | `es + t → est` | 4 | Superlative suffix |
| 5 | `est + </w> → est</w>` | 4 | Word-final superlative |
| 6 | `e + r → er` | 3 | Comparative suffix |
| 7 | `er + </w> → er</w>` | 3 | Word-final comparative |
| 8 | `w + i → wi` | 3 | Prefix component |
| 9 | `wi + d → wid` | 3 | Root formation |
| 10 | `low + </w> → low</w>` | 2 | Complete word token |

### 4.2 Vocabulary Evolution
```
Initial: {l, o, w, e, r, s, t, n, i, d, </w>}  — 11 tokens
Final:   {l, o, w, e, r, s, t, n, i, d, </w>, lo, low, es, est, est</w>, 
          er, er</w>, wi, wid, low</w>}        — 21 tokens
```

### 4.3 Encoding Examples

| Input | Tokens | Count |
|-------|--------|-------|
| `"lowest"` | `['low', 'est</w>']` | 2 |
| `"wider"` | `['wid', 'er</w>']` | 2 |
| `"newest"` | `['n', 'e', 'w', 'est</w>']` | 4 |

## 5. Comparison with GPT-2

### 5.1 Methodology

We compare our tokenizer against OpenAI's GPT-2 tokenizer, which uses Byte-Level BPE trained on ~40GB of web text with a vocabulary of 50,257 tokens.

### 5.2 Results

| Input | Our BPE | GPT-2 |
|-------|---------|-------|
| `"lowest wider"` | `['low', 'est</w>', 'wid', 'er</w>']` (4) | `['low', 'est', 'Ġwider']` (3) |
| `"newest lower"` | `['n', 'e', 'w', 'est</w>', 'low', 'er</w>']` (6) | `['new', 'est', 'Ġlower']` (3) |
| `"widest low"` | `['wid', 'est</w>', 'low</w>']` (3) | `['wid', 'est', 'Ġlow']` (3) |

*Note: GPT-2's `Ġ` represents a preceding space (word boundary marker).*

### 5.3 Analysis

**Efficiency Gap**: GPT-2 achieves higher compression because:

1. **Training scale**: GPT-2 trained on billions of tokens; we trained on 9 words
2. **Whole-word tokens**: GPT-2 learned `"wider"`, `"lower"`, `"new"` as atomic tokens
3. **Vocabulary size**: GPT-2 has 50,257 tokens vs. our 21

**Algorithmic Equivalence**: Despite the efficiency gap, both tokenizers:

- Successfully decompose unseen words into subword units
- Recognize the same morphological patterns (`est`, `er`, `wid`)
- Achieve identical token counts on well-represented vocabulary (`"widest low"`)

## 6. Discussion

### 6.1 Key Insights

1. **BPE learns morphology implicitly**: Without linguistic annotation, BPE discovers meaningful units like comparative (`-er`) and superlative (`-est`) suffixes.

2. **Frequency drives vocabulary**: High-frequency substrings become tokens regardless of linguistic validity—BPE is purely statistical.

3. **Merge order matters**: Encoding must apply merges in training order. Applying merges out-of-order produces different (incorrect) tokenizations.

4. **Vocabulary size controls granularity**: More merges → larger vocabulary → shorter sequences → more whole-word tokens.

### 6.2 Limitations

- **No semantic awareness**: BPE treats text as raw bytes/characters with no understanding of meaning
- **Language-dependent efficiency**: Agglutinative languages (Turkish, Finnish) may require larger vocabularies
- **Tokenization artifacts**: Semantically identical inputs may tokenize differently based on spacing or casing

### 6.3 Extensions

Modern variants of BPE include:

- **Byte-Level BPE** (GPT-2): Operates on UTF-8 bytes, eliminating unknown characters
- **WordPiece** (BERT): Uses likelihood-based scoring instead of frequency
- **Unigram** (SentencePiece): Probabilistic model that removes tokens iteratively
- **SentencePiece**: Language-agnostic, treats input as raw Unicode stream

## 7. Conclusion

We implemented Byte-Pair Encoding from first principles, demonstrating how modern LLMs convert text to numerical sequences. Our 50-line implementation captures the essential algorithm underlying tokenizers used in GPT, BERT, LLaMA, and virtually all contemporary language models. The comparison with GPT-2 confirms algorithmic correctness while illustrating how training scale impacts tokenization efficiency.

## 8. Usage

### Installation
```bash
git clone https://github.com/sgg21004/BPE-Tokenizer-Algo.git
cd BPE-Tokenizer-Algo
pip install transformers  # Optional: for GPT-2 comparison
```

### Run
```bash
python bpe.py
```

### Expected Output
```
Learned merges: [('l', 'o'), ('lo', 'w'), ('e', 's'), ('es', 't'), ...]

Input:   'lowest wider'
Tokens:  ['low', 'est</w>', 'wid', 'er</w>']
Decoded: 'lowest wider'

--- Comparison: Our BPE vs GPT-2 ---

Input: 'lowest wider'
  Our BPE:  ['low', 'est</w>', 'wid', 'er</w>'] (4 tokens)
  GPT-2:    ['low', 'est', 'Ġwider'] (3 tokens)
```

## References

1. Gage, P. (1994). A New Algorithm for Data Compression. *C Users Journal, 12*(2), 23-38.

2. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715-1725.

3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*.

4. Hugging Face. (2024). LLM Course: Tokenizers. https://huggingface.co/learn/llm-course/en/chapter2/4

## License

MIT

---

*Built as part of a systematic study of foundational AI/ML concepts.*
