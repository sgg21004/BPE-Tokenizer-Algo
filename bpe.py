# bpe.py

corpus = [
    "low lower newest",
    "widest wider low",
    "newest lowest wider"
]

def get_vocab(corpus):
    vocab = {}
    for sentence in corpus:
        words = sentence.split()
        for word in words:
            chars = ' '.join(list(word)) + ' </w>'
            if chars in vocab:
                vocab[chars] += 1
            else:
                vocab[chars] = 1
    return vocab

def get_pairs(vocab):
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair in pairs:
                pairs[pair] += freq
            else:
                pairs[pair] = freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def encode(text, merges):
    words = text.split()
    encoded = []
    
    for word in words:
        tokens = list(word) + ['</w>']
        
        for pair in merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [''.join(pair)] + tokens[i + 2:]
                else:
                    i += 1
        
        encoded.extend(tokens)
    
    return encoded

def decode(tokens):
    text = ''.join(tokens)
    text = text.replace('</w>', ' ')
    return text.strip()

# Training
num_merges = 10
vocab = get_vocab(corpus)
merges = []

for i in range(num_merges):
    pairs = get_pairs(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    merges.append(best_pair)
    vocab = merge_vocab(best_pair, vocab)

print("Learned merges:", merges)
print()

# Test encode/decode
test_text = "lowest wider"
tokens = encode(test_text, merges)
decoded = decode(tokens)

print(f"Input:   '{test_text}'")
print(f"Tokens:  {tokens}")
print(f"Decoded: '{decoded}'")

# Compare with GPT-2 tokenizer
from transformers import AutoTokenizer

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

test_texts = ["lowest wider", "newest lower", "widest low"]

print("\n--- Comparison: Our BPE vs GPT-2 ---\n")

for text in test_texts:
    our_tokens = encode(text, merges)
    gpt2_tokens = gpt2_tokenizer.tokenize(text)
    
    print(f"Input: '{text}'")
    print(f"  Our BPE:  {our_tokens} ({len(our_tokens)} tokens)")
    print(f"  GPT-2:    {gpt2_tokens} ({len(gpt2_tokens)} tokens)")
    print()