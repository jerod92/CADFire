"""
Byte-Pair Encoding tokenizer for CAD text prompts.

Designed to be simple, self-contained, and trainable on CAD-specific vocabulary.
Handles command names, numbers, colors, and natural language instructions.

Usage:
    tokenizer = BPETokenizer(vocab_size=4096)
    tokenizer.train(corpus_texts)   # or load from checkpoint
    ids = tokenizer.encode("Draw a red circle at 100,200")
    text = tokenizer.decode(ids)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class BPETokenizer:
    """Byte-pair encoding tokenizer with CAD-aware pre-tokenization."""

    def __init__(self, vocab_size: int = 4096, max_len: int = 128):
        self.vocab_size = vocab_size
        self.max_len = max_len
        # Initialize with byte-level vocab + special tokens
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Build base vocabulary: special tokens + all single bytes."""
        self.token_to_id = {}
        self.id_to_token = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok
        # Add all printable ASCII + common chars
        offset = len(SPECIAL_TOKENS)
        for b in range(256):
            ch = chr(b) if 32 <= b < 127 else f"<0x{b:02X}>"
            self.token_to_id[ch] = offset + b
            self.id_to_token[offset + b] = ch

    def _pre_tokenize(self, text: str) -> List[str]:
        """Split text into words, keeping numbers and punctuation separate."""
        # Split on whitespace, keep numbers together, separate punctuation
        tokens = re.findall(r'\d+\.?\d*|[a-zA-Z]+|[^\s\w]|\s+', text.lower())
        return [t for t in tokens if t.strip()]

    def _word_to_chars(self, word: str) -> List[str]:
        """Convert a word to list of character tokens."""
        return list(word)

    def train(self, texts: List[str], min_frequency: int = 2):
        """Train BPE merges from a corpus of texts."""
        # Build word frequency
        word_freqs: Counter = Counter()
        for text in texts:
            for word in self._pre_tokenize(text):
                word_freqs[" ".join(self._word_to_chars(word))] += 1

        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.token_to_id)
        for _ in range(max(0, num_merges)):
            # Count all adjacent pairs
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = pair_freqs.most_common(1)[0]
            if best_pair[1] < min_frequency:
                break

            a, b = best_pair[0]
            merged = a + b
            self.merges.append((a, b))

            # Add to vocab
            if merged not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[merged] = idx
                self.id_to_token[idx] = merged

            # Apply merge to all words
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = word.replace(f"{a} {b}", merged)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

    def _apply_merges(self, chars: List[str]) -> List[str]:
        """Apply learned merges to a character sequence."""
        for a, b in self.merges:
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                    new_chars.append(a + b)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
        return chars

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token ids."""
        ids = []
        if add_special:
            ids.append(self.token_to_id[BOS_TOKEN])

        for word in self._pre_tokenize(text):
            chars = self._word_to_chars(word)
            merged = self._apply_merges(chars)
            for tok in merged:
                ids.append(self.token_to_id.get(tok, self.token_to_id[UNK_TOKEN]))

        if add_special:
            ids.append(self.token_to_id[EOS_TOKEN])

        # Pad or truncate
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]

        return ids

    def encode_padded(self, text: str, add_special: bool = True) -> List[int]:
        """Encode and pad to max_len."""
        ids = self.encode(text, add_special)
        pad_id = self.token_to_id[PAD_TOKEN]
        while len(ids) < self.max_len:
            ids.append(pad_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text."""
        tokens = []
        for idx in ids:
            tok = self.id_to_token.get(idx, UNK_TOKEN)
            if tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def save(self, path: str):
        """Save tokenizer state to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "merges": self.merges,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load tokenizer state from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.max_len = data["max_len"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        tok = cls()
        tok.load(path)
        return tok

    def vocab_actual_size(self) -> int:
        return len(self.token_to_id)
