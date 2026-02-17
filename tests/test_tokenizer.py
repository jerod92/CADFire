"""Tests for the BPE tokenizer."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cadfire.tokenizer.bpe import BPETokenizer, PAD_TOKEN


class TestBPETokenizer:
    def test_encode_decode(self):
        tok = BPETokenizer(vocab_size=4096, max_len=128)
        text = "draw a red circle"
        ids = tok.encode(text)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)
        decoded = tok.decode(ids)
        assert "draw" in decoded

    def test_encode_padded(self):
        tok = BPETokenizer(vocab_size=4096, max_len=32)
        ids = tok.encode_padded("hello")
        assert len(ids) == 32
        assert ids[-1] == tok.token_to_id[PAD_TOKEN]

    def test_truncation(self):
        tok = BPETokenizer(vocab_size=4096, max_len=10)
        text = "this is a very long text that should be truncated because it exceeds the maximum"
        ids = tok.encode(text)
        assert len(ids) <= 10

    def test_train(self):
        tok = BPETokenizer(vocab_size=512, max_len=64)
        corpus = [
            "draw a red circle",
            "draw a blue rectangle",
            "create a green line",
            "draw a circle at 100 200",
        ] * 10
        tok.train(corpus, min_frequency=2)
        assert tok.vocab_actual_size() > 260  # base vocab + merges

    def test_special_tokens(self):
        tok = BPETokenizer()
        ids = tok.encode("test", add_special=True)
        # Should start with BOS and end with EOS
        assert ids[0] == tok.token_to_id["<BOS>"]
        assert ids[-1] == tok.token_to_id["<EOS>"]

    def test_empty_string(self):
        tok = BPETokenizer()
        ids = tok.encode("", add_special=True)
        assert len(ids) == 2  # just BOS + EOS
