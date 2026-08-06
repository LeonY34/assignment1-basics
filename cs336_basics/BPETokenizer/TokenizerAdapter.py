from __future__ import annotations

from collections.abc import Iterable

import regex as re

from cs336_basics.BPETokenizer.BPEtokenizer import BPEtokenizer


class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        bpe_tokenizer: BPEtokenizer | None = None,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        if bpe_tokenizer is not None:
            vocab = bpe_tokenizer.vocab_map
            merges = list(bpe_tokenizer.merge_ranks.keys())
            if special_tokens is None:
                special_tokens = bpe_tokenizer.special_words

        if vocab is None or merges is None:
            raise ValueError("vocab and merges are required")

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.special_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }

    def pre_tokenize_no_special(self, text: str) -> list[bytes]:
        return [
            match.group().encode("utf-8")
            for match in re.finditer(self.PAT, text)
        ]

    def encode_bytes(self, word: bytes) -> list[int]:
        pieces = [bytes([byte]) for byte in word]

        while len(pieces) > 1:
            best_rank = len(self.merges)
            best_position = -1

            for position, pair in enumerate(zip(pieces[:-1], pieces[1:])):
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_position = position

            if best_position == -1:
                break

            pieces[best_position : best_position + 2] = [
                pieces[best_position] + pieces[best_position + 1]
            ]

        return [self.token_to_id[piece] for piece in pieces]

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return [
                token_id
                for word in self.pre_tokenize_no_special(text)
                for token_id in self.encode_bytes(word)
            ]

        special_pattern = "(" + "|".join(
            re.escape(token)
            for token in sorted(self.special_tokens, key=len, reverse=True)
        ) + ")"

        token_ids: list[int] = []
        for part in re.split(special_pattern, text):
            if not part:
                continue
            if part in self.special_to_id:
                token_ids.append(self.special_to_id[part])
                continue
            for word in self.pre_tokenize_no_special(part):
                token_ids.extend(self.encode_bytes(word))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode(
            "utf-8", errors="replace"
        )
