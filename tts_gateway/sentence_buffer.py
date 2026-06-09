"""Sentence-aware buffer for streaming text input.

Why this exists: Fun-CosyVoice3 is a sentence-level synthesizer with low
latency, not a per-character one. Pushing each token straight to the engine
either wastes batches or produces choppy prosody. This buffer flushes:

  - on strong punctuation (。！？!?) immediately, regardless of length
  - on weak punctuation (，,；;) only after threshold characters
  - on max_chars overflow (defensive cap; real text rarely hits this)

The first chunk uses a smaller threshold (first_min_chars) to keep TTFB low.
After the first flush we switch to the normal threshold for better prosody.

Single-pass scan: O(n) per push, not O(n*k) where k is the punctuation set.
"""

from __future__ import annotations

from typing import Optional


class SentenceBuffer:
    STRONG = frozenset("。！？!?")
    WEAK = frozenset("，,；;:")

    def __init__(
        self,
        first_min_chars: int = 6,
        min_chars: int = 12,
        max_chars: int = 80,
    ):
        self.buf = ""
        self.first = True
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self.max_chars = max_chars

    def push(self, text: str) -> list[str]:
        """Append text and return any complete sentences ready for synthesis."""
        if not text:
            return []
        self.buf += text
        out: list[str] = []

        while self.buf:
            threshold = self.first_min_chars if self.first else self.min_chars
            cut = self._find_cut(threshold)

            if cut is not None:
                out.append(self.buf[:cut])
                self.buf = self.buf[cut:]
                self.first = False
                continue

            if len(self.buf) >= self.max_chars:
                out.append(self.buf[: self.max_chars])
                self.buf = self.buf[self.max_chars :]
                self.first = False
                continue

            break

        return [x.strip() for x in out if x.strip()]

    def flush(self) -> list[str]:
        """Force-flush remaining buffered text. Used on tts.end / tts.flush."""
        if not self.buf.strip():
            self.buf = ""
            return []
        out = [self.buf.strip()]
        self.buf = ""
        self.first = False
        return out

    def reset(self) -> None:
        self.buf = ""
        self.first = True

    def _find_cut(self, threshold: int) -> Optional[int]:
        """Return the slice length to flush (i.e., cut index past the boundary).

        Rules:
          - Any strong punctuation flushes immediately, regardless of length.
          - The first weak punctuation at or after `threshold` flushes too.
          - Otherwise return None (wait for more text or hit max_chars cap).

        Single-pass O(n) over the buffer.
        """
        for i, ch in enumerate(self.buf):
            if ch in self.STRONG:
                return i + 1
            if ch in self.WEAK and i + 1 >= threshold:
                return i + 1
        return None
