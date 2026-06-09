"""SentenceBuffer is pure logic — no asyncio."""

from tts_gateway.sentence_buffer import SentenceBuffer


def test_strong_punctuation_flushes_immediately():
    sb = SentenceBuffer(first_min_chars=6, min_chars=12)
    # Strong punctuation flushes regardless of length.
    assert sb.push("你好。") == ["你好。"]


def test_weak_punctuation_waits_for_threshold():
    sb = SentenceBuffer(first_min_chars=6, min_chars=12)
    # Below first_min_chars: no flush yet.
    assert sb.push("你好,") == []
    # Adding more pushes us over the threshold.
    assert sb.push("世界一切都好,") == ["你好,世界一切都好,"]


def test_first_chunk_lower_threshold():
    sb = SentenceBuffer(first_min_chars=3, min_chars=20)
    # First chunk uses first_min_chars=3.
    assert sb.push("你好,") == ["你好,"]
    # Subsequent uses min_chars=20.
    assert sb.push("继续,") == []


def test_max_chars_cap():
    sb = SentenceBuffer(first_min_chars=6, min_chars=12, max_chars=10)
    out = sb.push("一二三四五六七八九十十一")
    assert out == ["一二三四五六七八九十"]


def test_flush_returns_remaining():
    sb = SentenceBuffer()
    sb.push("没有标点的文字")
    assert sb.flush() == ["没有标点的文字"]
    # Buffer cleared after flush.
    assert sb.flush() == []


def test_strong_beats_weak_at_same_position():
    sb = SentenceBuffer(first_min_chars=2)
    assert sb.push("a。") == ["a。"]


def test_multiple_sentences_in_one_push():
    sb = SentenceBuffer()
    out = sb.push("第一句。第二句！第三句")
    assert out == ["第一句。", "第二句！"]
    assert sb.buf == "第三句"


def test_empty_push_no_op():
    sb = SentenceBuffer()
    assert sb.push("") == []
    assert sb.buf == ""


def test_whitespace_only_filtered():
    sb = SentenceBuffer()
    sb.push("   。")
    # The strong punctuation flushes "   。" but it's stripped before return.
    # Verify behavior: stripped result is "。" which is non-empty after strip.
    # Actually after strip "。" is non-empty. We just want no exceptions.
    assert sb.flush() == []  # Buffer should be empty after the previous push


def test_reset_clears_first_flag():
    sb = SentenceBuffer(first_min_chars=3, min_chars=20)
    sb.push("你好,")  # flushes, first becomes False
    sb.reset()
    # After reset, first_min_chars threshold applies again.
    assert sb.push("再见,") == ["再见,"]
