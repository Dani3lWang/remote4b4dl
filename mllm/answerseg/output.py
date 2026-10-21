"""Text-only span probes and SEG insertion gated by successful segmentation."""

from .language import MARKERS


def answer_spans(answer, spans=None):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("a nonempty free answer is required")
    if any(marker in answer for marker in (*MARKERS, "<video>")):
        raise ValueError("segment_answer expects plain text without reserved markers")
    if spans is None:
        # The first interface is single-referent. Use the complete answer so
        # post-nominal descriptions also precede the internal causal probe.
        end = len(answer.rstrip(" \t\n\r.。!！?？;；"))
        start = len(answer) - len(answer.lstrip())
        spans = [[start, end]]
    if not isinstance(spans, list) or not 1 <= len(spans) <= 8:
        raise ValueError("one to eight answer spans are required")
    cursor = 0
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(value) is not int for value in span)
            or not cursor <= span[0] < span[1] <= len(answer)
            or not answer[span[0] : span[1]].strip()
        ):
            raise ValueError(
                "answer spans must be ordered, nonempty and nonoverlapping"
            )
        cursor = span[1]
    return spans


def probe_answer(answer, spans):
    """Internal selector probes; these markers are NOT an output or a prediction."""
    answer_spans(answer, spans)
    parts, cursor = [], 0
    for _, end in spans:
        parts.extend((answer[cursor:end], " <SEG>"))
        cursor = end
    parts.append(answer[cursor:])
    return "".join(parts)


def render_segmented_answer(answer, spans, objects):
    """Insert a marker only after an actual nonempty mask has been selected."""
    answer_spans(answer, spans)
    if len(spans) != len(objects):
        raise ValueError("one segmentation decision per answer span is required")
    parts, cursor, length, marker_index = [], 0, 0, 0
    for span, record in zip(spans, objects):
        end = span[1]
        piece = answer[cursor:end]
        parts.append(piece)
        length += len(piece)
        record.update(
            text_span=span,
            phrase=answer[span[0] : end],
            seg_index=None,
            output_marker_span=None,
        )
        if record["status"] == "selected":
            if record.get("mask_index") != marker_index:
                raise ValueError(
                    "selected markers must correspond to ordered mask rows"
                )
            parts.append(" <SEG>")
            record.update(
                seg_index=marker_index, output_marker_span=[length + 1, length + 6]
            )
            length += 6
            marker_index += 1
        cursor = end
    parts.append(answer[cursor:])
    return "".join(parts)
