"""Context budgeting for scene chat, independent of CUDA and model weights."""

from __future__ import annotations


LIDAR_PREFIX = "<4DLiDAR>\n<video>\n"


def prepare_chat_prompt(conversation, tokenize, feature_count, context_limit, max_new_tokens):
    """Copy/prune complete turns while retaining one full-scene feature token.

    `tokenize` accepts a prompt and returns a sequence (or a 1-D tensor).
    The generation budget includes the LiDAR embeddings replacing <video>.
    Failed generation must not mutate the caller's conversation state.
    """
    candidate = conversation.copy()
    dropped = 0
    while True:
        for message in candidate.messages:
            if message[1] is not None:
                message[1] = message[1].replace(LIDAR_PREFIX, "")
        candidate.messages[0][1] = LIDAR_PREFIX + candidate.messages[0][1]
        input_ids = tokenize(candidate.get_prompt())
        expanded_length = len(input_ids) - 1 + feature_count
        if expanded_length + max_new_tokens <= context_limit:
            return candidate, input_ids, dropped
        if len(candidate.messages) <= 2:
            raise ValueError(
                f"当前问题与 LiDAR 特征需要 {expanded_length + max_new_tokens} tokens，"
                f"超过上下文上限 {context_limit}；请缩短问题或降低 --max_new_tokens"
            )
        del candidate.messages[:2]
        dropped += 1
