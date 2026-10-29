"""Build the six-task query using only the question and ego metadata.

This mirrors the free branch of evaluation.test_b4dl.build_query without
importing the video inference dependencies or accepting a ground-truth answer.
"""
import re

from scripts.ego_text import build_metatoken_line, render_metatoken_for_range


def build_free_query(human_value, ego_meta, scene_id, frame_motion):
    question = human_value.split("<meta>", 1)[0]
    for tag in ("<video>", "<4DLiDAR>"):
        question = question.replace(tag+"\n", "").replace(tag+" ", "").replace(tag, "")
    question = question.strip()
    parts = ["<4DLiDAR>", "<video>", question]
    if ego_meta and scene_id:
        matches = [int(value) for value in re.findall(r"frame\s+(\d+)", question, re.IGNORECASE)]
        first, last = (min(matches), max(matches)) if matches else (None, None)
        body = None
        if first is not None and frame_motion is not None:
            body = render_metatoken_for_range(frame_motion, scene_id, first, last)
        if body is None:
            scene = None
            if first is not None:
                scene = ego_meta.get(f"{scene_id}_{first}_{last}")
                if scene is None:
                    scene = ego_meta.get(f"{scene_id}_{last}_{first}")
            if scene is None:
                scene = ego_meta.get(scene_id, {})
            if isinstance(scene, dict):
                first_text, last_text = scene.get("first_frame", ""), scene.get("last_frame", "")
            else:
                first_text, last_text = scene, ""
            if first_text and last_text:
                body = build_metatoken_line(first_text, last_text)
            elif first_text:
                body = f"The metadata of the first frame is '{first_text}'"
        if body is None:
            body = "No ego motion metadata available for this scene."
        parts.append("<meta> " + body)
    return "\n".join(parts)
