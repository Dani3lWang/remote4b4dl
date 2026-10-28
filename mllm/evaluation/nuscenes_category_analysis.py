"""Auditable text-only nuScenes category association for every generated prediction.

No questions/ground truths are used to infer entities missing from the prediction.
Occurrence-level assertion cues are kept separately from the taxonomy mapping.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
import argparse
from datetime import date
from pathlib import Path

TASKS = ("description", "temporal_understanding", "comprehensive_reasoning")
STATUSES = ("肯定提及", "否定存在", "否定新增", "条件或可能", "泛指或建议")
CLASSES = [
    (1, "barrier", "隔离栏/临时路障", "Thing", "movable_object.barrier"),
    (2, "bicycle", "自行车", "Thing", "vehicle.bicycle"),
    (3, "bus", "公交车", "Thing", "vehicle.bus.bendy; vehicle.bus.rigid"),
    (4, "car", "汽车/SUV/van", "Thing", "vehicle.car"),
    (5, "construction_vehicle", "工程车辆", "Thing", "vehicle.construction"),
    (6, "motorcycle", "摩托车", "Thing", "vehicle.motorcycle"),
    (7, "pedestrian", "行人", "Thing", "human.pedestrian.adult; human.pedestrian.child; human.pedestrian.construction_worker; human.pedestrian.police_officer"),
    (8, "traffic_cone", "交通锥", "Thing", "movable_object.trafficcone"),
    (9, "trailer", "挂车", "Thing", "vehicle.trailer"),
    (10, "truck", "卡车/皮卡", "Thing", "vehicle.truck"),
    (11, "driveable_surface", "可行驶表面", "Stuff", "flat.driveable_surface"),
    (12, "other_flat", "其他平坦区域", "Stuff", "flat.other"),
    (13, "sidewalk", "人行道/步行道", "Stuff", "flat.sidewalk"),
    (14, "terrain", "自然地表/低矮草地", "Stuff", "flat.terrain"),
    (15, "manmade", "建筑/人工结构", "Stuff", "static.manmade"),
    (16, "vegetation", "树木/灌木/植被", "Stuff", "static.vegetation"),
]
ORDER = {c[1]: c[0] for c in CLASSES}
CN = {c[1]: c[2] for c in CLASSES}
KINDS = {c[1]: c[3] for c in CLASSES}
SOURCES = [
    ("官方16类及ignore映射", "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/utils.py"),
    ("官方Thing/Stuff划分", "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/panoptic/utils.py"),
    ("对象类别定义：SUV/van/施工车辆/骑行者", "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md"),
    ("地面/植被/人工结构定义", "https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_lidarseg.md"),
]

def rx(s):
    return re.compile(s, re.I)

# Composite labels take priority over embedded nouns (ego car, police car,
# bicycle rack, bus stop). Specific patterns precede broad category patterns.
EXTRA_RULES = [
    ("vehicle.ego", rx(r"\bego[ _-]?(?:vehicles?|cars?)\b")),
    ("vehicle.emergency.police", rx(r"\bpolice\s+(?:vehicles?|cars?|motorcycles?|bicycles?)\b")),
    ("vehicle.emergency.ambulance", rx(r"\bambulances?\b")),
    ("static_object.bicycle_rack", rx(r"\b(?:bicycle|bike)\s+racks?\b")),
    ("human.pedestrian.personal_mobility", rx(r"\b(?:personal mobility (?:devices?|vehicles?)|segways?|skateboards?|kick scooters?)\b")),
    ("human.pedestrian.wheelchair", rx(r"\bwheelchairs?\b")),
    ("human.pedestrian.stroller", rx(r"\b(?:strollers?|baby carriages?|prams?)\b")),
    ("animal", rx(r"\b(?:animals?|dogs?|cats?|birds?|deer|rats?)\b")),
    ("movable_object.debris", rx(r"\b(?:debris|trash bags?|garbage bags?|temporary road[ -]?signs?)\b")),
    ("movable_object.pushable_pullable", rx(r"\b(?:shopping carts?|wheelbarrows?|doll(?:ey|ie)s?|wheeled (?:bins?|garbage bins?))\b")),
    ("noise", rx(r"\b(?:lidar noise|sensor noise|noise points?)\b")),
    ("static.other", rx(r"\b(?:unidentifiable static objects?|unknown static objects?)\b")),
]
RULES = [
    ("manmade", rx(r"\b(?:bus stops?|bus shelters?|traffic lights?|traffic signals?|traffic signs?|street signs?|road signs?|signposts?|lamp[ -]?posts?|street[ -]?lights?|guard[ -]?rails?|railings?|buildings?|walls?|bridges?|fences?|poles?|tunnels?|houses?|hydrants?|parking meters?|artificial structures?|man[ -]?made structures?|static structures?)\b"), "直接词/语义同义词"),
    ("construction_vehicle", rx(r"\b(?:construction[ _-]vehicles?|excavators?|bulldozers?|road rollers?|crane (?:trucks?|vehicles?))\b"), "直接词/语义同义词"),
    ("truck", rx(r"\b(?:concrete mixer trucks?|dump trucks?|semi[ -]trailer trucks?|semi[ -]tractors?|pickup(?: trucks?)?|pick[ -]up trucks?|trucks?|lorr(?:y|ies))\b"), "直接词/语义同义词"),
    ("car", rx(r"\b(?:cars?|SUVs?|vans?|mini[ -]?vans?|sedans?|jeeps?|taxis|taxi|hatchbacks?|automobiles?)\b"), "直接词/语义同义词"),
    ("bus", rx(r"\b(?:buses|bus|coaches|coach|shuttle buses)\b"), "直接词/语义同义词"),
    ("motorcycle", rx(r"\b(?:(?:motorcycle|motorbike) riders?|motorcycles?|motorbikes?|motorcyclists?)\b"), "直接词/骑行者关联"),
    ("bicycle", rx(r"\b(?:bicycles?|cyclists?|bicyclists?|bicycle riders?)\b"), "直接词/骑行者关联"),
    ("pedestrian", rx(r"\b(?:pedestrians?|human[ _.-]pedestrian(?:[ _.-](?:adults?|child|construction[ _.-]worker|police[ _.-]officer))?|construction workers?|police officers?|persons?|people|humans?|walking (?:persons?|people)|children|child)\b"), "直接词/语义同义词"),
    ("barrier", rx(r"\b(?:barriers?|barricades?|roadblocks?)\b"), "直接词/语义同义词"),
    ("traffic_cone", rx(r"\b(?:(?:traffic[ _-]?|construction |safety |road )?cones?|trafficcone)\b"), "直接词/语义同义词"),
    ("trailer", rx(r"\b(?:trailers?|semi[ -]trailers?)\b"), "直接词/语义同义词"),
    ("vegetation", rx(r"\b(?:trees?|bushes|bush|shrubs?|vegetation|plants?|tall grass)\b"), "直接词/语义同义词"),
    ("terrain", rx(r"\b(?:terrain|grassy (?:areas?|surfaces?|patches?)|grass|soil|sand|gravel)\b"), "语义关联，草地高度需场景核对"),
    ("other_flat", rx(r"\b(?:other[ _-]flat|other flat surfaces?|lakes?|rivers?|bodies of water|rail tracks?)\b"), "直接词/语义关联"),
    ("sidewalk", rx(r"\b(?:sidewalks?|footpaths?|pedestrian walkways?|walkways?|bike paths?|bicycle paths?)\b"), "直接词/语义同义词"),
    ("driveable_surface", rx(r"\b(?:driv(?:e)?able[ _-](?:surfaces?|areas?)|roads?|roadways?|streets?|lanes?|intersections?|crosswalks?|road markings?|parking (?:lots?|areas?))\b"), "道路/车道语义关联"),
]
AMBIGUOUS_RULES = [
    ("vehicle泛称", rx(r"\b(?:vehicles?|automotive objects?)\b"), "car/bus/truck/trailer/construction_vehicle/motorcycle/bicycle；不能确定具体类"),
    ("obstacle/object泛称", rx(r"\b(?:obstacles?|objects?)\b"), "对象类型未说明；不能直接推出barrier等类别"),
    ("施工区域", rx(r"\b(?:construction (?:zones?|areas?|sites?|scenes?|elements?)|roadwork)\b"), "施工情境；不能直接推出construction_vehicle"),
    ("施工设备", rx(r"\b(?:(?:construction|heavy) (?:equipment|machinery)|equipment|machinery|concrete mixers?|cement mixers?|cranes?)\b"), "construction_vehicle/truck/manmade或原始ignore类；需确认设备形态"),
    ("structure泛称", rx(r"\b(?:structures?|fixed objects?|static objects?)\b"), "人工结构可关联manmade，但泛称本身不足以确认具体分割类"),
    ("bike/rider/scooter泛称", rx(r"\b(?:bikes?|riders?|scooters?)\b"), "bicycle/motorcycle或human.pedestrian.personal_mobility；需确认交通工具类型"),
    ("ground/pavement泛称", rx(r"\b(?:ground|pavement|curbs?)\b"), "driveable_surface/sidewalk/terrain/other_flat；需确认表面用途"),
    ("仅环境描述", rx(r"\b(?:urban|suburban|cityscape|environment|construction zone)\b"), "环境或场景属性，不独立对应一个分割类"),
]
NEG_PREFIX = rx(r"\b(?:no|without|absence of|lack of|neither|not any)\s+(?:(?:visible|dynamic|moving|stationary|static|nearby|other|any|unexpected|small|smaller|large|significant|immediate|clear|detectable|apparent|additional|specific|physical)\s+)*(?:(?:objects?|obstacles?)\s+(?:like|such as)\s+)?$")
NEG_POST = rx(r"^\s*(?:and\s+(?:other\s+)?\w+\s*)?(?:(?:is|are|was|were)\s+(?:not\s+(?:(?:currently|clearly|readily|immediately)\s+)?(?:visible|present|seen|detected|observed|there)|absent|missing)|(?:cannot|can't|could not)\s+be\s+(?:seen|detected)|(?:do|does)\s+not\s+(?:appear|exist)|(?:are|is)\s+nowhere)")
NOUN_NEG = rx(r"\b(?:no|without|absence of|lack of|neither|not any)\b")
ACTION_OR_RISK_NEG = rx(r"\b(?:changes?|movements?|motion|activity|risks?|threats?|hazards?|danger|issues?|challenges?|need|difference|displacement|variation|speed|collision|impact|effects?|concerns?|importance|change in|changes in)\b")
INDEPENDENT_BREAK = rx(r"[,;]|\b(?:but|while|whereas|although|with|because|causing|indicating|suggesting)\b|\band\s+(?=(?:the|a|an)\s)")
EXISTENCE_CUE = rx(r"\b(?:visible|present|seen|detected|observed|shows?|captures?|includes?|featur(?:es|ing)|there\s+(?:is|are)|presence of|moving|moves?|approach\w*|parked|stationary|remain\w*|positioned|cross(?:es|ing)|located|on the|in front|behind|ahead)\b")
ADVICE = rx(r"\b(?:should|must|need(?:s)? to|requires?|requiring|be aware|be cautious|caution|ensure|ensuring|safe navigation|careful navigation|potential hazards?|risks?|danger|collision)\b")
FRAME = rx(r"\b(?:between\s+)?frames?\s+\d+(?:\s*(?:to|and|[-–])\s*(?:frames?\s+)?\d+)?|\bfrom\s+frame\s+\d+\s+to\s+frame\s+\d+")
# Intentionally limited to a complete, entity-free answer. The omitted referent
# stays unknown and is still sampled for context-sensitive review.
FRAME_ONLY = rx(r"\s*(?:the answer is\s+)?(?:from|between)\s+frames?\s+\d+\s+(?:to|and)\s+(?:frames?\s+)?\d+\s*[.!]?\s*")
# Preserve the scope of this observed coordinated phrase. Do not generalize
# across arbitrary intervening verbs, clauses, or independently negated nouns.
NO_NEW_PEDESTRIANS = rx(r"\bno\s+new\s+(?:dynamic\s+)?obstacles?\s+or\s+(?P<entity>pedestrians?)\b")
SPATIAL = rx(r"\b(?:front[ _-]left|front[ _-]right|back[ _-]left|back[ _-]right|front|back|rear|left|right|ahead|behind|sidewalk|sidewalks|both sides|opposite direction|distance|closer|further away|nearby|adjacent|centrally|center|on the sides)\b")
MOTION = [
    ("靠近/距离缩短", rx(r"\b(?:approach\w*|closer|decreas\w*\s+(?:in\s+)?distance|distance\s+(?:decreas\w*|reduces?))\b")),
    ("远离/距离增大", rx(r"\b(?:further away|moving away|moves away|move away|more distant|receding|increas\w*\s+(?:in\s+)?distance)\b")),
    ("横穿/横向移动", rx(r"\b(?:cross(?:es|ing|ed)?|left to right|right to left|lateral movement)\b")),
    ("转向/换道", rx(r"\b(?:turn\w*|lane changes?|chang\w* lanes|merg\w*|overtak\w*)\b")),
    ("静止/停放", rx(r"\b(?:stationary|static|parked|stopped|remain\w* (?:constant|stable|in place)|no (?:significant )?movement)\b")),
    ("向前/向后运动", rx(r"\b(?:forward|backward|moving|moves|move|walking|travel\w*)\b")),
    ("速度/停车", rx(r"\b(?:speed|slow\w*|accelerat\w*|decelerat\w*|stop\w*)\b")),
]

def uniq(items):
    return list(dict.fromkeys(items))

def overlapping(start, end, spans):
    return any(start < b and end > a for a, b in spans)

def sentence_bounds(text, start, end):
    # Decimal numbers are not present in the frame-based predictions. Dots within
    # category names are ignored unless followed by whitespace/end-of-string.
    bounds = [(m.start(), m.end()) for m in re.finditer(r"[.!?;](?:\s+|$)", text)]
    left = max((b for a, b in bounds if b <= start), default=0)
    right = min((a for a, b in bounds if a >= end), default=len(text))
    return left, right

def assertion(text, start, end):
    left, right = sentence_bounds(text, start, end)
    sentence = text[left:right]
    new_match = next((m for m in NO_NEW_PEDESTRIANS.finditer(text) if m.span("entity") == (start, end)), None)
    if new_match and rx(r"\b(?:if|unless|whether|may|might|could|would)\b").search(text[left:new_match.start()]):
        return "条件或可能", sentence.strip(), "条件/可能性中的无新增行人描述"
    if new_match:
        return "否定新增", sentence.strip(), "否定新增行人，未否定已有行人"
    rel = start - left
    prefix = sentence[:rel]
    suffix = sentence[end-left:]
    # Response-leading "No," answers the question and is not entity negation.
    prefix = re.sub(r"^\s*(?:The answer is\s+)?No,\s*", "", prefix, flags=re.I)
    breaks = list(INDEPENDENT_BREAK.finditer(prefix))
    local = prefix[breaks[-1].end():] if breaks else prefix
    # "road hazards/road conditions/pedestrian activity" names an attribute or
    # topic; absence of that attribute cannot establish absence of its class.
    if rx(r"^\s+(?:hazards?|conditions?|safety|activity|behavior|density|discipline|restrictions?|closures?|users?|crossings)\b").search(suffix) or (NOUN_NEG.search(prefix) and rx(r"^\s+changes?\b").search(suffix)):
        return "泛指或建议", sentence.strip(), "仅讨论类别相关属性/情境"
    negatives = list(NOUN_NEG.finditer(local))
    if negatives:
        between = local[negatives[-1].end():]
        # A negated action/risk does not imply an absent entity.
        if not ACTION_OR_RISK_NEG.search(between) and not rx(r"\b(?:on|along|in|from|towards|to|with|through|between|over|under|near|behind|ahead|allowing|requiring|providing|indicating|suggesting)\b").search(between):
            return "否定存在", sentence.strip(), "名词前的否定存在结构"
    # Long coordinated noun lists: "no pedestrians, animals, or vehicles".
    all_negs = list(NOUN_NEG.finditer(prefix))
    if all_negs:
        between = prefix[all_negs[-1].end():]
        if len(between) < 115 and not ACTION_OR_RISK_NEG.search(between) and not rx(r"\b(?:is|are|was|were|with|while|but|visible|detected|seen|remain\w*|on|along|in|from|towards|to|through|between|over|under|near|behind|ahead|allowing|requiring|providing|indicating|suggesting)\b|\band\s+(?:the|a|an)\s").search(between):
            return "否定存在", sentence.strip(), "并列名词中的否定作用域"
    if NEG_POST.search(suffix):
        return "否定存在", sentence.strip(), "名词后的不可见/不存在谓语"
    # "No dynamic objects like pedestrians are detected".
    if rx(r"^\s*(?:or\s+[\w -]+\s*)?(?:are|is)\s+(?:not\s+)?(?:visible|detected|seen)").search(suffix) and all_negs:
        between = prefix[all_negs[-1].end():]
        if (len(between) < 100 and not ACTION_OR_RISK_NEG.search(between)
                and not rx(r"\b(?:but|while|whereas|although|because)\b").search(between)):
            return "否定存在", sentence.strip(), "否定的对象集合"
    # Hypothetical existence is distinct from an existing object's uncertain action.
    if rx(r"\bif\s+(?:a|an|any|some|there\s+(?:is|are))\s+(?:\w+\s+){0,4}$").search(prefix):
        return "条件或可能", sentence.strip(), "if引入的条件对象"
    if rx(r"\b(?:could|might|may|would|possibly|potentially)\s+(?:be|have|include|contain|encounter|appear|see)\s+(?:(?:a|an|the|some|any)\s+)?(?:\w+\s+){0,3}$").search(local):
        return "条件或可能", sentence.strip(), "可能存在的对象"
    if rx(r"\b(?:potential|possible|unexpected)\s+(?:hazards?|obstacles?|objects?)\s+(?:like|such as|including)\s+(?:\w+[ ,]+){0,4}$").search(local):
        return "条件或可能", sentence.strip(), "假设风险中的例举对象"
    if rx(r"\bpotential for\s+(?:obstacles?|objects?)\s+(?:like|such as)\s+(?:\w+[ ,]+){0,4}$").search(local):
        return "条件或可能", sentence.strip(), "潜在对象举例"
    # Bare references to activity do not assert an actual instance.
    if rx(r"^\s+(?:activity|movement|behavior|density)\b").search(suffix) and not EXISTENCE_CUE.search(sentence):
        return "泛指或建议", sentence.strip(), "仅讨论类别活动/行为"
    if rx(r"^\s*(?:The answer is\s+)?(?:The )?driver\s+(?:should|must|needs?|has to)|^\s*(?:Drivers?\s+)?(?:should|must)\b").search(sentence):
        if not rx(r"\b(?:visible|present|seen|detected|observed|presence of|approach\w*|parked|stationary|positioned|crossing|moving|walking|proximity|navigate around|on (?:the )?(?:left|right|sidewalks?)|ahead|behind|in front)\b").search(sentence):
            return "泛指或建议", sentence.strip(), "驾驶建议中的泛指对象"
    # A clear subject ("the bus could move") still asserts the bus as a referent;
    # motion modality is captured separately and does not negate the noun.
    return "肯定提及", sentence.strip(), "直接描写/关系描述中的对象或区域"

def extract(text):
    occupied = []
    mentions = []
    extras = []
    for name, pattern in EXTRA_RULES:
        for m in pattern.finditer(text):
            if overlapping(m.start(), m.end(), occupied):
                continue
            status, context, rationale = assertion(text, m.start(), m.end())
            extras.append(dict(category=name, word=m.group(), status=status,
                               span=[m.start(),m.end()], context=context, rationale=rationale))
            occupied.append(m.span())
    for name, pattern, method in RULES:
        for m in pattern.finditer(text):
            if overlapping(m.start(), m.end(), occupied):
                continue
            # "lane change" is an action phrase; it does not independently
            # establish a road region. Same applies to "roadwork" (word boundary).
            if name == "driveable_surface" and m.group().lower() in ("lane", "lanes") and rx(r"^\s+chang").search(text[m.end():]):
                continue
            status, context, rationale = assertion(text, m.start(), m.end())
            mentions.append(dict(category=name, word=m.group(), status=status,
                                 span=[m.start(),m.end()], context=context,
                                 rationale=rationale, mapping=method))
            occupied.append(m.span())
    mentions.sort(key=lambda v:v["span"][0])
    # A disjunction between distinct categories cannot establish both as present.
    for first, second in zip(mentions, mentions[1:]):
        between = text[first["span"][1]:second["span"][0]]
        if first["category"] != second["category"] and rx(r"^\s*,?\s*(?:or|and/or)\s+(?:(?:a|an|the|other)\s+)?$").fullmatch(between):
            if first["status"] != "否定存在" and second["status"] != "否定存在":
                first["status"] = second["status"] = "条件或可能"
                first["rationale"] = second["rationale"] = "不同类别由or连接，类别选择不确定"
    uncertain = []
    for label, pattern, candidates in AMBIGUOUS_RULES:
        for m in pattern.finditer(text):
            if overlapping(m.start(),m.end(),occupied):
                continue
            status, context, rationale = assertion(text,m.start(),m.end())
            uncertain.append(dict(label=label, word=m.group(), status=status,
                                  candidates=candidates, span=list(m.span()), context=context))
    # A construction-equipment mention followed by an explicit excavator in the
    # same sentence is resolved by that named entity; retain its generic wording
    # as context but do not mark the already-resolved occurrence for review.
    for item in uncertain:
        item["resolved"] = item["label"] == "施工设备" and any(
            v["category"] == "construction_vehicle" and v["context"] == item["context"]
            for v in mentions)
    states = {name:[] for _,name,*_ in CLASSES}
    for m in mentions:
        if m["status"] not in states[m["category"]]:
            states[m["category"]].append(m["status"])
    return mentions, extras, uncertain, states

HEADERS = ["任务类别", "类别内序号", "预测原文", "逐条类别分析", "文本肯定提及类别", "文本否定存在类别", "否定新增类别", "条件或可能类别", "泛指或建议类别", "原始ignore类别", "类别不明的对象或情境", "类别匹配证据", "帧范围原文", "空间关系词", "运动及状态", "风险或驾驶建议", "需复核", "复核原因", "未提及的16类", "原问题", "参考答案"]

def sort_names(names):
    return sorted(names,key=ORDER.__getitem__)

def compact_list(names):
    return "；".join(names) or "无"

def make_record(task,index,text,question,gt):
    mentions, extras, uncertain, states = extract(text)
    by_status = {s:sort_names([n for n,v in states.items() if s in v]) for s in STATUSES}
    positive = by_status["肯定提及"]
    things = [n for n in positive if KINDS[n]=="Thing"]
    stuff = [n for n in positive if KINDS[n]=="Stuff"]
    reasons = []
    for name,v in states.items():
        if "肯定提及" in v and "否定存在" in v:
            reasons.append(f"{name}同时有肯定与否定语句，需按对象/帧范围核对")
        if "条件或可能" in v:
            reasons.append(f"{name}的存在或类别含条件/可能性")
    for u in uncertain:
        if u["label"] in ("施工设备","bike/rider/scooter泛称","ground/pavement泛称") and not u["resolved"]:
            reasons.append(f"{u['word']}：{u['candidates']}")
    frame_only = not mentions and FRAME_ONLY.fullmatch(text) is not None
    if not mentions and not frame_only:
        reasons.append("预测未明确给出官方16类；不能借用问题或参考答案补全")
    if any(m["mapping"].startswith("语义关联，草地") for m in mentions):
        reasons.append("草地/地表按terrain语义关联；高度和真实点标签未核对")
    if by_status["泛指或建议"]:
        reasons.append("泛指或驾驶建议不能单独作为实例存在证据")
    if any(m["word"].lower() in ("cyclist","cyclists","bicyclist","bicyclists","motorcyclist","motorcyclists") for m in mentions):
        reasons.append("骑行者关联到bicycle/motorcycle，未额外推断pedestrian")
    frames = uniq(m.group() for m in FRAME.finditer(text))
    space = uniq(m.group().lower() for m in SPATIAL.finditer(text))
    movement = [n for n,p in MOTION if p.search(text)]
    risk = "是" if ADVICE.search(text) else "否"
    if things and stuff:
        relation = "对象与环境区域"
    elif things:
        relation = "仅对象类别"
    elif stuff:
        relation = "仅环境区域"
    elif mentions:
        relation = "仅否定/条件/泛指类别"
    elif extras:
        relation = "仅原始ignore类别"
    else:
        relation = "无明确分割类别"
    analysis = []
    if things: analysis.append("对象："+"、".join(f"{n}（{CN[n]}）" for n in things))
    if stuff: analysis.append("区域："+"、".join(f"{n}（{CN[n]}）" for n in stuff))
    for status,prefix in [("否定存在","否定存在"),("否定新增","否定新增"),("条件或可能","条件/可能"),("泛指或建议","泛指/建议")]:
        if by_status[status]: analysis.append(prefix+"："+"、".join(by_status[status]))
    if extras:
        extra_names = uniq(f"{m['category']}（{m['status']}）" for m in extras)
        analysis.append("原始ignore："+"、".join(extra_names))
    if not mentions: analysis.append("未明确给出官方16类，无法仅从预测确定具体分割类别")
    hard_uncertain = uniq(u["word"] for u in uncertain if u["label"] not in ("仅环境描述",) and not u["resolved"])
    if hard_uncertain: analysis.append("泛称/待定："+"、".join(hard_uncertain))
    if frames: analysis.append("时间："+"、".join(frames))
    if movement: analysis.append("关系/属性："+"、".join(movement))
    if risk=="是": analysis.append("含风险判断/驾驶建议；该部分不是分割类别")
    evidence = uniq(f"{m['category']}←{m['word']}[{m['status']}]" for m in mentions)
    row = [task,index,text,"；".join(analysis),*(compact_list(by_status[s]) for s in STATUSES),
           compact_list(uniq(f"{m['category']}[{m['status']}]" for m in extras)),
           compact_list(uniq(f"{u['word']}→{u['candidates']}" for u in uncertain if u["label"]!="仅环境描述")),
           compact_list(evidence),compact_list(frames),compact_list(space),compact_list(movement),risk,
           "是" if reasons else "否",compact_list(uniq(reasons)),
           compact_list(n for _,n,*_ in CLASSES if not states[n]),question,gt]
    return dict(task=task,index=index,prediction=text,row=row,mentions=mentions,extras=extras,
                uncertain=uncertain,states=states,positive_things=things,positive_stuff=stuff,
                relation=relation,frames=frames,movement=movement,risk=risk,
                 review_reasons=uniq(reasons),
                 auto_decision="纯帧范围短答：原文无类别名称，目标指代仍未知" if frame_only else "")

def self_check():
    # Contrastive, meaningful language cases for the requested presence analysis.
    cases = [
        ("No, the barriers remain static.", {"barrier":["肯定提及"]}),
        ("There are no pedestrians, animals, or other vehicles nearby.", {"pedestrian":["否定存在"]}),
        ("The pedestrians do not move.", {"pedestrian":["肯定提及"]}),
        ("There is no significant movement from the pedestrians.", {"pedestrian":["肯定提及"]}),
        ("If a pedestrian crosses the road, the driver should stop.", {"pedestrian":["条件或可能"]}),
        ("The pedestrian is visible on the sidewalk and could cross the road.", {"pedestrian":["肯定提及"]}),
        ("The ego vehicle moves forward.", {"car":[]}),
        ("A bus stop is next to a bicycle rack.", {"bus":[],"bicycle":[],"manmade":["肯定提及"]}),
        ("A construction zone is visible.", {"construction_vehicle":[]}),
        ("A construction vehicle and an SUV are parked.", {"construction_vehicle":["肯定提及"],"car":["肯定提及"]}),
        ("A car or truck is ahead.", {"car":["条件或可能"],"truck":["条件或可能"]}),
        ("A truck is stationary or moving slowly.", {"truck":["肯定提及"]}),
        ("Yes.", {"car":[],"pedestrian":[]}),
        ("There are no pedestrians on the road.", {"pedestrian":["否定存在"],"driveable_surface":["肯定提及"]}),
        ("No significant road hazards are noted.", {"driveable_surface":["泛指或建议"]}),
        ("The road is clear with no immediate obstacles, indicating a route through the urban street.", {"driveable_surface":["肯定提及"]}),
        ("The pedestrian's movement across the road requires the driver to be cautious.", {"pedestrian":["肯定提及"],"driveable_surface":["肯定提及"]}),
        ("The scene is static with no pedestrians or moving vehicles, and the road is clear.", {"pedestrian":["否定存在"],"driveable_surface":["肯定提及"]}),
        ("No significant terrain changes are observed.", {"terrain":["泛指或建议"]}),
        ("No detected active road users are present.", {"driveable_surface":["泛指或建议"]}),
        ("A pedestrian crossing the road is visible.", {"pedestrian":["肯定提及"]}),
        ("The car changes lanes.", {"car":["肯定提及"]}),
    ]
    for text,expected in cases:
        states = extract(text)[3]
        for name,want in expected.items():
            assert states[name] == want,(text,name,states[name],want)

def write_csv(path,headers,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(headers)
        w.writerows(rows)


def review_flags(record):
    """Return overlapping, human-readable causes, not category labels."""
    reasons = record["review_reasons"]
    flags = []
    if any("同时有肯定与否定" in s for s in reasons): flags.append("肯否定并存")
    if any("存在或类别含条件" in s for s in reasons): flags.append("条件或类别选择")
    if any(u["label"] == "施工设备" and not u["resolved"] for u in record["uncertain"]): flags.append("施工设备泛称")
    if any(u["label"] in ("bike/rider/scooter泛称", "ground/pavement泛称") and not u["resolved"] for u in record["uncertain"]): flags.append("两轮或地面泛称")
    if any("草地/地表" in s for s in reasons): flags.append("草地高度及标签")
    if any("骑行者关联" in s for s in reasons): flags.append("骑行者关联")
    if any("泛指或驾驶建议" in s for s in reasons): flags.append("泛指或驾驶建议")
    if any("预测未明确给出官方16类" in s for s in reasons): flags.append("无明确16类")
    if record.get("auto_decision"): flags.append("纯帧范围但目标未知")
    if reasons and not flags:
        raise ValueError(f"未归类的复核原因: {record['task']} #{record['index']}: {reasons}")
    return flags


def primary_stratum(record):
    flags = review_flags(record)
    for flag in ("肯否定并存", "条件或类别选择", "施工设备泛称", "两轮或地面泛称", "草地高度及标签", "骑行者关联", "泛指或驾驶建议"):
        if flag in flags:
            return flag
    if "纯帧范围但目标未知" in flags:
        return "纯帧范围但目标未知"
    if "无明确16类" in flags:
        labels = {u["label"] for u in record["uncertain"] if not u["resolved"]}
        if "vehicle泛称" in labels: return "无16类：车辆泛称"
        if "obstacle/object泛称" in labels: return "无16类：对象泛称"
        if "施工区域" in labels or "structure泛称" in labels: return "无16类：场景或结构泛称"
        if re.fullmatch(r"\s*(?:yes|no)\s*[.!]?\s*", record["prediction"], re.I): return "无16类：是非短答"
        if rx(r"\b(?:they|them|their|it|its|these|those)\b").search(record["prediction"]): return "无16类：代词指代"
        if any(e["category"] == "vehicle.ego" for e in record["extras"]): return "无16类：自车或关系"
        return "无16类：其他表述"
    raise ValueError(f"不能为记录分层: {record['task']} #{record['index']}")


def review_instruction(stratum):
    if stratum == "肯否定并存": return "检查两个语句的对象、时间和位置范围；保留限定条件，不合并为全场景结论。"
    if stratum == "条件或类别选择": return "核对假设或 or 的作用范围；不要把候选类别都当作已存在。"
    if stratum == "施工设备泛称": return "核对原文是否给出设备形态；没有足够证据时维持候选。"
    if stratum == "两轮或地面泛称": return "核对交通工具类型或地表用途；词义不够明确时维持候选。"
    if stratum == "草地高度及标签": return "核对地表高度与真实点级标签；仅有文本时不可确认。"
    if stratum == "骑行者关联": return "核对骑行者与车辆的关联边界；不要自动增加独立行人实例。"
    if stratum == "泛指或驾驶建议": return "区分情境、建议与明确对象；建议本身不是对象存在证据。"
    if stratum == "纯帧范围但目标未知": return "确认这只是时间范围短答；目标指代需上下文，不能据此补出类别或真值。"
    return "核对是否漏掉明确类别词或存在未解析指代；原问题仅供理解指代，不能替代预测的存在证据。"


def review_outputs(root, baseline, records):
    old = {(r["task"], r["index"]): r for r in baseline}
    new = {(r["task"], r["index"]): r for r in records}
    prior = [r for r in baseline if r["review_reasons"]]
    current = [r for r in records if r["review_reasons"]]
    # A frame-only decision resolves the text-level review, not its missing target.
    pending = [r for r in records if r["review_reasons"] or r["auto_decision"]]
    old_ids = {(r["task"], r["index"]) for r in prior}
    current_ids = {(r["task"], r["index"]) for r in current}
    pending_ids = {(r["task"], r["index"]) for r in pending}

    cluster_rows = []
    combo_rows = []
    for phase, rows in (("原始review", prior), ("重新分析review", current), ("人工抽查候选（含目标未知短答）", pending)):
        for task in TASKS:
            subset = [r for r in rows if r["task"] == task]
            counts = collections.Counter(f for r in subset for f in review_flags(r))
            cluster_rows.extend([phase, task, flag, n] for flag, n in sorted(counts.items()))
            combos = collections.Counter(tuple(review_flags(r)) for r in subset)
            combo_rows.extend([phase, task, "；".join(flags), n] for flags, n in sorted(combos.items()))
    write_csv(root/"复核触发原因.csv", ["阶段", "任务", "触发原因（可重叠）", "记录数"], cluster_rows)
    write_csv(root/"复核触发组合.csv", ["阶段", "任务", "原因组合（互斥）", "记录数"], combo_rows)

    changed = []
    for key, r in new.items():
        before = old[key]
        if before["states"] != r["states"] or before["review_reasons"] != r["review_reasons"]:
            if r["auto_decision"]:
                decision = r["auto_decision"]
            elif any(m["status"] == "否定新增" for m in r["mentions"]):
                decision = "限定否定作用域：无新增行人"
            else:
                decision = "独立分句边界：否定不跨转折分句"
            changed.append([r["task"],r["index"],r["prediction"],
                json.dumps({k:v for k,v in before["states"].items() if v},ensure_ascii=False),
                json.dumps({k:v for k,v in r["states"].items() if v},ensure_ascii=False),
                compact_list(before["review_reasons"]),compact_list(r["review_reasons"]),
                decision])
    write_csv(root/"规则变化记录.csv", ["任务","序号","预测原文","旧类别状态","新类别状态","旧复核原因","新复核原因","判定及边界"], changed)

    groups = {}
    for r in pending:
        stratum = primary_stratum(r)
        question = r["row"][-2]
        key = (r["task"], stratum, r["prediction"], question)
        groups.setdefault(key, []).append(r)
    by_stratum = collections.defaultdict(list)
    for key, group in groups.items():
        digest = hashlib.sha256(json.dumps(key,ensure_ascii=False,separators=(",", ":")).encode("utf-8")).hexdigest()
        by_stratum[key[:2]].append((digest, key, group))
    sample = []
    strata = []
    group_rows = []
    for (task, stratum), items in sorted(by_stratum.items()):
        # Always include a most-frequent exact (answer, question) group, plus at
        # most two distinct groups chosen by deterministic SHA-256 order.
        common = sorted(items, key=lambda item: (-len(item[2]), item[0]))[0]
        selected = [common] + [item for item in sorted(items) if item[0] != common[0]][:2]
        chosen = {item[0] for item in selected}
        strata.append([task,stratum,sum(len(v[2]) for v in items),len(items),len(selected),sum(len(v[2]) for v in selected)])
        for digest, key, group in sorted(items):
            ids = [r["index"] for r in group]
            group_rows.append([task,stratum,digest,len(ids),"；".join(map(str,ids)),"是" if digest in chosen else "否"])
        for digest, key, group in selected:
            r = group[0]
            before = old[(task,r["index"])]
            evidence = [f"{m['word']}→{m['category']}[{m['status']}]" for m in r["mentions"]]
            evidence += [f"{u['word']}→{u['candidates']}" for u in r["uncertain"] if not u["resolved"]]
            sample.append([task,stratum,digest,sum(len(v[2]) for v in items),len(items),len(group),
                "；".join(str(x["index"]) for x in group),r["prediction"],key[3],
                compact_list(before["review_reasons"]),compact_list(r["review_reasons"]),
                compact_list(evidence),review_instruction(stratum),"",""])
    write_csv(root/"人工抽查分层汇总.csv", ["任务","互斥主层","层内记录数","按原文及问题分组数","抽查组数","抽中组包含记录数"], strata)
    write_csv(root/"人工抽查分组索引.csv", ["任务","互斥主层","分组SHA256","组内记录数","任务内序号","抽中"], group_rows)
    write_csv(root/"人工抽查清单.csv", ["任务","互斥主层","分组SHA256","层内记录数","层内分组数","本组记录数","任务内序号","预测原文","原问题（仅辅助理解指代）","原触发原因","重新分析复核原因","原文命中与候选","检查项","人工判定","备注"], sample)
    return {"baseline_review":len(prior), "current_review":len(current),
            "pending_context_including_frame_only":len(pending), "changed_records":len(changed),
            "baseline_review_cleared":len(old_ids-current_ids), "new_review":len(current_ids-old_ids),
            "baseline_review_still_sample_eligible":len(old_ids & pending_ids),
            "sample_groups":len(sample), "sample_covered_records":sum(row[5] for row in sample),
            "sampling":"每个任务×互斥主层取最高频的原文及原问题完全相同组，再按SHA256顺序取至多两组；不用于估计错误率"}


def main():
    parser = argparse.ArgumentParser(description="复核历史B3预测的文本类别关联；不生成伪掩码。")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    source_path = args.source.resolve()
    baseline_paths = (args.baseline_summary.resolve(), args.baseline_records.resolve())
    if root in (source_path, *baseline_paths) or any(root in p.parents for p in (source_path, *baseline_paths)):
        parser.error("输出目录不能包含源文件或基线文件")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        parser.error("输出目录必须不存在或为空，避免覆盖已有结果")
    self_check()
    raw=source_path.read_bytes()
    source_sha=hashlib.sha256(raw).hexdigest()
    old_summary=json.loads(baseline_paths[0].read_text(encoding="utf-8-sig"))
    if old_summary["source_sha256"] != source_sha:
        raise ValueError("预测源文件与基线summary的SHA-256不一致")
    with baseline_paths[1].open(encoding="utf-8") as f:
        baseline=[json.loads(line) for line in f]
    source=json.loads(raw.decode("utf-8-sig"))
    records=[]
    for task in TASKS:
        group=source[task]
        assert len(group["predictions"])==len(group["questions"])==len(group["ground_truths"])
        assert all(isinstance(s,str) for s in group["predictions"])
        for i,(p,q,gt) in enumerate(zip(group["predictions"],group["questions"],group["ground_truths"]),1):
            records.append(make_record(task,i,p,q,gt))
    if len(baseline) != len(records) or len(records) != old_summary["total"]:
        raise ValueError("源文件和基线记录数量不一致")
    for before, after in zip(baseline, records):
        if (before["task"],before["index"],before["prediction"],before["row"][-2:]) != (after["task"],after["index"],after["prediction"],after["row"][-2:]):
            raise ValueError(f"源文件和基线记录顺序或内容不一致: {after['task']} #{after['index']}")
    if any(sum(bool(r["review_reasons"]) for r in baseline if r["task"]==task) != old_summary["tasks"][task]["review"] for task in TASKS):
        raise ValueError("基线逐条review与summary不一致")
    root.mkdir(parents=True, exist_ok=True)
    summary=dict(source_file=str(source_path),source_sha256=source_sha,
                 baseline_summary_sha256=hashlib.sha256(baseline_paths[0].read_bytes()).hexdigest(),
                 baseline_records_sha256=hashlib.sha256(baseline_paths[1].read_bytes()).hexdigest(),
                 protocol="text-only-taxonomy-association-v2",date=date.today().isoformat(),tasks={},classes={},
                 total=len(records),sources=SOURCES)
    for task in TASKS:
        rs=[r for r in records if r["task"]==task]
        counts=dict(total=len(rs),unique_predictions=len({r["prediction"] for r in rs}),
                    with_official_mention=sum(bool(r["mentions"]) for r in rs),
                    with_positive=sum(any("肯定提及" in v for v in r["states"].values()) for r in rs),
                    with_positive_thing=sum(bool(r["positive_things"]) for r in rs),
                    with_positive_stuff=sum(bool(r["positive_stuff"]) for r in rs),
                    with_both=sum(bool(r["positive_things"] and r["positive_stuff"]) for r in rs),
                    with_negated=sum(any("否定存在" in v for v in r["states"].values()) for r in rs),
                    with_negated_new=sum(any("否定新增" in v for v in r["states"].values()) for r in rs),
                    with_conditional=sum(any("条件或可能" in v for v in r["states"].values()) for r in rs),
                    with_generic=sum(any("泛指或建议" in v for v in r["states"].values()) for r in rs),
                    with_frames=sum(bool(r["frames"]) for r in rs),
                    with_risk=sum(r["risk"]=="是" for r in rs),
                    review=sum(bool(r["review_reasons"]) for r in rs),
                    no_official_mention=sum(not r["mentions"] for r in rs))
        summary["tasks"][task]=counts
        summary["tasks"][task]["relations"]=dict(collections.Counter(r["relation"] for r in rs))
        write_csv(root/f"{task}_逐条分析.csv",HEADERS,(r["row"] for r in rs))
    class_rows=[]
    for cid,name,cn,kind,rawnames in CLASSES:
        entry={}
        for task in TASKS:
            rs=[r for r in records if r["task"]==task]
            vals={s:sum(s in r["states"][name] for r in rs) for s in STATUSES}
            vals["任意提及"]=sum(bool(r["states"][name]) for r in rs)
            entry[task]=vals
            class_rows.append([task,cid,name,cn,kind,len(rs),vals["任意提及"],vals["肯定提及"],vals["否定存在"],vals["否定新增"],vals["条件或可能"],vals["泛指或建议"],vals["肯定提及"]/len(rs)])
        summary["classes"][name]=entry
    write_csv(root/"全部预测_逐条类别对应.csv",HEADERS,(r["row"] for r in records))
    write_csv(root/"类别频次汇总.csv",["任务类别","评测类别编号","类别","中文含义","Thing或Stuff","总预测数","任意提及条数","肯定提及条数","否定存在条数","否定新增条数","条件或可能条数","泛指或建议条数","肯定提及比例"],class_rows)
    write_csv(root/"16类存在关系矩阵.csv",["任务类别","类别内序号"]+[n for _,n,*_ in CLASSES],([r["task"],r["index"]]+["/".join(r["states"][n]) or "未提及" for _,n,*_ in CLASSES] for r in records))
    write_csv(root/"需复核预测.csv",HEADERS,(r["row"] for r in records if r["review_reasons"]))
    with (root/"逐条分析.jsonl").open("w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
    summary["review_analysis"] = review_outputs(root, baseline, records)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("运行期间预测源文件发生变化")
    (root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"total":len(records),"review_analysis":summary["review_analysis"],"tasks":summary["tasks"],"sha256":source_sha},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
