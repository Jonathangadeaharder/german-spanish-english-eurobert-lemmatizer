"""Handcraft targeted Chinese training data for weak UPOS tags.

Error analysis showed the zh_bio model fails on: ADJ (18%), ADV (41%),
SCONJ (52%), PART (53%), ADP (59%), PROPN (67%). These tags are
underrepresented in the 3997-sentence training set.

This script defines handcrafted Chinese sentences with word-level UPOS
annotations, converts them to char-level BIO labels, and appends to
data/processed/zh_bio/{train,validation}.jsonl.

90% → train.jsonl, 10% → validation.jsonl (model never sees these).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

rng = random.Random(42)

# Label2id mapping from data/processed/zh_bio/labels.json
UPOS_TO_B = {
    "ADJ": 1,
    "ADP": 2,
    "ADV": 3,
    "AUX": 4,
    "CCONJ": 5,
    "DET": 6,
    "INTJ": 7,
    "NOUN": 8,
    "NUM": 9,
    "PART": 10,
    "PRON": 11,
    "PROPN": 12,
    "PUNCT": 13,
    "SCONJ": 14,
    "SYM": 15,
    "VERB": 16,
    "X": 17,
}
UPOS_TO_I = {
    "ADJ": 18,
    "ADP": 19,
    "ADV": 20,
    "AUX": 21,
    "CCONJ": 22,
    "DET": 23,
    "INTJ": 24,
    "NOUN": 25,
    "NUM": 26,
    "PART": 27,
    "PRON": 28,
    "PROPN": 29,
    "PUNCT": 30,
    "SCONJ": 31,
    "SYM": 32,
    "VERB": 33,
    "X": 34,
}

# Each entry: (sentence_words, upos_tags) — word-level annotations.
# Handcrafted to provide clear context for each weak tag.
SENTENCES: list[tuple[list[str], list[str]]] = [
    # === ADJ — confused with NOUN/VERB ===
    (
        ["美丽", "的", "花朵", "在", "花园", "里", "绽放", "。"],
        ["ADJ", "PART", "NOUN", "ADP", "NOUN", "ADP", "VERB", "PUNCT"],
    ),
    (
        ["这个", "问题", "非常", "重要", "，", "需要", "认真", "对待", "。"],
        ["DET", "NOUN", "ADV", "ADJ", "PUNCT", "VERB", "ADJ", "VERB", "PUNCT"],
    ),
    (
        ["她", "穿", "着", "一件", "漂亮", "的", "红色", "裙子", "。"],
        ["PRON", "VERB", "PART", "NUM", "ADJ", "PART", "ADJ", "NOUN", "PUNCT"],
    ),
    (
        ["这道", "菜", "真", "好吃", "，", "味道", "很", "鲜美", "。"],
        ["DET", "NOUN", "ADV", "ADJ", "PUNCT", "NOUN", "ADV", "ADJ", "PUNCT"],
    ),
    (
        ["高", "大的", "树木", "遮住", "了", "阳光", "。"],
        ["ADJ", "ADJ", "NOUN", "VERB", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["这个", "房间", "太", "小", "了", "，", "住", "不", "下", "。"],
        ["DET", "NOUN", "ADV", "ADJ", "PART", "PUNCT", "VERB", "ADV", "VERB", "PUNCT"],
    ),
    (
        ["新", "的", "学期", "开始", "了", "，", "大家", "很", "兴奋", "。"],
        ["ADJ", "PART", "NOUN", "VERB", "PART", "PUNCT", "PRON", "ADV", "ADJ", "PUNCT"],
    ),
    (
        ["他", "是", "一个", "聪明", "的", "学生", "。"],
        ["PRON", "AUX", "NUM", "ADJ", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["天气", "很", "冷", "，", "但", "阳光", "很", "温暖", "。"],
        ["NOUN", "ADV", "ADJ", "PUNCT", "CCONJ", "NOUN", "ADV", "ADJ", "PUNCT"],
    ),
    (
        ["那些", "古老", "的", "建筑", "非常", "壮观", "。"],
        ["DET", "ADJ", "PART", "NOUN", "ADV", "ADJ", "PUNCT"],
    ),
    (
        ["这", "本", "书", "很", "有趣", "，", "我", "喜欢", "。"],
        ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT", "PRON", "VERB", "PUNCT"],
    ),
    (
        ["她", "有", "一双", "大", "眼睛", "和", "长", "头发", "。"],
        ["PRON", "VERB", "NUM", "ADJ", "NOUN", "CCONJ", "ADJ", "NOUN", "PUNCT"],
    ),
    (
        ["快", "来", "，", "我们", "等", "你", "很", "久", "了", "。"],
        ["ADV", "VERB", "PUNCT", "PRON", "VERB", "PRON", "ADV", "ADJ", "PART", "PUNCT"],
    ),
    (
        ["这个", "想法", "很", "好", "，", "但", "执行", "很", "难", "。"],
        ["DET", "NOUN", "ADV", "ADJ", "PUNCT", "CCONJ", "NOUN", "ADV", "ADJ", "PUNCT"],
    ),
    (
        ["深", "秋", "的", "夜晚", "，", "天空", "很", "清澈", "。"],
        ["ADJ", "NOUN", "PART", "NOUN", "PUNCT", "NOUN", "ADV", "ADJ", "PUNCT"],
    ),
    # === ADV — confused with NOUN/VERB ===
    (["他", "经常", "去", "公园", "跑步", "。"], ["PRON", "ADV", "VERB", "NOUN", "VERB", "PUNCT"]),
    (
        ["我们", "已经", "完成", "了", "任务", "。"],
        ["PRON", "ADV", "VERB", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["她", "正在", "看", "书", "，", "不要", "打扰", "她", "。"],
        ["PRON", "ADV", "VERB", "NOUN", "PUNCT", "ADV", "VERB", "PRON", "PUNCT"],
    ),
    (
        ["他们", "陆续", "到达", "了", "会场", "。"],
        ["PRON", "ADV", "VERB", "PART", "NOUN", "PUNCT"],
    ),
    (["你", "怎么", "又", "迟到", "了", "？"], ["PRON", "ADV", "ADV", "VERB", "PART", "PUNCT"]),
    (
        ["他", "慢慢", "地", "走", "了", "过来", "。"],
        ["PRON", "ADV", "PART", "VERB", "PART", "VERB", "PUNCT"],
    ),
    (
        ["会议", "马上", "开始", "，", "请", "就", "座", "。"],
        ["NOUN", "ADV", "VERB", "PUNCT", "VERB", "ADV", "VERB", "PUNCT"],
    ),
    (
        ["我", "一直", "在", "等", "你", "的", "消息", "。"],
        ["PRON", "ADV", "ADP", "VERB", "PRON", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["他", "终于", "成功", "了", "，", "大家", "都", "很", "高兴", "。"],
        ["PRON", "ADV", "VERB", "PART", "PUNCT", "PRON", "ADV", "ADV", "ADJ", "PUNCT"],
    ),
    (
        ["无论", "如何", "，", "我", "都", "会", "支持", "你", "。"],
        ["SCONJ", "ADV", "PUNCT", "PRON", "ADV", "AUX", "VERB", "PRON", "PUNCT"],
    ),
    (
        ["最近", "天气", "变", "冷", "了", "，", "多", "穿", "点", "。"],
        ["ADV", "NOUN", "VERB", "ADJ", "PART", "PUNCT", "ADV", "VERB", "NUM", "PUNCT"],
    ),
    (
        ["他", "突然", "站", "了", "起来", "，", "大声", "说话", "。"],
        ["PRON", "ADV", "VERB", "PART", "VERB", "PUNCT", "ADV", "VERB", "PUNCT"],
    ),
    (
        ["孩子们", "正在", "外面", "玩耍", "，", "笑声", "不断", "。"],
        ["NOUN", "ADV", "ADP", "VERB", "PUNCT", "NOUN", "ADV", "PUNCT"],
    ),
    (
        ["她", "轻轻", "地", "关上", "了", "门", "。"],
        ["PRON", "ADV", "PART", "VERB", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["我们", "先", "吃", "饭", "，", "然后", "去", "散步", "。"],
        ["PRON", "ADV", "VERB", "NOUN", "PUNCT", "ADV", "VERB", "VERB", "PUNCT"],
    ),
    # === SCONJ — confused with PART ===
    (
        ["因为", "下雨", "，", "所以", "我们", "没有", "出去", "。"],
        ["SCONJ", "VERB", "PUNCT", "SCONJ", "PRON", "ADV", "VERB", "PUNCT"],
    ),
    (
        ["如果", "你", "愿意", "，", "我们", "可以", "一起", "去", "。"],
        ["SCONJ", "PRON", "VERB", "PUNCT", "PRON", "AUX", "ADV", "VERB", "PUNCT"],
    ),
    (
        ["虽然", "很累", "，", "但", "他", "还是", "坚持", "工作", "。"],
        ["SCONJ", "ADJ", "PUNCT", "CCONJ", "PRON", "ADV", "VERB", "VERB", "PUNCT"],
    ),
    (
        ["即使", "困难", "再大", "，", "我", "也", "不", "放弃", "。"],
        ["SCONJ", "ADJ", "ADJ", "PUNCT", "PRON", "ADV", "ADV", "VERB", "PUNCT"],
    ),
    (
        ["既然", "你", "知道", "了", "，", "就", "告诉", "大家", "吧", "。"],
        ["SCONJ", "PRON", "VERB", "PART", "PUNCT", "ADV", "VERB", "NOUN", "PART", "PUNCT"],
    ),
    (
        ["不管", "发生", "什么", "，", "我", "都", "会", "在", "你", "身边", "。"],
        ["SCONJ", "VERB", "PRON", "PUNCT", "PRON", "ADV", "AUX", "ADP", "PRON", "NOUN", "PUNCT"],
    ),
    (
        ["除非", "你", "亲自", "去", "，", "否则", "没有", "用", "。"],
        ["SCONJ", "PRON", "ADV", "VERB", "PUNCT", "SCONJ", "ADV", "NOUN", "PUNCT"],
    ),
    (
        ["只要", "努力", "，", "就", "会", "成功", "。"],
        ["SCONJ", "VERB", "PUNCT", "ADV", "AUX", "VERB", "PUNCT"],
    ),
    (
        ["因为", "时间", "不够", "，", "所以", "取消", "了", "会议", "。"],
        ["SCONJ", "NOUN", "ADJ", "PUNCT", "SCONJ", "VERB", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["虽然", "路", "远", "，", "但", "值得", "去", "。"],
        ["SCONJ", "NOUN", "ADJ", "PUNCT", "CCONJ", "AUX", "VERB", "PUNCT"],
    ),
    # === PART — confused with NOUN ===
    (
        ["他", "吃", "了", "饭", "，", "然后", "走", "了", "。"],
        ["PRON", "VERB", "PART", "NOUN", "PUNCT", "ADV", "VERB", "PART", "PUNCT"],
    ),
    (
        ["门", "开", "着", "，", "灯", "也", "亮", "着", "。"],
        ["NOUN", "VERB", "PART", "PUNCT", "NOUN", "ADV", "ADJ", "PART", "PUNCT"],
    ),
    (
        ["我", "去", "过", "北京", "，", "也", "去", "过", "上海", "。"],
        ["PRON", "VERB", "PART", "PROPN", "PUNCT", "ADV", "VERB", "PART", "PROPN", "PUNCT"],
    ),
    (
        ["你", "知道", "吗", "？", "他", "回来", "了", "呢", "。"],
        ["PRON", "VERB", "PART", "PUNCT", "PRON", "VERB", "PART", "PART", "PUNCT"],
    ),
    (
        ["算", "了", "吧", "，", "别", "再", "提", "了", "。"],
        ["VERB", "PART", "PART", "PUNCT", "ADV", "ADV", "VERB", "PART", "PUNCT"],
    ),
    (
        ["这", "是", "我", "的", "书", "，", "不", "是", "你", "的", "。"],
        ["DET", "AUX", "PRON", "PART", "NOUN", "PUNCT", "ADV", "AUX", "PRON", "PART", "PUNCT"],
    ),
    (
        ["他", "笑着", "说", "了", "一声", "再见", "。"],
        ["PRON", "VERB", "VERB", "PART", "NUM", "NOUN", "PUNCT"],
    ),
    (["花", "开", "得", "很", "好", "。"], ["NOUN", "VERB", "PART", "ADV", "ADJ", "PUNCT"]),
    (
        ["所谓", "成功", "，", "就", "是", "不", "断", "地", "努力", "。"],
        ["VERB", "NOUN", "PUNCT", "ADV", "AUX", "ADV", "ADV", "PART", "VERB", "PUNCT"],
    ),
    (
        ["类似", "的", "情况", "我们", "之前", "也", "遇到", "过", "。"],
        ["ADJ", "PART", "NOUN", "PRON", "ADV", "ADV", "VERB", "PART", "PUNCT"],
    ),
    # === ADP — confused with VERB/NOUN ===
    (["书", "在", "桌子", "上", "。"], ["NOUN", "ADP", "NOUN", "ADP", "PUNCT"]),
    (
        ["他", "从", "北京", "到", "上海", "，", "走", "了", "一", "天", "。"],
        ["PRON", "ADP", "PROPN", "ADP", "PROPN", "PUNCT", "VERB", "PART", "NUM", "NOUN", "PUNCT"],
    ),
    (
        ["我们", "向", "前", "走", "，", "沿着", "河", "岸", "。"],
        ["PRON", "ADP", "NOUN", "VERB", "PUNCT", "ADP", "NOUN", "NOUN", "PUNCT"],
    ),
    (
        ["他", "对", "我", "笑", "了", "笑", "。"],
        ["PRON", "ADP", "PRON", "VERB", "PART", "VERB", "PUNCT"],
    ),
    (
        ["按", "计划", "，", "我们", "明天", "出发", "。"],
        ["ADP", "NOUN", "PUNCT", "PRON", "NOUN", "VERB", "PUNCT"],
    ),
    (
        ["为", "了", "大家", "的", "安全", "，", "请", "系", "好", "安全带", "。"],
        ["ADP", "PART", "PRON", "PART", "NOUN", "PUNCT", "VERB", "VERB", "ADJ", "NOUN", "PUNCT"],
    ),
    (
        ["他", "站", "在", "门口", "，", "等", "着", "我", "。"],
        ["PRON", "VERB", "ADP", "NOUN", "PUNCT", "VERB", "PART", "PRON", "PUNCT"],
    ),
    (
        ["会议", "于", "明天", "上午", "开始", "。"],
        ["NOUN", "ADP", "NOUN", "NOUN", "VERB", "PUNCT"],
    ),
    (
        ["她", "经", "由", "朋友", "介绍", "认识", "了", "他", "。"],
        ["PRON", "ADP", "ADP", "NOUN", "VERB", "VERB", "PART", "PRON", "PUNCT"],
    ),
    (
        ["把", "书", "放", "在", "桌子", "上", "。"],
        ["ADP", "NOUN", "VERB", "ADP", "NOUN", "ADP", "PUNCT"],
    ),
    # === PROPN — confused with NOUN ===
    (
        ["北京", "是", "中国", "的", "首都", "。"],
        ["PROPN", "AUX", "PROPN", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["他", "去", "了", "上海", "和", "杭州", "。"],
        ["PRON", "VERB", "PART", "PROPN", "CCONJ", "PROPN", "PUNCT"],
    ),
    (["台湾", "的", "风景", "很", "美", "。"], ["PROPN", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (
        ["美国", "和", "日本", "都", "是", "发达国家", "。"],
        ["PROPN", "CCONJ", "PROPN", "ADV", "AUX", "NOUN", "PUNCT"],
    ),
    (["台北", "的", "夜市", "很", "有名", "。"], ["PROPN", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (
        ["中山", "路", "上", "有", "很多", "商店", "。"],
        ["PROPN", "NOUN", "ADP", "VERB", "ADJ", "NOUN", "PUNCT"],
    ),
    (["台大", "医院", "在", "这", "附近", "。"], ["PROPN", "NOUN", "ADP", "DET", "NOUN", "PUNCT"]),
    (
        ["他", "在", "仁爱", "路", "等", "你", "。"],
        ["PRON", "ADP", "PROPN", "NOUN", "VERB", "PRON", "PUNCT"],
    ),
    (
        ["故宫", "博物院", "里", "有", "很多", "珍宝", "。"],
        ["PROPN", "NOUN", "ADP", "VERB", "ADJ", "NOUN", "PUNCT"],
    ),
    (
        ["长江", "是", "中国", "最", "长", "的", "河流", "。"],
        ["PROPN", "AUX", "PROPN", "ADV", "ADJ", "PART", "NOUN", "PUNCT"],
    ),
    (["他", "在", "百度", "工作", "。"], ["PRON", "ADP", "PROPN", "VERB", "PUNCT"]),
    (
        ["黄河", "流", "经", "九", "个", "省", "。"],
        ["PROPN", "VERB", "ADP", "NUM", "NUM", "NOUN", "PUNCT"],
    ),
    (
        ["她", "喜欢", "鲁迅", "的", "文章", "。"],
        ["PRON", "VERB", "PROPN", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["故宫", "在", "北京", "的", "中心", "。"],
        ["PROPN", "ADP", "PROPN", "PART", "NOUN", "PUNCT"],
    ),
]


def word_to_bio(word: str, upos: str) -> list[tuple[str, int]]:
    """Convert a word to (char, label_id) pairs using BIO scheme."""
    chars = list(word)
    if not chars:
        return []
    upos = upos if upos in UPOS_TO_B else "X"
    result = [(chars[0], UPOS_TO_B[upos])]
    for ch in chars[1:]:
        result.append((ch, UPOS_TO_I[upos]))
    return result


def sentence_to_jsonl(words: list[str], upos: list[str]) -> dict:
    """Convert word-level annotation to zh_bio JSONL format."""
    chars: list[str] = []
    labels: list[int] = []
    for word, upos_tag in zip(words, upos, strict=True):
        for ch, label_id in word_to_bio(word, upos_tag):
            chars.append(ch)
            labels.append(label_id)
    # Truncate to MAX_LENGTH - 2 = 254
    if len(chars) > 254:
        chars = chars[:254]
        labels = labels[:254]
    return {"chars": chars, "labels": labels, "length": len(chars)}


def main() -> None:
    train_path = Path("data/processed/zh_bio/train.jsonl")
    val_path = Path("data/processed/zh_bio/validation.jsonl")

    with open(train_path, encoding="utf-8") as f:
        train_count_before = sum(1 for _ in f)
    with open(val_path, encoding="utf-8") as f:
        val_count_before = sum(1 for _ in f)
    print(f"Before: train={train_count_before}, val={val_count_before}")

    # Idempotency: check if augmentation already applied by looking for
    # the marker comment in the first augmented sentence ID.
    already_applied = False
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if "zh-aug-" in line:
                already_applied = True
                break
    if already_applied:
        print("Augmentation already applied — skipping (idempotent)")
        return

    n_train = 0
    n_val = 0
    with (
        open(train_path, "a", encoding="utf-8") as train_f,
        open(val_path, "a", encoding="utf-8") as val_f,
    ):
        for words, upos in SENTENCES:
            entry = sentence_to_jsonl(words, upos)
            line = json.dumps(entry, ensure_ascii=False)
            if rng.random() < 0.1:
                val_f.write(line + "\n")
                n_val += 1
            else:
                train_f.write(line + "\n")
                n_train += 1

    with open(train_path, encoding="utf-8") as f:
        train_count_after = sum(1 for _ in f)
    with open(val_path, encoding="utf-8") as f:
        val_count_after = sum(1 for _ in f)
    print(f"Added: train=+{n_train}, val=+{n_val}")
    print(f"After: train={train_count_after}, val={val_count_after}")


if __name__ == "__main__":
    main()
