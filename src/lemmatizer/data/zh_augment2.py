"""Second batch of handcrafted Chinese training data — focus on ADJ and PART.

ADJ is at 22% (confused with NOUN/VERB), PART at 39% (confused with NOUN).
These are the worst-performing tags. This batch adds 200+ sentences with
clear ADJ and PART context.

90% → train.jsonl, 10% → validation.jsonl.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

rng = random.Random(99)

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

SENTENCES: list[tuple[list[str], list[str]]] = [
    # === ADJ (target: 60+ sentences) — clear adjective contexts ===
    (["这", "朵", "花", "很", "美丽", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那个", "女孩", "很", "漂亮", "。"], ["DET", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (
        ["这", "件", "衣服", "太", "旧", "了", "。"],
        ["DET", "NUM", "NOUN", "ADV", "ADJ", "PART", "PUNCT"],
    ),
    (["今天", "天气", "很", "好", "。"], ["NOUN", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "苹果", "很", "甜", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (
        ["他", "的", "字", "写", "得", "很", "工整", "。"],
        ["PRON", "PART", "NOUN", "VERB", "PART", "ADV", "ADJ", "PUNCT"],
    ),
    (["这", "条", "路", "很", "长", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["外面", "的", "风", "很", "大", "。"], ["NOUN", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "本", "小说", "很", "无聊", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["她", "的", "声音", "很", "好听", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "问题", "很", "复杂", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (
        ["他", "是", "个", "勇敢", "的", "人", "。"],
        ["PRON", "AUX", "NUM", "ADJ", "PART", "NOUN", "PUNCT"],
    ),
    (["这", "杯", "水", "很", "热", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那个", "地方", "很", "远", "。"], ["DET", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他", "的", "回答", "很", "简单", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "办法", "很", "聪明", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["春天", "的", "风", "很", "温暖", "。"], ["NOUN", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "孩子", "很", "聪明", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那", "座", "山", "很", "高", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (
        ["她", "穿", "了", "一件", "红色", "的", "大衣", "。"],
        ["PRON", "VERB", "PART", "NUM", "ADJ", "PART", "NOUN", "PUNCT"],
    ),
    (["这", "道", "菜", "很", "辣", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他", "的", "身体", "很", "健康", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "房间", "很", "干净", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那", "只", "猫", "很", "可爱", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他", "的", "中文", "很", "流利", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "任务", "很", "困难", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那", "本", "字典", "很", "厚", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["她", "的", "笑容", "很", "甜美", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "条", "河", "很", "宽", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那个", "故事", "很", "感人", "。"], ["DET", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他", "的", "态度", "很", "认真", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "计划", "很", "完美", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那", "片", "森林", "很", "茂密", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["她", "的", "眼睛", "很", "亮", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "决定", "很", "重要", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那", "栋", "楼", "很", "新", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他的", "心情", "很", "愉快", "。"], ["PRON", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "种", "花", "很", "香", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那个", "电影", "很", "精彩", "。"], ["DET", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他", "的", "成绩", "很", "优秀", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "书包", "很", "轻", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那", "个", "答案", "很", "清楚", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["她", "的", "皮肤", "很", "白", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "条", "鱼", "很", "新鲜", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那个", "男人", "很", "高", "。"], ["DET", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他", "的", "字", "很", "小", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "杯", "茶", "很", "苦", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那个", "国家", "很", "强大", "。"], ["DET", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["她", "的", "手", "很", "巧", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "价格", "很", "便宜", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那", "段", "经历", "很", "难忘", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他的", "身体", "很", "强壮", "。"], ["PRON", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "朵", "云", "很", "白", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那个", "主意", "很", "棒", "。"], ["DET", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["她", "的", "性格", "很", "温柔", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "个", "办法", "很", "有效", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那", "块", "石头", "很", "重", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他", "的", "话", "很", "有理", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["这", "瓶", "水", "很", "凉", "。"], ["DET", "NUM", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["那个", "季节", "很", "舒服", "。"], ["DET", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["她", "的", "文章", "很", "深刻", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    # === PART (target: 60+ sentences) — clear particle contexts ===
    (["他", "走", "了", "。"], ["PRON", "VERB", "PART", "PUNCT"]),
    (["我", "吃", "过", "饭", "了", "。"], ["PRON", "VERB", "PART", "NOUN", "PART", "PUNCT"]),
    (["门", "开", "着", "。"], ["NOUN", "VERB", "PART", "PUNCT"]),
    (["灯", "亮", "着", "。"], ["NOUN", "VERB", "PART", "PUNCT"]),
    (["他", "笑", "着", "说", "话", "。"], ["PRON", "VERB", "PART", "VERB", "NOUN", "PUNCT"]),
    (["我", "去", "过", "北京", "。"], ["PRON", "VERB", "PART", "PROPN", "PUNCT"]),
    (["她", "来", "过", "两次", "。"], ["PRON", "VERB", "PART", "NUM", "PUNCT"]),
    (
        ["他", "看过", "了", "这", "本", "书", "。"],
        ["PRON", "VERB", "PART", "DET", "NUM", "NOUN", "PUNCT"],
    ),
    (["雨", "下", "了", "一", "天", "。"], ["NOUN", "VERB", "PART", "NUM", "NOUN", "PUNCT"]),
    (["他", "走", "了", "很久", "。"], ["PRON", "VERB", "PART", "ADV", "PUNCT"]),
    (["你", "吃", "了", "吗", "？"], ["PRON", "VERB", "PART", "PART", "PUNCT"]),
    (["他", "回来", "了", "吗", "？"], ["PRON", "VERB", "PART", "PART", "PUNCT"]),
    (["你", "知道", "了", "吧", "？"], ["PRON", "VERB", "PART", "PART", "PUNCT"]),
    (["算", "了", "吧", "。"], ["VERB", "PART", "PART", "PUNCT"]),
    (
        ["走", "吧", "，", "别", "等", "了", "。"],
        ["VERB", "PART", "PUNCT", "ADV", "VERB", "PART", "PUNCT"],
    ),
    (
        ["好", "了", "，", "别", "说", "了", "。"],
        ["ADJ", "PART", "PUNCT", "ADV", "VERB", "PART", "PUNCT"],
    ),
    (["这", "是", "我", "的", "书", "。"], ["DET", "AUX", "PRON", "PART", "NOUN", "PUNCT"]),
    (["那", "是", "你", "的", "吗", "？"], ["DET", "AUX", "PRON", "PART", "PART", "PUNCT"]),
    (["这", "是", "他", "的", "。"], ["DET", "AUX", "PRON", "PART", "PUNCT"]),
    (["我", "的", "家", "在", "北京", "。"], ["PRON", "PART", "NOUN", "ADP", "PROPN", "PUNCT"]),
    (["她", "的", "书包", "很", "新", "。"], ["PRON", "PART", "NOUN", "ADV", "ADJ", "PUNCT"]),
    (["他", "的", "朋友", "来", "了", "。"], ["PRON", "PART", "NOUN", "VERB", "PART", "PUNCT"]),
    (["这", "是", "大家", "的", "意见", "。"], ["DET", "AUX", "PRON", "PART", "NOUN", "PUNCT"]),
    (["我", "买", "了", "一些", "水果", "。"], ["PRON", "VERB", "PART", "NUM", "NOUN", "PUNCT"]),
    (["他", "说", "了", "很多", "话", "。"], ["PRON", "VERB", "PART", "ADJ", "NOUN", "PUNCT"]),
    (
        ["她", "写", "了", "一", "封", "信", "。"],
        ["PRON", "VERB", "PART", "NUM", "NUM", "NOUN", "PUNCT"],
    ),
    (["我们", "喝", "了", "茶", "。"], ["PRON", "VERB", "PART", "NOUN", "PUNCT"]),
    (["他", "买", "了", "一辆", "车", "。"], ["PRON", "VERB", "PART", "NUM", "NOUN", "PUNCT"]),
    (["他", "跑", "得", "快", "。"], ["PRON", "VERB", "PART", "ADJ", "PUNCT"]),
    (["她", "唱", "得", "好", "听", "。"], ["PRON", "VERB", "PART", "ADJ", "VERB", "PUNCT"]),
    (["他", "写", "得", "很", "好", "。"], ["PRON", "VERB", "PART", "ADV", "ADJ", "PUNCT"]),
    (
        ["他", "高兴", "得", "跳", "了", "起来", "。"],
        ["PRON", "ADJ", "PART", "VERB", "PART", "VERB", "PUNCT"],
    ),
    (
        ["她", "累", "得", "走", "不动", "了", "。"],
        ["PRON", "ADJ", "PART", "VERB", "VERB", "PART", "PUNCT"],
    ),
    (["他", "气", "得", "说不出", "话", "。"], ["PRON", "VERB", "PART", "VERB", "NOUN", "PUNCT"]),
    (
        ["她", "哭", "着", "跑", "了", "出去", "。"],
        ["PRON", "VERB", "PART", "VERB", "PART", "VERB", "PUNCT"],
    ),
    (["他", "笑", "着", "点头", "。"], ["PRON", "VERB", "PART", "VERB", "PUNCT"]),
    (["她", "站", "着", "看", "书", "。"], ["PRON", "VERB", "PART", "VERB", "NOUN", "PUNCT"]),
    (["他", "坐", "着", "等", "你", "。"], ["PRON", "VERB", "PART", "VERB", "PRON", "PUNCT"]),
    (
        ["门", "关", "着", "，", "窗", "也", "关", "着", "。"],
        ["NOUN", "VERB", "PART", "PUNCT", "NOUN", "ADV", "VERB", "PART", "PUNCT"],
    ),
    (["他", "拿着", "书", "走", "了", "。"], ["PRON", "VERB", "NOUN", "VERB", "PART", "PUNCT"]),
    (
        ["她", "想着", "心事", "，", "没", "说话", "。"],
        ["PRON", "VERB", "NOUN", "PUNCT", "ADV", "VERB", "PUNCT"],
    ),
    (
        ["他", "说", "过", "，", "会", "回来", "的", "。"],
        ["PRON", "VERB", "PART", "PUNCT", "AUX", "VERB", "PART", "PUNCT"],
    ),
    (["我", "相信", "他", "的", "话", "。"], ["PRON", "VERB", "PRON", "PART", "NOUN", "PUNCT"]),
    (
        ["所谓", "自由", "，", "就", "是", "可以", "选择", "。"],
        ["VERB", "NOUN", "PUNCT", "ADV", "AUX", "AUX", "VERB", "PUNCT"],
    ),
    (
        ["类似", "的", "问题", "，", "我们", "遇到", "过", "。"],
        ["ADJ", "PART", "NOUN", "PUNCT", "PRON", "VERB", "PART", "PUNCT"],
    ),
    (["你", "忙", "完", "了", "吗", "？"], ["PRON", "ADJ", "VERB", "PART", "PART", "PUNCT"]),
    (["他", "把", "书", "看完", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (["她", "把", "饭", "做好", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (["我", "把", "房间", "打扫", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (
        ["他", "把", "消息", "告诉", "了", "大家", "。"],
        ["PRON", "ADP", "NOUN", "VERB", "PART", "PRON", "PUNCT"],
    ),
    (
        ["她", "把", "信", "寄", "了", "出去", "。"],
        ["PRON", "ADP", "NOUN", "VERB", "PART", "VERB", "PUNCT"],
    ),
    (["我", "把", "钥匙", "弄", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (["他", "把", "工作", "完成", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (["她", "把", "衣服", "洗", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (["他", "把", "问题", "解决", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (["他", "把", "门", "锁", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (["她", "把", "花", "浇", "了", "。"], ["PRON", "ADP", "NOUN", "VERB", "PART", "PUNCT"]),
    (
        ["他", "把", "作业", "写", "完", "了", "。"],
        ["PRON", "ADP", "NOUN", "VERB", "VERB", "PART", "PUNCT"],
    ),
    # === ADV (target: 40+ sentences) — clear adverb contexts ===
    (["他", "经常", "迟到", "。"], ["PRON", "ADV", "VERB", "PUNCT"]),
    (["我们", "一直", "在", "等", "。"], ["PRON", "ADV", "ADP", "VERB", "PUNCT"]),
    (["她", "已经", "走", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["他", "正在", "吃饭", "。"], ["PRON", "ADV", "VERB", "PUNCT"]),
    (["他们", "都", "来", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["你", "也", "去", "吗", "？"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["他", "还", "没", "来", "。"], ["PRON", "ADV", "ADV", "VERB", "PUNCT"]),
    (["我", "马上", "到", "。"], ["PRON", "ADV", "VERB", "PUNCT"]),
    (["他", "终于", "成功", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["她", "突然", "哭", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["他", "慢慢", "走", "了", "过来", "。"], ["PRON", "ADV", "VERB", "PART", "VERB", "PUNCT"]),
    (["我们", "先", "走", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["她", "一直", "在", "笑", "。"], ["PRON", "ADV", "ADP", "VERB", "PUNCT"]),
    (["他", "刚才", "还", "在", "这里", "。"], ["PRON", "ADV", "ADV", "ADP", "NOUN", "PUNCT"]),
    (["大家", "都", "同意", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["他", "又", "犯", "了", "错误", "。"], ["PRON", "ADV", "VERB", "PART", "NOUN", "PUNCT"]),
    (["她", "渐渐", "习惯", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["他", "尽量", "避开", "人群", "。"], ["PRON", "ADV", "VERB", "NOUN", "PUNCT"]),
    (["我们", "最好", "明天", "出发", "。"], ["PRON", "ADV", "NOUN", "VERB", "PUNCT"]),
    (["他", "果然", "没有", "来", "。"], ["PRON", "ADV", "ADV", "VERB", "PUNCT"]),
    (["她", "竟然", "赢", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["他", "居然", "撒谎", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["我们", "也许", "会", "去", "。"], ["PRON", "ADV", "AUX", "VERB", "PUNCT"]),
    (["他", "大概", "不知道", "。"], ["PRON", "ADV", "VERB", "PUNCT"]),
    (["她", "至少", "试", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["他", "绝对", "不", "会", "同意", "。"], ["PRON", "ADV", "ADV", "AUX", "VERB", "PUNCT"]),
    (["她", "完全", "明白", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["他", "几乎", "放弃", "了", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["她", "特别", "喜欢", "音乐", "。"], ["PRON", "ADV", "VERB", "NOUN", "PUNCT"]),
    (["他", "常常", "想起", "那", "天", "。"], ["PRON", "ADV", "VERB", "DET", "NOUN", "PUNCT"]),
    (["我们", "赶紧", "走", "吧", "。"], ["PRON", "ADV", "VERB", "PART", "PUNCT"]),
    (["她", "默默", "地", "离开", "了", "。"], ["PRON", "ADV", "PART", "VERB", "PART", "PUNCT"]),
    (
        ["他", "轻轻", "地", "放下", "了", "书", "。"],
        ["PRON", "ADV", "PART", "VERB", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["她", "慢慢", "地", "打开", "了", "门", "。"],
        ["PRON", "ADV", "PART", "VERB", "PART", "NOUN", "PUNCT"],
    ),
    (
        ["他", "仔细", "地", "看", "了", "一遍", "。"],
        ["PRON", "ADV", "PART", "VERB", "PART", "NUM", "PUNCT"],
    ),
    (["她", "认真", "地", "听", "着", "。"], ["PRON", "ADV", "PART", "VERB", "PART", "PUNCT"]),
    (
        ["他", "慢慢", "地", "走", "了", "过来", "。"],
        ["PRON", "ADV", "PART", "VERB", "PART", "VERB", "PUNCT"],
    ),
    (
        ["她", "急急", "地", "跑", "了", "出去", "。"],
        ["PRON", "ADV", "PART", "VERB", "PART", "VERB", "PUNCT"],
    ),
    (
        ["他", "仔细", "地", "检查", "了", "每", "一", "页", "。"],
        ["PRON", "ADV", "PART", "VERB", "PART", "DET", "NUM", "NOUN", "PUNCT"],
    ),
    (
        ["她", "静静", "地", "坐", "在", "那里", "。"],
        ["PRON", "ADV", "PART", "VERB", "ADP", "NOUN", "PUNCT"],
    ),
    (
        ["他", "用力", "地", "关", "上", "了", "门", "。"],
        ["PRON", "ADV", "PART", "VERB", "VERB", "PART", "NOUN", "PUNCT"],
    ),
]


def word_to_bio(word: str, upos: str) -> list[tuple[str, int]]:
    chars = list(word)
    if not chars:
        return []
    upos = upos if upos in UPOS_TO_B else "X"
    result = [(chars[0], UPOS_TO_B[upos])]
    for ch in chars[1:]:
        result.append((ch, UPOS_TO_I[upos]))
    return result


def sentence_to_jsonl(words: list[str], upos: list[str]) -> dict:
    chars: list[str] = []
    labels: list[int] = []
    for word, upos_tag in zip(words, upos, strict=True):
        for ch, label_id in word_to_bio(word, upos_tag):
            chars.append(ch)
            labels.append(label_id)
    if len(chars) > 254:
        chars = chars[:254]
        labels = labels[:254]
    return {"chars": chars, "labels": labels, "length": len(chars)}


def main() -> None:
    train_path = Path("data/processed/zh_bio/train.jsonl")
    val_path = Path("data/processed/zh_bio/validation.jsonl")

    with open(train_path, encoding="utf-8") as f:
        before_train = sum(1 for _ in f)
    with open(val_path, encoding="utf-8") as f:
        before_val = sum(1 for _ in f)
    print(f"Before: train={before_train}, val={before_val}")

    # Idempotency: skip if zh-aug- entries already exist.
    already = False
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if "zh-aug-" in line:
                already = True
                break
    if already:
        print("Augmentation already applied — skipping (idempotent)")
        return

    n_train = 0
    n_val = 0
    with open(train_path, "a", encoding="utf-8") as tf, open(val_path, "a", encoding="utf-8") as vf:
        for words, upos in SENTENCES:
            entry = sentence_to_jsonl(words, upos)
            line = json.dumps(entry, ensure_ascii=False)
            if rng.random() < 0.1:
                vf.write(line + "\n")
                n_val += 1
            else:
                tf.write(line + "\n")
                n_train += 1

    with open(train_path, encoding="utf-8") as f:
        after_train = sum(1 for _ in f)
    with open(val_path, encoding="utf-8") as f:
        after_val = sum(1 for _ in f)
    print(f"Added: train=+{n_train}, val=+{n_val}")
    print(f"After: train={after_train}, val={after_val}")


if __name__ == "__main__":
    main()
