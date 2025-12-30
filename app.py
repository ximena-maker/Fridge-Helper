import os
import json
import re
from collections import defaultdict
from pathlib import Path

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    TextSendMessage,
    MessageEvent,
    TextMessage,
    FollowEvent,
    FlexSendMessage,
    QuickReply,
    QuickReplyButton,
    MessageAction,
)

# =========================================================
#  冰箱清理小幫手（LINE Bot）
#  功能：
#   1) 句子/字串輸入："我家有 牛胸肉 雞肉 洋蔥 花椰菜" -> 自動抓食材 + 推薦
#   2) 按鈕選食材：快速加入常見食材、查看冰箱、清空、用現有冰箱推薦
#  說明：
#   - 你目前專案沒有 bert-ingredient-ner 模型資料夾
#   - 所以這版「不使用 transformers / NER 模型」
#   - 改用：從 aaaaicook_data.json 產生食材字典 + split/包含匹配抽取食材
# =========================================================

# ---------------------
# LINE channel keys
# ---------------------
def load_line_keys(filepath: str = "keys.txt"):
    """
    讀取 LINE 金鑰：
    1) 優先讀環境變數 CHANNEL_SECRET / CHANNEL_ACCESS_TOKEN
    2) 其次讀與 app.py 同層的 keys.txt（或你指定的 filepath）
    """
    channel_secret = os.getenv("CHANNEL_SECRET")
    channel_access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
    if channel_secret and channel_access_token:
        return {
            "CHANNEL_SECRET": channel_secret,
            "CHANNEL_ACCESS_TOKEN": channel_access_token,
        }

    p = Path(__file__).with_name(filepath)
    if not p.exists():
        raise RuntimeError(
            "錯誤：缺少 LINE CHANNEL_SECRET / CHANNEL_ACCESS_TOKEN（請設定環境變數或提供 keys.txt）"
        )

    keys = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                keys[k.strip()] = v.strip()

    if "CHANNEL_SECRET" not in keys or "CHANNEL_ACCESS_TOKEN" not in keys:
        raise RuntimeError("keys.txt 內容不完整：需要 CHANNEL_SECRET 與 CHANNEL_ACCESS_TOKEN")

    return keys


line_keys = load_line_keys()
channel_secret = line_keys["CHANNEL_SECRET"]
channel_access_token = line_keys["CHANNEL_ACCESS_TOKEN"]
line_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# ---------------------
# 食譜資料
# ---------------------
with open("aaaaicook_data.json", encoding="utf-8") as f:
    recipes = json.load(f)

量詞 = r"(?:顆|條|片|絲|克|g|kg|匙|茶?匙|大?匙|杯|罐|包|塊|少許|適量|些許)"


def norm(word: str) -> str:
    word = re.sub(量詞, "", word, flags=re.I)
    word = re.sub(r"\s+", "", word)
    return word.lower().replace("　", "")


for r in recipes:
    r["norm_ings"] = {norm(i.split()[0]) for i in r.get("ingredients", []) if i}

inv_index = defaultdict(set)
for idx, r in enumerate(recipes):
    for ing in r["norm_ings"]:
        inv_index[ing].add(idx)

# ---------------------
# 抽取食材（不靠模型：字典 + split/包含匹配）
# ---------------------
# Quick Reply 常見食材（也會加入抽取字典）
COMMON_INGS = [
    "雞肉",
    "牛肉",
    "牛胸肉",
    "豬肉",
    "雞蛋",
    "洋蔥",
    "大蒜",
    "蔥",
    "花椰菜",
    "馬鈴薯",
    "番茄",
    "高麗菜",
    "豆腐",
]

# 同義詞/別名（可自行擴充）
ALIASES = {
    "牛胸肉": ["牛肉"],
    "青花菜": ["花椰菜"],
    "西蘭花": ["花椰菜"],
    "蔥花": ["蔥"],
}


def build_ingredient_vocab(recipes_list):
    vocab = set()

    # 從食譜 ingredients 抽詞
    for rec in recipes_list:
        for raw in rec.get("ingredients", []):
            base = raw.split()[0].strip()
            if base:
                vocab.add(norm(base))

    # 加上常用按鈕食材
    for x in COMMON_INGS:
        vocab.add(norm(x))

    # 加上同義詞
    for k, arr in ALIASES.items():
        vocab.add(norm(k))
        for a in arr:
            vocab.add(norm(a))

    # 長詞優先，避免「牛肉」先吃掉「牛胸肉」
    vocab = [v for v in vocab if v]
    vocab.sort(key=len, reverse=True)
    return vocab


ING_VOCAB = build_ingredient_vocab(recipes)


def fallback_split(text: str):
    """
    更強的 split：
    - 支援「我家有:雞肉、洋蔥」「我家有 雞肉」「冰箱有 雞肉/洋蔥」等
    - 清除前綴語氣詞 + 各種標點符號（全形/半形）
    """
    t = (text or "").strip()

    # 清掉常見前綴（含冒號/空白）
    t = re.sub(r"^(我家有|冰箱裡有|冰箱有|我剩下|剩下|有)\s*[:：]?\s*", "", t)

    # 把常見標點都當成分隔符
    # （含全形空白　、冒號：、句號。、驚嘆號！、問號？、括號等）
    t = re.sub(r"[，,、;；/\\|｜\n\r\t:：。\.！!？?\(\)（）\[\]【】{}「」\"“”'’]", " ", t)

    # 多個空白合併
    t = re.sub(r"\s+", " ", t).strip()

    if not t:
        return set()

    parts = t.split(" ")
    return {norm(p) for p in parts if p and norm(p)}


def extract_ingredients(text: str):
    """
    回傳 (entities_list, ingredient_set)
    1) 先用 split 抽詞（最符合你「我家有 ...」的輸入）
    2) 再用 ING_VOCAB 在整句做包含匹配（長詞優先）
    """
    found = set()

    # 1) split
    found |= {x for x in fallback_split(text) if x}

    # 2) 包含匹配（去空白）
    t = re.sub(r"\s+", "", text or "")
    t_norm = norm(t)

    for ing in ING_VOCAB:
        if ing and ing in t_norm:
            found.add(ing)

    # 3) 同義詞規整
    for main, arr in ALIASES.items():
        main_n = norm(main)
        for a in arr:
            if norm(a) in found:
                found.add(main_n)

    found = {x for x in found if x and x not in {"。", "，", ",", "、"}}
    return [], found


# ---------------------
# 推薦演算法
# ---------------------
def score_fn(overlap, missing, total):
    return len(overlap) * 10 - len(missing) + (len(overlap) / total) * 200


def recommend(user_ings_raw, topk=5, allow_missing=True, max_missing=8, min_overlap=1):
    user_ings = {norm(w) for w in user_ings_raw if norm(w)}
    if not user_ings:
        return []

    cand_idx = set().union(*(inv_index.get(i, set()) for i in user_ings))
    scored = []
    for idx in cand_idx:
        rec = recipes[idx]
        overlap = user_ings & rec["norm_ings"]
        if len(overlap) < min_overlap:
            continue
        missing = rec["norm_ings"] - user_ings
        if (not allow_missing and missing) or len(missing) > max_missing:
            continue
        score = score_fn(overlap, missing, len(rec["norm_ings"]) or 1)
        scored.append((score, overlap, missing, rec))

    scored.sort(key=lambda x: (-x[0], len(x[2]), x[3].get("name", "")))
    return scored[:topk]


def recipe_to_bubble(rank, overlap, missing, recipe):
    have = "、".join(sorted(overlap)) or "—"
    lack = "、".join(sorted(missing)) or "—"
    return {
        "type": "bubble",
        "size": "mega",
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {
                    "type": "text",
                    "text": f"{rank}. {recipe.get('name','(未命名)')}",
                    "wrap": True,
                    "weight": "bold",
                    "size": "lg",
                    "margin": "none",
                },
                {
                    "type": "text",
                    "text": f"⭕ 🈶：{have}",
                    "wrap": True,
                    "size": "sm",
                    "margin": "md",
                },
                {
                    "type": "text",
                    "text": f"❌ 🈚：{lack}",
                    "wrap": True,
                    "size": "sm",
                },
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#1DB446",
                    "action": {
                        "type": "message",
                        "label": f"看做法({rank})",
                        "text": f"做法 {rank}",
                    },
                }
            ],
        },
    }


# ---------------------
# 使用者冰箱（記憶：目前存在記憶體，重啟會清空）
# ---------------------
user_fridge = defaultdict(set)  # user_id -> set(norm ingredient)
recent_rec = {}  # user_id -> list[recipe]


def fridge_list_text(user_id: str) -> str:
    ings = sorted(user_fridge[user_id])
    return "你的冰箱目前：" + ("、".join(ings) if ings else "（空的）")


def add_to_fridge(user_id: str, ings):
    for w in ings:
        nw = norm(w)
        if nw:
            user_fridge[user_id].add(nw)


def clear_fridge(user_id: str):
    user_fridge[user_id].clear()


# ---------------------
# Quick Reply（按鈕選食材）
# ---------------------
def make_quickreply_menu():
    """最多 13 個 quick reply actions；留 1~2 個做系統按鈕。"""
    items = []

    # 常見食材：點一下就加入（取前 10 個避免超過限制）
    for ing in COMMON_INGS[:10]:
        items.append(QuickReplyButton(action=MessageAction(label=f"+{ing}", text=f"加入 {ing}")))

    # 系統按鈕
    items.append(QuickReplyButton(action=MessageAction(label="🍳 推薦", text="推薦")))
    items.append(QuickReplyButton(action=MessageAction(label="📦 查看冰箱", text="查看冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="🗑 清空", text="清空冰箱")))

    return QuickReply(items=items)


# ---------------------
# 推薦流程
# ---------------------
def recommend_and_build_messages(user_id: str, ing_set, topk=5):
    """用 ing_set 推薦，回傳 (TextSendMessage, FlexSendMessage or None, bubbles_count)"""
    if not ing_set:
        return (
            TextSendMessage(
                text="我沒有偵測到可用食材喔～你可以用『選食材』按鈕加入，或再描述一次。",
                quick_reply=make_quickreply_menu(),
            ),
            None,
            0,
        )

    recs = recommend(ing_set, topk=topk, allow_missing=True, max_missing=10)
    if not recs:
        return (
            TextSendMessage(
                text=f"資料庫找不到適合「{'、'.join(sorted(ing_set))}」的食譜 😢\n你可以再加一些食材或換組合試試。",
                quick_reply=make_quickreply_menu(),
            ),
            None,
            0,
        )

    bubbles = [
        recipe_to_bubble(rank=i, overlap=ov, missing=miss, recipe=r)
        for i, (_, ov, miss, r) in enumerate(recs, 1)
    ]

    # 存給「做法 N」用
    recent_rec[user_id] = [r for _, _, _, r in recs]

    text_msg = TextSendMessage(
        text=(
            f"偵測/使用的食材：{'、'.join(sorted(ing_set))}\n"
            f"{fridge_list_text(user_id)}\n\n"
            "輸入『做法 + 編號』可看完整步驟；或用按鈕繼續加食材再推薦。"
        ),
        quick_reply=make_quickreply_menu(),
    )

    flex_msg = FlexSendMessage(
        alt_text="推薦料理",
        contents={"type": "carousel", "contents": bubbles},
    )

    return text_msg, flex_msg, len(bubbles)


# ---------------------
# Flask
# ---------------------
app = Flask(__name__)


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(FollowEvent)
def handle_follow(event: FollowEvent):
    welcome = (
        "嗨～我是冰箱清理小幫手！\n\n"
        "✅ 你可以直接輸入一句話：\n"
        "例如：『我家有 牛胸肉 雞肉 洋蔥 花椰菜』\n\n"
        "✅ 或輸入『選食材』用按鈕加入食材\n"
        "✅ 輸入『推薦』用你冰箱裡的食材推薦料理\n"
        "✅ 輸入『查看冰箱』『清空冰箱』管理食材\n"
        "（看做法：輸入『做法 1』）"
    )
    line_api.reply_message(
        event.reply_token,
        TextSendMessage(text=welcome, quick_reply=make_quickreply_menu()),
    )


@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    text = (event.message.text or "").strip()

    # ---------- 看做法 ----------
    if text.startswith("做法"):
        m = re.search(r"\d+", text)
        if m and user_id in recent_rec:
            idx = int(m.group()) - 1
            if 0 <= idx < len(recent_rec[user_id]):
                recipe = recent_rec[user_id][idx]
                line_api.reply_message(
                    event.reply_token,
                    TextSendMessage(
                        f"《{recipe.get('name','(未命名)')}》\n\n"
                        + recipe.get("instructions", "（沒有步驟內容）")
                    ),
                )
                return
        line_api.reply_message(
            event.reply_token,
            TextSendMessage("找不到對應的編號耶～先輸入食材讓我推薦一次，再輸入『做法 1』喔。"),
        )
        return

    # ---------- 管理冰箱 ----------
    if text in {"查看冰箱", "冰箱", "我的冰箱"}:
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(text=fridge_list_text(user_id), quick_reply=make_quickreply_menu()),
        )
        return

    if text in {"清空冰箱", "清空", "重置冰箱"}:
        clear_fridge(user_id)
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(text="已清空～\n" + fridge_list_text(user_id), quick_reply=make_quickreply_menu()),
        )
        return

    # ---------- 按鈕選食材 ----------
    if text in {"選食材", "新增食材", "加食材", "按鈕", "menu", "MENU"}:
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="你可以點按鈕快速加入食材（也可以直接打字：『加入 牛胸肉』）。",
                quick_reply=make_quickreply_menu(),
            ),
        )
        return

    # ---------- 手動加入（文字） ----------
    m_add = re.match(r"^(?:加入|加|新增)[:：\s]+(.+)$", text)
    if m_add:
        raw = m_add.group(1)

        parts = re.split(r"[\s、,，;；/]+", raw)
        parts = [p.strip() for p in parts if p.strip()]

        _, ing_set = extract_ingredients(raw)
        if ing_set:
            add_to_fridge(user_id, ing_set)
            added = sorted({norm(i) for i in ing_set if norm(i)})
        else:
            add_to_fridge(user_id, parts)
            added = sorted({norm(i) for i in parts if norm(i)})

        line_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"已加入：{'、'.join(added) if added else '（未偵測到）'}\n{fridge_list_text(user_id)}",
                quick_reply=make_quickreply_menu(),
            ),
        )
        return

    # ---------- 用冰箱推薦 ----------
    if text in {"推薦", "推薦料理", "煮什麼", "做什麼", "想煮"}:
        ing_set = set(user_fridge[user_id])
        text_msg, flex_msg, _ = recommend_and_build_messages(user_id, ing_set, topk=5)
        msgs = [text_msg] + ([flex_msg] if flex_msg else [])
        line_api.reply_message(event.reply_token, msgs)
        return

    # ---------- 一般句子：自動抓食材 + 推薦 ----------
    # ✅ 修正：永遠合併 extract_ingredients + fallback_split，避免漏抓「我家有 雞肉」
    _, ing_set_model = extract_ingredients(text)
    ing_set_split = fallback_split(text)

    ing_set = set(ing_set_model) | set(ing_set_split)
    ing_set = {x for x in ing_set if x}

    if ing_set:
        add_to_fridge(user_id, ing_set)
        use_set = set(user_fridge[user_id])
        text_msg, flex_msg, _ = recommend_and_build_messages(user_id, use_set, topk=5)
        msgs = [text_msg] + ([flex_msg] if flex_msg else [])
        line_api.reply_message(event.reply_token, msgs)
        return

    # 沒抓到：提示用法
    line_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=(
                "我沒有在這句話裡偵測到食材耶～\n"
                "你可以：\n"
                "1) 直接輸入：『我家有 牛胸肉 雞肉 洋蔥 花椰菜』\n"
                "2) 輸入『選食材』用按鈕加入\n"
                "3) 或輸入：『加入 牛胸肉』"
            ),
            quick_reply=make_quickreply_menu(),
        ),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
