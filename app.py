import os
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple

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

from google import genai

# =========================================================
#  冰箱清理小幫手（LINE Bot）- Google Gemini 版
#
#  功能：
#   1) 你輸入：「我家有 雞肉 洋蔥」→ Gemini 抽取食材 + 生成 3 道食譜
#   2) 按鈕快速加入常見食材：加入 雞肉 / 加入 洋蔥 ...
#   3) 文字加入：加入 雞肉 洋蔥
#   4) 查看冰箱 / 清空冰箱
#   5) 推薦：用冰箱現有食材生成 3 道食譜
#   6) 做法 1/2/3：看完整步驟與用量
#
#  你要設定的環境變數：
#   - CHANNEL_SECRET / CHANNEL_ACCESS_TOKEN（或 keys.txt）
#   - GEMINI_API_KEY（Google AI Studio 取得）
#
#  可選：
#   - GEMINI_MODEL：預設 gemini-2.5-flash
# =========================================================


# ---------------------
# LINE channel keys
# ---------------------
def load_line_keys(filepath: str = "keys.txt") -> Dict[str, str]:
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
# Gemini client
# ---------------------
# 會自動從 GEMINI_API_KEY / GOOGLE_API_KEY 環境變數讀取
if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    raise RuntimeError("缺少 GEMINI_API_KEY（請在本機或 Render 設定環境變數）")

client = genai.Client()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Structured Outputs JSON Schema（強制回傳 JSON）
RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "extracted_ingredients": {
            "type": "array",
            "items": {"type": "string"},
            "description": "從 user_input 抽取到的食材（去掉數量/單位），以繁體中文為主。"
        },
        "recipes": {
            "type": "array",
            "description": "生成的食譜清單（最多 3 道）。",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "servings": {"type": "string"},
                    "time_minutes": {"type": "integer"},
                    "ingredients": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item": {"type": "string"},
                                "amount": {"type": "string"},
                            },
                            "required": ["item", "amount"],
                        },
                    },
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "missing": {"type": "array", "items": {"type": "string"}},
                    "tips": {"type": "string"},
                },
                "required": ["title", "servings", "time_minutes", "ingredients", "steps", "missing"],
            },
        },
        "followup_question": {"type": "string"},
    },
    "required": ["extracted_ingredients", "recipes"],
}


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Structured Outputs 理論上會回 JSON，但仍做保險：
    1) 直接 json.loads
    2) 抽第一段 { ... } 再 loads
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def gemini_extract_and_generate(user_input: str, fridge_ingredients: List[str], topk: int = 3) -> Dict[str, Any]:
    """
    給 Gemini：
      - 抽取食材
      - 以「抽取食材 + 冰箱食材」生成最多 topk 道料理（含缺少食材）
    回傳符合 RESPONSE_JSON_SCHEMA 的 dict
    """
    fridge_ingredients = fridge_ingredients or []
    # 盡量讓模型不亂編：要求 missing 列出需要但家裡沒有的
    prompt = f"""
你是「冰箱清理小幫手」料理助理。請用繁體中文回覆，並且只輸出 JSON（不要加任何多餘文字）。
目標：
1) 從 user_input 抽取食材，放到 extracted_ingredients（去掉數量、單位；例如「雞肉」「洋蔥」「雞蛋」）。
2) 使用可用食材 = extracted_ingredients + fridge_ingredients 來生成最多 {topk} 道可做的家常料理（recipes）。
3) 每道 recipes 需包含：
   - title（菜名）
   - servings（份量字串，例如「2人份」）
   - time_minutes（整數分鐘）
   - ingredients：列出「主要需要」的食材與用量（item/amount）
   - steps：步驟陣列（3~10步）
   - missing：你認為要做這道菜還需要、但可用食材沒有的項目（例如醬油、鹽、胡椒，可列出）
   - tips：一段小技巧（可空字串）
規則：
- 不要把不存在於可用食材的東西假裝「家裡有」；如果需要就放到 missing。
- 如果可用食材太少，recipes 可以是空陣列，並在 followup_question 問使用者還有哪些食材。
- extracted_ingredients 若抽不到就回傳空陣列。

fridge_ingredients = {fridge_ingredients}
user_input = {user_input}
"""

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": RESPONSE_JSON_SCHEMA,
            # 可選：讓輸出更穩（你也可自行調）
            "temperature": 0.4,
        },
    )
    data = _safe_json_loads(resp.text)

    # 防呆整理
    data.setdefault("extracted_ingredients", [])
    data.setdefault("recipes", [])
    if not isinstance(data["extracted_ingredients"], list):
        data["extracted_ingredients"] = []
    if not isinstance(data["recipes"], list):
        data["recipes"] = []

    # 去重、清理空字
    data["extracted_ingredients"] = sorted({str(x).strip() for x in data["extracted_ingredients"] if str(x).strip()})

    # 限制 recipes 數量
    data["recipes"] = data["recipes"][: max(0, int(topk))]
    return data


# ---------------------
# 使用者冰箱（記憶：目前存在記憶體，重啟會清空）
# ---------------------
user_fridge = defaultdict(set)  # user_id -> set(ingredient str)
recent_rec = {}  # user_id -> list[recipe dict]


def fridge_list_text(user_id: str) -> str:
    ings = sorted(user_fridge[user_id])
    return "你的冰箱目前：" + ("、".join(ings) if ings else "（空的）")


def add_to_fridge(user_id: str, ings: List[str] | set):
    for w in ings:
        w = str(w).strip()
        if w:
            user_fridge[user_id].add(w)


def clear_fridge(user_id: str):
    user_fridge[user_id].clear()


# ---------------------
# Quick Reply（按鈕選食材）
# ---------------------
COMMON_INGS = [
    "雞肉", "牛肉", "豬肉", "雞蛋", "洋蔥",
    "大蒜", "蔥", "番茄", "馬鈴薯", "花椰菜",
    "高麗菜", "豆腐",
]


def make_quickreply_menu():
    """
    LINE Quick Reply actions 有數量上限，保守做法：
      - 10 個常見食材
      - + 推薦 / 查看冰箱 / 清空
    """
    items = []
    for ing in COMMON_INGS[:10]:
        items.append(QuickReplyButton(action=MessageAction(label=f"+{ing}", text=f"加入 {ing}")))

    items.append(QuickReplyButton(action=MessageAction(label="🍳 推薦", text="推薦")))
    items.append(QuickReplyButton(action=MessageAction(label="📦 查看冰箱", text="查看冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="🗑 清空", text="清空冰箱")))

    return QuickReply(items=items)


# ---------------------
# 文字/訊息格式化
# ---------------------
def recipes_to_summary_text(recipes: List[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(recipes, 1):
        title = r.get("title", "(無標題)")
        t = r.get("time_minutes", "?")
        miss = r.get("missing", []) or []
        miss_txt = ("（缺：" + "、".join(miss[:6]) + ("…" if len(miss) > 6 else "") + "）") if miss else ""
        lines.append(f"{i}. {title}｜約 {t} 分鐘{miss_txt}")
    return "\n".join(lines)


def recipes_to_flex(recipes: List[Dict[str, Any]]) -> FlexSendMessage:
    bubbles = []
    for i, r in enumerate(recipes, 1):
        title = r.get("title", "(無標題)")
        t = r.get("time_minutes", "?")
        miss = r.get("missing", []) or []
        miss_txt = "、".join(miss[:10]) if miss else "（無）"

        bubble = {
            "type": "bubble",
            "size": "mega",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "md",
                "contents": [
                    {"type": "text", "text": f"{i}. {title}", "wrap": True, "weight": "bold", "size": "lg"},
                    {"type": "text", "text": f"⏱ 約 {t} 分鐘", "wrap": True, "size": "sm"},
                    {"type": "text", "text": f"❌ 缺少：{miss_txt}", "wrap": True, "size": "sm"},
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
                        "action": {"type": "message", "label": f"看做法({i})", "text": f"做法 {i}"},
                    }
                ],
            },
        }
        bubbles.append(bubble)

    return FlexSendMessage(
        alt_text="推薦料理",
        contents={"type": "carousel", "contents": bubbles},
    )


def build_recipe_detail(recipe: Dict[str, Any]) -> str:
    title = recipe.get("title", "(無標題)")
    servings = recipe.get("servings", "?")
    time_minutes = recipe.get("time_minutes", "?")
    missing = recipe.get("missing", []) or []
    tips = recipe.get("tips", "")

    ing_lines = []
    for x in recipe.get("ingredients", []) or []:
        item = str(x.get("item", "")).strip()
        amount = str(x.get("amount", "")).strip()
        if item or amount:
            ing_lines.append(f"- {item}：{amount}")

    steps = recipe.get("steps", []) or []
    step_lines = [f"{i+1}. {s}" for i, s in enumerate(steps)] if steps else ["（沒有步驟內容）"]

    msg = (
        f"《{title}》\n"
        f"份量：{servings}\n"
        f"時間：約 {time_minutes} 分鐘\n\n"
        f"食材：\n" + ("\n".join(ing_lines) if ing_lines else "（未提供）") + "\n\n"
        f"步驟：\n" + "\n".join(step_lines)
    )

    if missing:
        msg += "\n\n缺少：\n- " + "\n- ".join(missing)

    if tips.strip():
        msg += "\n\n小技巧：\n" + tips.strip()

    return msg


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
        "嗨～我是冰箱清理小幫手（Gemini 版）！\n\n"
        "✅ 直接輸入一句話我會自動抓食材 + 推薦：\n"
        "例如：『我家有 雞肉 洋蔥 雞蛋』\n\n"
        "✅ 或輸入『選食材』用按鈕加入食材\n"
        "✅ 輸入『推薦』用你冰箱裡的食材生成食譜\n"
        "✅ 輸入『查看冰箱』『清空冰箱』管理食材\n"
        "✅ 看做法：輸入『做法 1』"
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
            recs = recent_rec.get(user_id, [])
            if 0 <= idx < len(recs):
                recipe = recs[idx]
                line_api.reply_message(event.reply_token, TextSendMessage(text=build_recipe_detail(recipe)))
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
                text="你可以點按鈕快速加入食材（也可以直接打字：『加入 雞肉 洋蔥』）。",
                quick_reply=make_quickreply_menu(),
            ),
        )
        return

    # ---------- 手動加入（文字） ----------
    m_add = re.match(r"^(?:加入|加|新增)[:：\s]+(.+)$", text)
    if m_add:
        raw = m_add.group(1).strip()
        parts = [p.strip() for p in re.split(r"[\s、,，;；/]+", raw) if p.strip()]
        if not parts:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text="我沒看到你要加入的食材～例如：加入 雞肉 洋蔥", quick_reply=make_quickreply_menu()),
            )
            return
        add_to_fridge(user_id, parts)
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=f"已加入：{'、'.join(sorted(set(parts)))}\n{fridge_list_text(user_id)}",
                quick_reply=make_quickreply_menu(),
            ),
        )
        return

    # ---------- 用冰箱推薦 ----------
    if text in {"推薦", "推薦料理", "煮什麼", "做什麼", "想煮"}:
        fridge = sorted(user_fridge[user_id])
        if not fridge:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="你的冰箱目前是空的～先輸入：『我家有 雞肉 洋蔥』或用『選食材』加入吧！",
                    quick_reply=make_quickreply_menu(),
                ),
            )
            return

        try:
            data = gemini_extract_and_generate(
                user_input="請用我冰箱現有食材幫我生成可做的家常料理",
                fridge_ingredients=fridge,
                topk=3,
            )
        except Exception as e:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"Gemini 產生食譜時出錯了：{e}", quick_reply=make_quickreply_menu()),
            )
            return

        recipes = data.get("recipes", [])
        if not recipes:
            q = data.get("followup_question") or "我目前想不到合適的菜色，你可以再提供更多食材嗎？"
            line_api.reply_message(event.reply_token, TextSendMessage(text=q, quick_reply=make_quickreply_menu()))
            return

        recent_rec[user_id] = recipes

        summary = (
            f"{fridge_list_text(user_id)}\n\n"
            "我用你的冰箱食材生成了：\n"
            + recipes_to_summary_text(recipes)
            + "\n\n想看完整步驟：輸入 做法 1 / 做法 2 / 做法 3"
        )
        msgs = [
            TextSendMessage(text=summary, quick_reply=make_quickreply_menu()),
            recipes_to_flex(recipes),
        ]
        line_api.reply_message(event.reply_token, msgs)
        return

    # ---------- 一般句子：交給 Gemini 抽食材 + 生成食譜 ----------
    fridge_before = sorted(user_fridge[user_id])
    try:
        data = gemini_extract_and_generate(user_input=text, fridge_ingredients=fridge_before, topk=3)
    except Exception as e:
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=(
                    f"Gemini 解析時出錯了：{e}\n\n"
                    "你可以改用：\n"
                    "1) 我家有 雞肉 洋蔥\n"
                    "2) 加入 雞肉 洋蔥\n"
                    "3) 選食材"
                ),
                quick_reply=make_quickreply_menu(),
            ),
        )
        return

    extracted = data.get("extracted_ingredients", []) or []
    if extracted:
        add_to_fridge(user_id, extracted)

    fridge_now = sorted(user_fridge[user_id])
    recipes = data.get("recipes", []) or []

    # 若完全沒有抽到食材，也沒有食譜：引導
    if (not extracted) and (not recipes):
        q = data.get("followup_question") or (
            "我沒有在這句話裡抓到食材耶～\n"
            "你可以：\n"
            "1) 直接輸入：『我家有 雞肉 洋蔥』\n"
            "2) 輸入『選食材』用按鈕加入\n"
            "3) 或輸入：『加入 雞肉』"
        )
        line_api.reply_message(event.reply_token, TextSendMessage(text=q, quick_reply=make_quickreply_menu()))
        return

    # 有食譜就回覆推薦；沒有就只回更新冰箱
    if recipes:
        recent_rec[user_id] = recipes
        summary = (
            f"我抓到的食材：{'、'.join(extracted) if extracted else '（未新增）'}\n"
            f"{fridge_list_text(user_id)}\n\n"
            "我幫你生成了：\n"
            + recipes_to_summary_text(recipes)
            + "\n\n想看完整步驟：輸入 做法 1 / 做法 2 / 做法 3"
        )
        msgs = [
            TextSendMessage(text=summary, quick_reply=make_quickreply_menu()),
            recipes_to_flex(recipes),
        ]
        line_api.reply_message(event.reply_token, msgs)
        return

    # 沒食譜，但有抽到食材：只回冰箱更新 + 引導再推薦
    line_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text=(
                f"已加入：{'、'.join(extracted)}\n"
                f"{fridge_list_text(user_id)}\n\n"
                "你可以輸入『推薦』用冰箱食材生成料理。"
            ),
            quick_reply=make_quickreply_menu(),
        ),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
