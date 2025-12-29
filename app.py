import os
import json
import re
from collections import defaultdict
from typing import List, Dict, Any

import httpx
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
# 1) 設定：LINE Keys / Gemini Keys
# =========================================================

def load_line_keys(filename="keys.txt"):
    keys = {}

    # 永遠讀 app.py 同層的 keys.txt（避免 cwd 問題）
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, filename)

    if os.path.exists(filepath):
        # utf-8-sig 會自動吃掉 BOM
        with open(filepath, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip().lstrip("\ufeff")  # 保險：再手動去一次 BOM
                    keys[k] = v.strip()

    return keys



file_keys = load_line_keys()

CHANNEL_SECRET = os.environ.get("CHANNEL_SECRET") or file_keys.get("CHANNEL_SECRET", "")
CHANNEL_ACCESS_TOKEN = os.environ.get("CHANNEL_ACCESS_TOKEN") or file_keys.get("CHANNEL_ACCESS_TOKEN", "")

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    raise RuntimeError("缺少 LINE CHANNEL_SECRET / CHANNEL_ACCESS_TOKEN（請設定環境變數或 keys.txt）")

line_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)
print("TOKEN length:", len(CHANNEL_ACCESS_TOKEN), "SECRET length:", len(CHANNEL_SECRET))

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or file_keys.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # 你也可以改成其他可用 model


# =========================================================
# 2) 狀態：使用者選的食材 / 料理結果 / quick reply 頁數
# =========================================================

user_selected = defaultdict(set)          # user_id -> set(食材)
user_last_recipes: Dict[str, List[Dict[str, Any]]] = {}   # user_id -> Gemini 回傳的 recipes
user_page = defaultdict(int)             # user_id -> quick reply page index

# QuickReply 食材清單（你可以自由擴充）
COMMON_INGS = [
    "雞蛋", "牛奶", "吐司", "起司", "奶油", "優格",
    "番茄", "洋蔥", "蒜頭", "青蔥", "薑", "辣椒",
    "高麗菜", "小黃瓜", "紅蘿蔔", "馬鈴薯", "玉米", "花椰菜",
    "豆腐", "豆干", "金針菇", "香菇", "鴻喜菇", "杏鮑菇",
    "雞胸", "雞腿", "豬肉", "牛肉", "絞肉", "培根",
    "鮭魚", "鯖魚", "蝦仁", "花枝", "蛤蜊",
    "白飯", "麵條", "冬粉", "烏龍麵",
    "醬油", "鹽", "胡椒", "味噌", "番茄醬", "咖哩塊",
]

PAGE_SIZE = 8


# =========================================================
# 3) Gemini 呼叫：REST generateContent（JSON 回傳）
# =========================================================

def _extract_json(text: str) -> str:
    """
    保險用：如果模型沒乾淨輸出 JSON，嘗試抓第一段 [ ... ] 或 { ... }
    """
    text = text.strip()
    # 優先抓 list JSON
    m = re.search(r"(\[\s*{.*}\s*\])", text, re.DOTALL)
    if m:
        return m.group(1)
    # 再抓 object JSON
    m = re.search(r"(\{\s*\".*\}\s*)", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def gemini_recipe_search(selected_ings: List[str], topk: int = 5) -> List[Dict[str, Any]]:
    """
    用 Gemini 依食材生成 topk 道食譜（回傳 list[dict]）
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("缺少 GEMINI_API_KEY（請設定環境變數）")

    prompt = f"""
你是料理助理。使用者手上有這些食材：{", ".join(selected_ings)}。
請推薦 {topk} 道「盡量用到上述食材」的家常料理。

請只輸出 JSON（不要多任何文字），格式如下：
[
  {{
    "name": "料理名",
    "time_min": 20,
    "ingredients": ["雞蛋 2顆", "番茄 1顆", "..."],
    "steps": ["步驟1...", "步驟2..."],
    "tips": "可選，1句小提醒"
  }}
]
要求：
- steps 要具體可操作
- ingredients 請用「食材 + 大概份量」表示
- 不要輸出網址
""".strip()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.6
        }
    }

    r = httpx.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()

    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    text = _extract_json(text)

    recipes = json.loads(text)
    if not isinstance(recipes, list):
        raise RuntimeError("Gemini 回傳格式不是 list JSON")
    return recipes[:topk]


# =========================================================
# 4) LINE UI：QuickReply（按鈕選食材）、Flex（顯示推薦）
# =========================================================

def build_ing_quick_reply(user_id: str) -> QuickReply:
    """
    12 個食材 + 控制按鈕（更多/完成/清空/已選）
    """
    page = user_page[user_id]
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    chunk = COMMON_INGS[start:end]

    # 如果超出範圍，回到第一頁
    if not chunk:
        user_page[user_id] = 0
        start = 0
        end = PAGE_SIZE
        chunk = COMMON_INGS[start:end]

    items = []
    for ing in chunk:
        items.append(
            QuickReplyButton(action=MessageAction(label=ing, text=f"+{ing}"))
        )

    items.extend([
        QuickReplyButton(action=MessageAction(label="➕更多", text="更多")),
        QuickReplyButton(action=MessageAction(label="✅完成查食譜", text="完成")),
        QuickReplyButton(action=MessageAction(label="🗑️清空", text="清空")),
        QuickReplyButton(action=MessageAction(label="📌已選", text="已選")),
        QuickReplyButton(action=MessageAction(label="❓幫助", text="幫助")),
    ])

    return QuickReply(items=items)


def recipe_to_bubble(recipe: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """
    產生 Flex bubble
    """
    name = str(recipe.get("name", f"料理{rank}"))
    time_min = recipe.get("time_min", "?")
    ingredients = recipe.get("ingredients", [])
    tips = recipe.get("tips", "")

    if isinstance(ingredients, list):
        ing_preview = "\n".join([f"• {x}" for x in ingredients[:6]])
        if len(ingredients) > 6:
            ing_preview += "\n• ..."
    else:
        ing_preview = str(ingredients)

    body_contents = [
        {
            "type": "text",
            "text": f"{rank}. {name}",
            "weight": "bold",
            "size": "lg",
            "wrap": True
        },
        {
            "type": "text",
            "text": f"⏱ 約 {time_min} 分鐘",
            "size": "sm",
            "margin": "md",
            "wrap": True
        },
        {
            "type": "text",
            "text": "🧺 食材（部分）",
            "size": "sm",
            "margin": "md",
            "weight": "bold"
        },
        {
            "type": "text",
            "text": ing_preview or "（未提供）",
            "size": "sm",
            "wrap": True,
            "margin": "sm"
        }
    ]

    if tips:
        body_contents.append({
            "type": "text",
            "text": f"💡 {tips}",
            "size": "sm",
            "wrap": True,
            "margin": "md"
        })

    bubble = {
        "type": "bubble",
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": body_contents
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",
                    "action": {
                        "type": "message",
                        "label": "看做法",
                        "text": f"做法 {rank}"
                    }
                }
            ]
        }
    }
    return bubble


def build_recipe_carousel(recipes: List[Dict[str, Any]]) -> FlexSendMessage:
    bubbles = [recipe_to_bubble(r, i + 1) for i, r in enumerate(recipes[:10])]
    return FlexSendMessage(
        alt_text="推薦料理",
        contents={
            "type": "carousel",
            "contents": bubbles
        }
    )


def help_text() -> str:
    return (
        "🍳 冰箱食譜小幫手\n\n"
        "指令：\n"
        "1) 選食材：開始用按鈕選食材\n"
        "2) +食材：手動加入（例：+雞蛋）\n"
        "3) 已選：查看目前已選食材\n"
        "4) 清空：清掉已選食材\n"
        "5) 完成：用 Gemini 依食材推薦食譜\n"
        "6) 做法 N：查看第 N 道料理步驟\n"
    )


# =========================================================
# 5) Flask webhook
# =========================================================

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


# =========================================================
# 6) LINE Events
# =========================================================

@handler.add(FollowEvent)
def on_follow(event):
    user_id = event.source.user_id
    user_page[user_id] = 0
    line_api.reply_message(
        event.reply_token,
        TextSendMessage("嗨！輸入「選食材」開始用按鈕選食材～\n也可輸入「幫助」看指令。")
    )


@handler.add(MessageEvent, message=TextMessage)
def on_message(event):
    user_id = event.source.user_id
    text = (event.message.text or "").strip()

    # --- 幫助 ---
    if text in ["幫助", "help", "?"]:
        line_api.reply_message(event.reply_token, TextSendMessage(help_text()))
        return

    # --- 開始選食材 ---
    if text in ["選食材", "開始", "開始選食材"]:
        user_page[user_id] = 0
        line_api.reply_message(
            event.reply_token,
            TextSendMessage("請點選你冰箱有的食材（可一直點）", quick_reply=build_ing_quick_reply(user_id))
        )
        return

    # --- 更多（翻頁） ---
    if text == "更多":
        user_page[user_id] += 1
        line_api.reply_message(
            event.reply_token,
            TextSendMessage("更多食材在這～", quick_reply=build_ing_quick_reply(user_id))
        )
        return

    # --- 已選 ---
    if text == "已選":
        now = "、".join(sorted(user_selected[user_id])) or "（尚未選）"
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(f"📌目前已選：{now}", quick_reply=build_ing_quick_reply(user_id))
        )
        return

    # --- 清空 ---
    if text == "清空":
        user_selected[user_id].clear()
        user_last_recipes.pop(user_id, None)
        line_api.reply_message(
            event.reply_token,
            TextSendMessage("🗑️已清空！重新選食材吧～", quick_reply=build_ing_quick_reply(user_id))
        )
        return

    # --- 點按鈕加入食材：+xxx ---
    if text.startswith("+"):
        ing = text[1:].strip()
        if ing:
            user_selected[user_id].add(ing)
        now = "、".join(sorted(user_selected[user_id])) or "（尚未選）"
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(f"✅已加入：{ing}\n目前已選：{now}", quick_reply=build_ing_quick_reply(user_id))
        )
        return

    # --- 做法 N ---
    if text.startswith("做法"):
        m = re.search(r"\d+", text)
        if not m:
            line_api.reply_message(event.reply_token, TextSendMessage("請輸入：做法 1 / 做法 2 ..."))
            return

        idx = int(m.group()) - 1
        recipes = user_last_recipes.get(user_id, [])
        if not recipes or idx < 0 or idx >= len(recipes):
            line_api.reply_message(event.reply_token, TextSendMessage("找不到這道料理～請先「完成」查食譜。"))
            return

        recipe = recipes[idx]
        steps = recipe.get("steps", [])
        if isinstance(steps, list):
            steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
        else:
            steps_text = str(steps)

        line_api.reply_message(
            event.reply_token,
            TextSendMessage(f"《{recipe.get('name','料理')}》\n\n{steps_text or '（未提供步驟）'}")
        )
        return

    # --- 完成：呼叫 Gemini ---
    if text == "完成":
        ings = sorted(user_selected[user_id])
        if not ings:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage("你還沒選食材喔～先輸入「選食材」", quick_reply=build_ing_quick_reply(user_id))
            )
            return

        try:
            recipes = gemini_recipe_search(ings, topk=5)
            user_last_recipes[user_id] = recipes

            # 先回一則文字 + 再回 Flex（同一個 reply_token 可一次回多則）
            selected_text = "、".join(ings)
            msg1 = TextSendMessage(f"🍽️你選的食材：{selected_text}\n我幫你找到了 {len(recipes)} 道料理：")
            msg2 = build_recipe_carousel(recipes)

            line_api.reply_message(event.reply_token, [msg1, msg2])

        except Exception as e:
            line_api.reply_message(event.reply_token, TextSendMessage(f"❌查詢失敗：{e}"))
        return

    # --- 其他輸入：提示 ---
    line_api.reply_message(
        event.reply_token,
        TextSendMessage("輸入「選食材」開始，或輸入「幫助」看指令。")
    )


# =========================================================
# 7) main
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
