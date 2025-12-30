# app.py
import os
import re
import json
import uuid
import base64
import logging
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
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

# Google GenAI SDK (pip install google-genai)
from google import genai
from google.genai import types

# =========================================================
# 冰箱清理小幫手（LINE Bot）
#
# ✅ ? / help / 幫助：叫出選單 + 使用方法
# ✅ 直接打「食材文字」：只加入冰箱（不自動推薦）
# ✅ 只有輸入「+ 食材」「- 食材」才會加/減食材（+ / - 單獨輸入是叫出選單/移除選單）
# ✅ 推薦：用 Gemini 產 3 道食譜（每道 1 張示意圖）
# ✅ 換食譜：同一批食材換 3 道
# ✅ 做法 N：一次輸出全部步驟圖（但會依 LINE Flex 限制自動分成多則 Flex）
# ✅ 看做法後：卡片按鈕可「換食譜」
# ✅ 打非食材內容：跳出選單及使用方法
#
# 圖片：預設用 Gemini 影像模型（IMAGE_MODEL=gemini-2.5-flash-image）
# 若你要改成 Imagen：把 IMAGE_MODEL 設成 imagen-*
#
# 必要環境變數（Render / 本機）：
# - CHANNEL_SECRET
# - CHANNEL_ACCESS_TOKEN
# - GEMINI_API_KEY (或 GOOGLE_API_KEY)
# - PUBLIC_BASE_URL 例：https://xxxxx.onrender.com  （必須 https，LINE 才顯示圖）
#
# requirements.txt 至少：
# flask
# gunicorn
# line-bot-sdk
# google-genai
# =========================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fridge-bot")

# ---------------------
# LINE keys
# ---------------------
def load_line_keys(filepath: str = "keys.txt") -> Dict[str, str]:
    channel_secret = os.getenv("CHANNEL_SECRET")
    channel_access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
    if channel_secret and channel_access_token:
        return {"CHANNEL_SECRET": channel_secret, "CHANNEL_ACCESS_TOKEN": channel_access_token}

    p = Path(__file__).with_name(filepath)
    if not p.exists():
        raise RuntimeError("缺少 LINE CHANNEL_SECRET / CHANNEL_ACCESS_TOKEN（請設定環境變數或提供 keys.txt）")

    keys: Dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                keys[k.strip()] = v.strip()

    if "CHANNEL_SECRET" not in keys or "CHANNEL_ACCESS_TOKEN" not in keys:
        raise RuntimeError("keys.txt 內容不完整：需要 CHANNEL_SECRET 與 CHANNEL_ACCESS_TOKEN")

    return keys


line_keys = load_line_keys()
line_api = LineBotApi(line_keys["CHANNEL_ACCESS_TOKEN"])
handler = WebhookHandler(line_keys["CHANNEL_SECRET"])

# ---------------------
# Google GenAI (Gemini + Image)
# ---------------------
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("缺少 GEMINI_API_KEY（請在 Render / 本機設定環境變數）")

client = genai.Client(api_key=GEMINI_API_KEY)

TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash").strip()

# ✅ 你要「Gemini 生成食物示意圖」：用 gemini-*-image 類型模型
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image").strip()

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")
MAX_KEEP_IMAGES = int(os.getenv("MAX_KEEP_IMAGES", "200"))

# 做法圖最多生成幾步（建議 8~12；越大越慢）
MAX_STEP_IMAGES = int(os.getenv("MAX_STEP_IMAGES", "12"))

# Flex carousel 每則最多 bubbles（保守用 12；你也可用環境變數調）
FLEX_CAROUSEL_MAX_BUBBLES = int(os.getenv("FLEX_CAROUSEL_MAX_BUBBLES", "12"))

# ---------------------
# Flask static for generated images
# ---------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
GEN_DIR = STATIC_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_images():
    """保留最近 MAX_KEEP_IMAGES 張"""
    try:
        files = sorted(GEN_DIR.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[MAX_KEEP_IMAGES:]:
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


def _safe_json_loads(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _norm_token(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip().lower()


def _img_bytes_from_generated_image(generated_image) -> Optional[bytes]:
    """Imagen generate_images 回傳物件抓 bytes"""
    img_obj = getattr(generated_image, "image", None)
    if img_obj is None:
        return None

    b = getattr(img_obj, "image_bytes", None)
    if b is None:
        b = getattr(img_obj, "imageBytes", None)

    if b is None:
        return None

    if isinstance(b, bytes):
        return b
    if isinstance(b, str):
        try:
            return base64.b64decode(b)
        except Exception:
            return None
    return None


def save_image_and_get_url(img_bytes: bytes) -> Optional[str]:
    cleanup_old_images()
    fname = f"{uuid.uuid4().hex}.png"
    fpath = GEN_DIR / fname
    with fpath.open("wb") as f:
        f.write(img_bytes)

    # LINE Flex 圖片一定要 https 的公開網址
    if not PUBLIC_BASE_URL.startswith("https://"):
        return None
    return f"{PUBLIC_BASE_URL}/static/generated/{fname}"


def _extract_inline_image_bytes(resp) -> Optional[bytes]:
    """
    Gemini 影像模型：從回傳內容抓 inline_data.data（通常 base64）
    """
    parts = getattr(resp, "parts", None)

    if not parts:
        cands = getattr(resp, "candidates", None) or []
        if cands:
            content = getattr(cands[0], "content", None)
            parts = getattr(content, "parts", None) if content else None

    if not parts:
        return None

    for part in parts:
        inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline is None:
            continue
        data = getattr(inline, "data", None)
        if not data:
            continue
        try:
            return base64.b64decode(data) if isinstance(data, str) else data
        except Exception:
            return None
    return None


def generate_image_url(prompt: str) -> Optional[str]:
    """
    ✅ 用 Gemini 影像模型生成示意圖（IMAGE_MODEL=gemini-*）
    備援：若 IMAGE_MODEL 不是 gemini-*，用 Imagen generate_images
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return None

    # 沒 https 就不要回圖（LINE 不顯示）
    if not PUBLIC_BASE_URL.startswith("https://"):
        return None

    # ---- 1) Gemini 影像模型 ----
    if IMAGE_MODEL.startswith("gemini-"):
        cfg = None
        try:
            cfg = types.GenerateContentConfig(response_modalities=["IMAGE"])
            if hasattr(types, "ImageConfig"):
                cfg.image_config = types.ImageConfig(aspect_ratio="16:9")
        except Exception:
            cfg = None

        try:
            if cfg is not None:
                resp = client.models.generate_content(model=IMAGE_MODEL, contents=[prompt], config=cfg)
            else:
                resp = client.models.generate_content(model=IMAGE_MODEL, contents=[prompt])
        except TypeError:
            resp = client.models.generate_content(model=IMAGE_MODEL, contents=[prompt])

        img_bytes = _extract_inline_image_bytes(resp)
        if not img_bytes:
            return None
        return save_image_and_get_url(img_bytes)

    # ---- 2) Imagen（備援）----
    resp = client.models.generate_images(
        model=IMAGE_MODEL,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )
    gen_list = getattr(resp, "generated_images", None) or []
    if not gen_list:
        return None

    img_bytes = _img_bytes_from_generated_image(gen_list[0])
    if not img_bytes:
        return None

    return save_image_and_get_url(img_bytes)


# =========================================================
# 使用者狀態（記憶體，重啟會清空）
# =========================================================
user_fridge_map = defaultdict(dict)  # user_id -> {norm: display}
recent_recipes: Dict[str, List[Dict[str, Any]]] = {}
last_used_ings: Dict[str, List[str]] = {}
last_titles = defaultdict(list)


def fridge_list(user_id: str) -> List[str]:
    return list(user_fridge_map[user_id].values())


def fridge_text(user_id: str) -> str:
    items = fridge_list(user_id)
    return "你的冰箱目前：" + ("、".join(items) if items else "（空的）")


def add_to_fridge(user_id: str, items: List[str]) -> List[str]:
    added: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        nx = _norm_token(x)
        if not nx:
            continue
        if nx not in user_fridge_map[user_id]:
            user_fridge_map[user_id][nx] = x
            added.append(x)
    return added


def clear_fridge(user_id: str):
    user_fridge_map[user_id].clear()


def remove_from_fridge(user_id: str, items: List[str]) -> List[str]:
    removed: List[str] = []
    if not items:
        return removed

    targets = [_norm_token(x) for x in items if (x or "").strip()]
    targets = [t for t in targets if t]
    if not targets:
        return removed

    keys = list(user_fridge_map[user_id].keys())
    for k in keys:
        disp = user_fridge_map[user_id].get(k, "")
        for t in targets:
            if t == k or (t in k) or (k in t):
                removed.append(disp)
                user_fridge_map[user_id].pop(k, None)
                break
    return removed


# =========================================================
# 抽取食材（保底）
# =========================================================
SEPS = r"[\s、,，;；/｜|]+"

def heuristic_extract_ingredients(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []

    m = re.search(r"(我家有|冰箱裡有|冰箱有|我剩下|剩下|有)\s*(.*)$", t)
    if m:
        tail = (m.group(2) or "").strip()
        if tail:
            return [p.strip() for p in re.split(SEPS, tail) if p.strip()]

    parts = [p.strip() for p in re.split(SEPS, t) if p.strip()]
    bad = {"我", "家", "有", "冰箱", "剩下", "想", "煮", "做", "可以", "幫我", "一下"}
    return [p for p in parts if p not in bad]


# =========================================================
# Help / 判斷是否像食材輸入
# =========================================================
HELP_TRIGGERS = {"?", "help", "幫助", "說明"}

def reply_help(reply_token: str):
    msg = (
        "📌 冰箱清理小幫手使用方法\n\n"
        "✅ 加入食材（擇一）：\n"
        "1) 直接打食材：雞腿排 洋蔥\n"
        "2) 我家有/冰箱有：我家有 霜降牛小排 洋蔥\n"
        "3) 用 + 加：+ 雞腿排 洋蔥\n\n"
        "✅ 移除食材：\n"
        "- 雞腿排 洋蔥\n"
        "（或輸入「-」叫出移除選單）\n\n"
        "✅ 生成 3 道食譜：輸入「推薦」\n"
        "✅ 不喜歡：輸入「換食譜」（同一批食材換 3 道）\n"
        "✅ 看做法：做法 1 / 做法 2 / 做法 3（一次輸出全部步驟圖）\n\n"
        "👉 叫出選單：輸入 ? / help / 幫助（或輸入 +）"
    )
    safe_reply(reply_token, TextSendMessage(text=msg, quick_reply=make_quickreply_menu()))


def looks_like_ingredients_text(text: str) -> bool:
    """
    盡量不誤判：只要看起來像「列食材」就 True
    """
    t = (text or "").strip()
    if not t:
        return False

    # 指令直接排除
    cmd_words = {
        "推薦", "換食譜", "查看冰箱", "清空冰箱", "清空", "重置冰箱",
        "+", "-", "開啟按鈕選單", "按鈕選單", "選單", "menu", "MENU",
        "幫助", "說明", "help", "?"
    }
    if t in cmd_words:
        return False

    # 常見聊天/問題句排除
    bad_phrases = ["怎麼", "為什麼", "可以嗎", "要不要", "幫我", "教我", "哪裡", "多少", "什麼", "是不是", "早安", "晚安"]
    if any(p in t for p in bad_phrases):
        return False

    # 有「我家有」等關鍵字 => 高機率是食材
    if re.search(r"(我家有|冰箱裡有|冰箱有|我剩下|剩下)\s*", t):
        return True

    # 含網址通常不是食材
    if re.search(r"https?://|www\.", t):
        return False

    parts = [p.strip() for p in re.split(SEPS, t) if p.strip()]
    if not parts:
        return False

    # 2 個以上短詞，通常是列食材
    short_enough = len("".join(parts)) <= 30 and all(len(p) <= 12 for p in parts)
    if len(parts) >= 2 and short_enough:
        return True

    # 單一詞：必須是中文且短
    only_cjk = bool(re.fullmatch(r"[\u4e00-\u9fff]{1,10}", parts[0]))
    return only_cjk


# =========================================================
# Quick Reply（按鈕） - <= 13 items
# =========================================================
COMMON_INGS = ["雞肉", "牛肉", "豬肉", "雞蛋", "洋蔥", "大蒜", "蔥"]

def make_quickreply_menu() -> QuickReply:
    items = []
    for ing in COMMON_INGS[:6]:
        items.append(QuickReplyButton(action=MessageAction(label=f"+{ing}", text=f"+ {ing}")))

    # 功能鍵（不含 上一頁/下一頁）
    items.append(QuickReplyButton(action=MessageAction(label="🍳 推薦", text="推薦")))
    items.append(QuickReplyButton(action=MessageAction(label="🔁 換食譜", text="換食譜")))
    items.append(QuickReplyButton(action=MessageAction(label="➖ 用完", text="-")))
    items.append(QuickReplyButton(action=MessageAction(label="📦 查看冰箱", text="查看冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="🗑 清空", text="清空冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="❓ 幫助", text="幫助")))
    return QuickReply(items=items)


def make_remove_quickreply(user_id: str) -> QuickReply:
    """
    最多 7 個食材 + 6 個功能 = 13
    """
    items = []
    current = fridge_list(user_id)[:7]
    for ing in current:
        items.append(QuickReplyButton(action=MessageAction(label=f"➖{ing}", text=f"- {ing}")))

    items.append(QuickReplyButton(action=MessageAction(label="🍳 推薦", text="推薦")))
    items.append(QuickReplyButton(action=MessageAction(label="🔁 換食譜", text="換食譜")))
    items.append(QuickReplyButton(action=MessageAction(label="📦 查看冰箱", text="查看冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="🗑 清空", text="清空冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="➕ 選單", text="+")))
    items.append(QuickReplyButton(action=MessageAction(label="❓ 幫助", text="幫助")))
    return QuickReply(items=items)


# =========================================================
# LINE 安全回覆（避免 400 造成 webhook 500）
# =========================================================
def safe_reply(reply_token: str, messages):
    """
    LINE reply 失敗常見原因：QuickReply items > 13、Flex 結構限制等。
    這裡保底：失敗就改成純文字回覆，不讓 webhook 直接 500。
    """
    try:
        line_api.reply_message(reply_token, messages)
    except LineBotApiError as e:
        log.exception("Line reply error: %s", e)
        # fallback：只回純文字（沒有 quick reply）
        try:
            if isinstance(messages, list) and messages:
                text = "發送失敗（可能是按鈕/圖卡限制）。\n請輸入「幫助」查看用法。"
            else:
                text = "發送失敗。\n請輸入「幫助」查看用法。"
            line_api.reply_message(reply_token, TextSendMessage(text=text))
        except Exception:
            pass


# =========================================================
# Gemini：抽食材 + 生成食譜（剛好 3 道）
# =========================================================
def gemini_generate_recipes(
    user_input: str,
    fridge_items: List[str],
    avoid_titles_in: List[str],
    n_recipes: int = 3,
) -> Dict[str, Any]:
    fridge_items = fridge_items or []
    avoid_titles_in = avoid_titles_in or []

    prompt = f"""
請只輸出 JSON（不要任何其他文字）。使用繁體中文（只有 image_prompt 用英文）。
你是料理助理。

【使用者輸入】
{user_input}

【目前冰箱已記錄食材（使用者原寫法，可能含部位/品項）】
{ "、".join(fridge_items) if fridge_items else "（空）" }

【要求 JSON 格式】
{{
  "ingredients": ["抽取/推斷到的食材（中文，去掉數量與單位，去重）"],
  "recipes": [
    {{
      "name": "菜名（中文）",
      "summary": "一句話介紹（中文）",
      "ingredients": ["關鍵食材（中文，盡量沿用 ingredients 裡的寫法，例如：霜降牛小排、雞腿排）"],
      "steps": ["步驟1（中文）","步驟2（中文）", "...至少 5 步"],
      "image_prompt": "English prompt for a photorealistic food photo of this dish, plated nicely, natural lighting, shallow depth of field, no text"
    }}
  ]
}}

【重要規則（請務必遵守）】
- ingredients 請「盡量保留使用者輸入的寫法與部位名稱」，不要把『霜降牛小排』改成『牛肉』；除非使用者本來就只寫『牛肉』
- recipes 必須「剛好 {n_recipes} 道」，每一道要明顯不同（菜名/做法不同）
- 若食材很少也要想辦法做出 {n_recipes} 道家常料理（可用常見調味料默認存在，但不要硬塞奇怪食材）
- 避免產出與以下菜名相同或高度相似的菜名：{ "、".join(avoid_titles_in[:12]) if avoid_titles_in else "（無）" }
- steps 至少 5 步，語句要讓人一看就能做
- image_prompt 務必英文，且能清楚呈現成品
"""

    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.6),
    )
    data = _safe_json_loads(getattr(resp, "text", "") or "")
    if not isinstance(data, dict):
        raise RuntimeError("Gemini 回覆不是有效 JSON")

    data.setdefault("ingredients", [])
    data.setdefault("recipes", [])

    recipes = data.get("recipes") or []
    if not isinstance(recipes, list):
        recipes = []

    # 補問（最多 2 次）
    tries = 0
    while len(recipes) < n_recipes and tries < 2:
        tries += 1
        prompt2 = f"""
只輸出 JSON（不要任何其他文字）。
用這些食材生成「剛好 {n_recipes} 道」recipes（同上格式），且避開菜名：{avoid_titles_in + [r.get("name","") for r in recipes if isinstance(r, dict)]}
食材：{sorted(set(fridge_items + (data.get("ingredients") or [])))}
並且仍要保留部位/品項寫法，不要泛化成大分類。
"""
        resp2 = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt2,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7),
        )
        d2 = _safe_json_loads(getattr(resp2, "text", "") or "")
        r2 = (d2.get("recipes") or []) if isinstance(d2, dict) else []
        if isinstance(r2, list):
            seen = {(_norm_token(r.get("name", ""))) for r in recipes if isinstance(r, dict)}
            for r in r2:
                if not isinstance(r, dict):
                    continue
                nm = _norm_token(r.get("name", ""))
                if nm and nm not in seen:
                    recipes.append(r)
                    seen.add(nm)

    data["recipes"] = recipes[:n_recipes]

    # ingredients 去重
    ings = data.get("ingredients") or []
    if not isinstance(ings, list):
        ings = []
    ings2, seen2 = [], set()
    for x in ings:
        x = (str(x) or "").strip()
        nx = _norm_token(x)
        if nx and nx not in seen2:
            seen2.add(nx)
            ings2.append(x)
    data["ingredients"] = ings2

    return data


# =========================================================
# Gemini：為每一步產生「英文步驟圖 prompt」
# =========================================================
def gemini_steps_with_prompts(recipe_name: str, steps: List[str]) -> List[Dict[str, str]]:
    steps = steps or []
    prompt = f"""
只輸出 JSON（不要任何其他文字）。
你要把每個步驟改寫得更清楚（繁體中文），並為每個步驟提供「英文」示意圖 prompt（教學感、手在做事、看圖就懂，不要有文字/水印）。

菜名：{recipe_name}
步驟（原始）：{steps}

輸出格式：
{{
  "steps": [
    {{
      "text": "中文步驟（清楚簡短）",
      "image_prompt": "English prompt for a photorealistic instructional cooking image showing THIS step in action (hands, utensils, ingredients), kitchen setting, natural lighting, no text, no watermark"
    }}
  ]
}}

規則：
- steps 數量要與原始步驟一致
- image_prompt 一定要英文
"""
    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.5),
    )
    data = _safe_json_loads(getattr(resp, "text", "") or "")
    if not isinstance(data, dict) or "steps" not in data or not isinstance(data["steps"], list):
        raise RuntimeError("Gemini 無法產生步驟 prompts JSON")

    out: List[Dict[str, str]] = []
    for s in data["steps"]:
        if not isinstance(s, dict):
            continue
        t = (s.get("text") or "").strip()
        p = (s.get("image_prompt") or "").strip()
        if t:
            out.append({"text": t, "image_prompt": p})
    return out


# =========================================================
# Flex：食譜卡 & 步驟卡（一次輸出全部，但自動切段）
# =========================================================
def recipe_to_bubble(rank: int, recipe: Dict[str, Any], image_url: Optional[str]) -> Dict[str, Any]:
    name = recipe.get("name", f"料理 {rank}")
    summary = recipe.get("summary", "")
    ings = recipe.get("ingredients") or []
    if isinstance(ings, list):
        ing_text = "、".join([str(x) for x in ings[:12] if str(x).strip()]) + ("…" if len(ings) > 12 else "")
    else:
        ing_text = "—"

    bubble: Dict[str, Any] = {
        "type": "bubble",
        "size": "mega",
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {"type": "text", "text": f"{rank}. {name}", "wrap": True, "weight": "bold", "size": "lg"},
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {"type": "button", "style": "primary", "color": "#1DB446",
                 "action": {"type": "message", "label": f"看做法({rank})", "text": f"做法 {rank}"}},
                {"type": "button", "style": "secondary",
                 "action": {"type": "message", "label": "換食譜", "text": "換食譜"}},
            ],
        },
    }

    if image_url:
        bubble["hero"] = {"type": "image", "url": image_url, "size": "full",
                         "aspectRatio": "16:9", "aspectMode": "cover"}

    if summary:
        bubble["body"]["contents"].append({"type": "text", "text": summary, "wrap": True, "size": "sm"})
    bubble["body"]["contents"].append({"type": "text", "text": f"🧾 食材：{ing_text}", "wrap": True, "size": "sm"})
    return bubble


def step_to_bubble(step_no: int, step_text: str, image_url: Optional[str]) -> Dict[str, Any]:
    bubble: Dict[str, Any] = {
        "type": "bubble",
        "size": "mega",
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {"type": "text", "text": f"步驟 {step_no}", "weight": "bold", "size": "lg"},
                {"type": "text", "text": step_text, "wrap": True, "size": "sm"},
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {"type": "button", "style": "secondary",
                 "action": {"type": "message", "label": "換食譜", "text": "換食譜"}},
            ],
        },
    }
    if image_url:
        bubble["hero"] = {"type": "image", "url": image_url, "size": "full",
                         "aspectRatio": "16:9", "aspectMode": "cover"}
    return bubble


def chunk_list(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def steps_to_flex_messages_all(step_items: List[Dict[str, str]], recipe_name: str) -> List[FlexSendMessage]:
    """
    一次輸出全部步驟圖：
    - 每個 carousel bubble 數有限制，所以切成多則 Flex
    - LINE reply 一次最多 5 則訊息：通常 header 佔 1，所以 Flex 最多 4
    """
    if not step_items:
        return []

    n = max(1, min(FLEX_CAROUSEL_MAX_BUBBLES, 12))
    max_flex_msgs = 4
    max_steps = max_flex_msgs * n

    step_items = step_items[:max_steps]
    chunks = chunk_list(step_items, n)

    flex_msgs: List[FlexSendMessage] = []
    total = len(step_items)
    start_index = 1
    for chunk in chunks[:max_flex_msgs]:
        bubbles = []
        for it in chunk:
            bubbles.append(step_to_bubble(start_index, it["text"], it.get("image_url")))
            start_index += 1
        flex_msgs.append(
            FlexSendMessage(
                alt_text=f"{recipe_name} 步驟圖（{start_index - len(chunk)}-{start_index - 1}/{total}）",
                contents={"type": "carousel", "contents": bubbles},
            )
        )
    return flex_msgs


# =========================================================
# 推薦 / 換食譜（每道菜 1 圖）
# =========================================================
def reply_recipes(user_id: str, reply_token: str, user_text: str, force_same_ingredients: bool = False):
    try:
        if force_same_ingredients:
            base_ings = last_used_ings.get(user_id) or fridge_list(user_id)
            if not base_ings:
                safe_reply(
                    reply_token,
                    TextSendMessage(
                        text="你還沒有可用食材～先輸入：『雞腿排 洋蔥』或用『+ 雞腿排』加入吧！",
                        quick_reply=make_quickreply_menu(),
                    ),
                )
                return

            data = gemini_generate_recipes(
                user_input=f"請用同一批食材換一組新食譜：{'、'.join(base_ings)}",
                fridge_items=base_ings,
                avoid_titles_in=last_titles[user_id],
                n_recipes=3,
            )
        else:
            data = gemini_generate_recipes(
                user_input=user_text,
                fridge_items=fridge_list(user_id),
                avoid_titles_in=[],
                n_recipes=3,
            )

        # 把抽到的食材加入冰箱（保留部位）
        extracted = data.get("ingredients") or []
        extracted = [str(x).strip() for x in extracted if str(x).strip()]
        if extracted:
            add_to_fridge(user_id, extracted)

        use_ings = fridge_list(user_id)
        last_used_ings[user_id] = use_ings

        recipes = data.get("recipes") or []
        if not isinstance(recipes, list) or len(recipes) < 3:
            raise RuntimeError("Gemini 沒產出足夠的食譜（少於 3 道）")

        bubbles = []
        titles = []
        final_recipes: List[Dict[str, Any]] = []

        for i, r in enumerate(recipes[:3], start=1):
            if not isinstance(r, dict):
                continue
            name = r.get("name", f"料理 {i}")
            titles.append(name)

            img_prompt = (r.get("image_prompt") or "").strip()
            if not img_prompt:
                img_prompt = (
                    f"Photorealistic food photo of {name}, plated nicely, natural lighting, "
                    "shallow depth of field, no text, no watermark"
                )

            dish_img_url = None
            try:
                dish_img_url = generate_image_url(img_prompt)
            except Exception:
                dish_img_url = None

            bubbles.append(recipe_to_bubble(i, r, dish_img_url))
            final_recipes.append(r)

        if len(final_recipes) < 3:
            raise RuntimeError("食譜資料格式異常（不足 3 道有效食譜）")

        recent_recipes[user_id] = final_recipes
        last_titles[user_id] = titles

        text_msg = TextSendMessage(
            text=(
                f"✅ 目前食材：{'、'.join(use_ings) if use_ings else '（空）'}\n"
                f"{fridge_text(user_id)}\n\n"
                "我給你 3 個選項～\n"
                "📌 看做法：輸入『做法 1/2/3』或點卡片按鈕\n"
                "🔁 不喜歡：按『換食譜』再換一批\n"
                "➖ 用完食材：輸入『- 雞腿排』或輸入『-』叫出移除選單\n"
                "❓ 需要說明：輸入『? / help / 幫助』或『+』"
            ),
            quick_reply=make_quickreply_menu(),
        )
        flex_msg = FlexSendMessage(
            alt_text="推薦料理（含示意圖）",
            contents={"type": "carousel", "contents": bubbles},
        )
        safe_reply(reply_token, [text_msg, flex_msg])

    except Exception as e:
        safe_reply(
            reply_token,
            TextSendMessage(
                text=(
                    f"Google 生成時出錯：{type(e).__name__}: {e}\n\n"
                    "你可以試：\n"
                    "1) 直接輸入食材：霜降牛小排 洋蔥\n"
                    "2) 用 + 加：+ 雞腿排 洋蔥\n"
                    "3) 推薦\n\n"
                    "（若看到 API key leaked/403：請換新的 GEMINI_API_KEY 並更新 Render 環境變數）"
                ),
                quick_reply=make_quickreply_menu(),
            ),
        )


# =========================================================
# 做法：每步驟一張圖（一次輸出全部，但自動切段）
# =========================================================
def reply_steps_with_images(user_id: str, reply_token: str, recipe_idx: int):
    if user_id not in recent_recipes:
        safe_reply(
            reply_token,
            TextSendMessage(text="你還沒有推薦清單～先輸入食材並按『推薦』。", quick_reply=make_quickreply_menu()),
        )
        return

    recipes = recent_recipes[user_id]
    if not (0 <= recipe_idx < len(recipes)):
        safe_reply(
            reply_token,
            TextSendMessage(text="這個編號不在清單內～請輸入『做法 1/2/3』。", quick_reply=make_quickreply_menu()),
        )
        return

    recipe = recipes[recipe_idx]
    recipe_name = recipe.get("name", f"料理 {recipe_idx+1}")
    steps = recipe.get("steps") or []
    if not isinstance(steps, list) or not steps:
        safe_reply(
            reply_token,
            TextSendMessage(text=f"《{recipe_name}》沒有步驟內容。你可以按『換食譜』換一批。", quick_reply=make_quickreply_menu()),
        )
        return

    # 先用 Gemini 產 prompts，再生成圖片
    try:
        step_objs = gemini_steps_with_prompts(recipe_name, steps)
    except Exception as e:
        safe_reply(
            reply_token,
            TextSendMessage(text=f"步驟圖 prompt 產生失敗：{type(e).__name__}: {e}", quick_reply=make_quickreply_menu()),
        )
        return

    # 遵守 MAX_STEP_IMAGES（同時也不超過 reply 4 則 Flex 的最大容量）
    max_by_flex = 4 * max(1, min(FLEX_CAROUSEL_MAX_BUBBLES, 12))
    hard_cap = max(1, min(MAX_STEP_IMAGES, max_by_flex))
    step_objs = step_objs[:hard_cap]

    step_texts: List[str] = []
    img_urls: List[Optional[str]] = []

    for s in step_objs:
        t = (s.get("text") or "").strip()
        p = (s.get("image_prompt") or "").strip()
        if not t:
            continue
        step_texts.append(t)

        url = None
        try:
            if not p:
                p = (
                    f"Photorealistic instructional cooking image showing a step in action for {recipe_name}, "
                    "hands, utensils, ingredients, kitchen, natural lighting, no text, no watermark"
                )
            url = generate_image_url(p)
        except Exception:
            url = None
        img_urls.append(url)

    if not step_texts:
        safe_reply(
            reply_token,
            TextSendMessage(text=f"《{recipe_name}》步驟整理失敗，請按『換食譜』或再試一次『做法 {recipe_idx+1}』。", quick_reply=make_quickreply_menu()),
        )
        return

    step_items = [{"text": t, "image_url": u} for t, u in zip(step_texts, img_urls)]
    flex_msgs = steps_to_flex_messages_all(step_items, recipe_name)

    shown = len(step_items)
    header = TextSendMessage(
        text=(
            f"《{recipe_name}》步驟示意圖（一次輸出全部）\n"
            f"（本次顯示 {shown} 步；上限 MAX_STEP_IMAGES={MAX_STEP_IMAGES}；每則最多 {FLEX_CAROUSEL_MAX_BUBBLES} 張圖）\n"
            "不喜歡可按『換食譜』換一批。"
        ),
        quick_reply=make_quickreply_menu(),
    )

    safe_reply(reply_token, [header] + flex_msgs)


# =========================================================
# Flask
# =========================================================
app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.get("/")
def index():
    return "OK"


@app.get("/healthz")
def healthz():
    return "healthy"


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        # 不讓 webhook 因為內部錯誤回 500（LINE 會一直重送）
        log.exception("Callback error: %s", e)
    return "OK"


@handler.add(FollowEvent)
def handle_follow(event: FollowEvent):
    welcome = (
        "嗨～我是冰箱清理小幫手（Google 版）！\n\n"
        "✅ 先把食材加進冰箱：\n"
        "・直接打：雞腿排 洋蔥\n"
        "・或用：+ 雞腿排 洋蔥\n\n"
        "✅ 要食譜：輸入『推薦』(我會給 3 道 + 料理示意圖)\n"
        "✅ 不喜歡：按『換食譜』\n"
        "✅ 看做法：輸入『做法 1』(一次輸出全部步驟圖)\n"
        "✅ 用完食材：輸入『- 雞腿排』或輸入『-』叫出移除選單\n"
        "✅ 需要選單/說明：輸入『? / help / 幫助』或『+』"
    )
    safe_reply(event.reply_token, TextSendMessage(text=welcome, quick_reply=make_quickreply_menu()))


@handler.add(MessageEvent, message=TextMessage)
def handle_text(event: MessageEvent):
    user_id = event.source.user_id
    text = (event.message.text or "").strip()
    t_low = text.lower()

    # 1) help：? / help / 幫助 => 叫出選單 + 用法
    if text in HELP_TRIGGERS or t_low in HELP_TRIGGERS:
        reply_help(event.reply_token)
        return

    # 2) + 單獨 => 叫出選單（不是加食材）
    if text in {"+", "開啟按鈕選單", "按鈕選單", "選單", "menu", "MENU"}:
        safe_reply(
            event.reply_token,
            TextSendMessage(
                text="這是按鈕選單～你可以快速加入/推薦/換食譜/移除食材 👇",
                quick_reply=make_quickreply_menu(),
            ),
        )
        return

    # 3) - 單獨 => 叫出移除選單
    if text in {"-", "用完食材", "移除食材", "刪食材", "減食材"}:
        if not fridge_list(user_id):
            safe_reply(
                event.reply_token,
                TextSendMessage(text="你的冰箱目前是空的～不用移除囉！", quick_reply=make_quickreply_menu()),
            )
            return
        safe_reply(
            event.reply_token,
            TextSendMessage(
                text="你可以點下面按鈕移除已用完的食材，或直接輸入：- 霜降牛小排 洋蔥",
                quick_reply=make_remove_quickreply(user_id),
            ),
        )
        return

    # 4) + 食材 => 加入
    m_plus = re.match(r"^\+\s*(.+)$", text)
    if m_plus:
        raw = m_plus.group(1).strip()
        parts = [p.strip() for p in re.split(SEPS, raw) if p.strip()]
        if not parts:
            safe_reply(
                event.reply_token,
                TextSendMessage(text="請輸入：+ 雞腿排 洋蔥（可一次加入多個）", quick_reply=make_quickreply_menu()),
            )
            return
        added = add_to_fridge(user_id, parts)
        msg = f"✅ 已加入：{'、'.join(added)}\n{fridge_text(user_id)}" if added else f"這些已經在冰箱裡了～\n{fridge_text(user_id)}"
        safe_reply(event.reply_token, TextSendMessage(text=msg, quick_reply=make_quickreply_menu()))
        return

    # 5) - 食材 => 移除
    m_minus = re.match(r"^-\s*(.+)$", text)
    if m_minus:
        raw = m_minus.group(1).strip()
        parts = [p.strip() for p in re.split(SEPS, raw) if p.strip()]
        if not parts:
            safe_reply(
                event.reply_token,
                TextSendMessage(text="請輸入：- 雞腿排 洋蔥（可一次移除多個）", quick_reply=make_remove_quickreply(user_id)),
            )
            return
        removed = remove_from_fridge(user_id, parts)
        if removed:
            safe_reply(
                event.reply_token,
                TextSendMessage(text=f"已移除：{'、'.join(removed)}\n{fridge_text(user_id)}", quick_reply=make_quickreply_menu()),
            )
        else:
            safe_reply(
                event.reply_token,
                TextSendMessage(text=f"我沒有在冰箱裡找到：{'、'.join(parts)}\n{fridge_text(user_id)}", quick_reply=make_quickreply_menu()),
            )
        return

    # 6) 查看/清空
    if text in {"查看冰箱", "冰箱", "我的冰箱"}:
        safe_reply(event.reply_token, TextSendMessage(text=fridge_text(user_id), quick_reply=make_quickreply_menu()))
        return

    if text in {"清空冰箱", "清空", "重置冰箱", "清空全部"}:
        clear_fridge(user_id)
        recent_recipes.pop(user_id, None)
        last_used_ings.pop(user_id, None)
        last_titles.pop(user_id, None)
        safe_reply(
            event.reply_token,
            TextSendMessage(text="🗑 已清空冰箱！\n你的冰箱目前：（空的）", quick_reply=make_quickreply_menu()),
        )
        return

    # 7) 做法 N
    m_steps = re.match(r"^(做法)\s*(\d+)\s*$", text)
    if m_steps:
        idx = int(m_steps.group(2)) - 1
        reply_steps_with_images(user_id, event.reply_token, recipe_idx=idx)
        return

    # 8) 換食譜 / 推薦
    if text in {"換食譜", "換", "重新推薦", "再推薦"}:
        reply_recipes(user_id, event.reply_token, user_text=text, force_same_ingredients=True)
        return

    if text in {"推薦", "給我食譜", "食譜", "煮什麼", "今天煮什麼"}:
        # 若冰箱空的，先提醒加食材
        if not fridge_list(user_id):
            safe_reply(
                event.reply_token,
                TextSendMessage(
                    text="你冰箱還是空的～先輸入食材（例如：雞腿排 洋蔥），再按『推薦』我就能出 3 道菜！",
                    quick_reply=make_quickreply_menu(),
                ),
            )
            return
        reply_recipes(user_id, event.reply_token, user_text=text, force_same_ingredients=False)
        return

    # 9) 直接打食材（不自動推薦，只加入）
    if looks_like_ingredients_text(text):
        parts = heuristic_extract_ingredients(text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            reply_help(event.reply_token)
            return

        added = add_to_fridge(user_id, parts)
        if added:
            msg = f"✅ 已加入冰箱：{'、'.join(added)}\n{fridge_text(user_id)}\n\n接著輸入「推薦」我會給你 3 道菜（含示意圖）～"
        else:
            msg = f"這些食材可能已經在冰箱裡了～\n{fridge_text(user_id)}\n\n輸入「推薦」我會給你 3 道菜（含示意圖）～"

        safe_reply(event.reply_token, TextSendMessage(text=msg, quick_reply=make_quickreply_menu()))
        return

    # 10) 非食材內容 => 跳 help + 選單
    reply_help(event.reply_token)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
