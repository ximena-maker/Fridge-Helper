# app.py
import os
import re
import json
import uuid
import base64
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

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

# Google GenAI SDK (pip install google-genai)
from google import genai
from google.genai import types

# =========================================================
# 冰箱清理小幫手（LINE Bot）- 全部用 Google Gemini + Imagen
#
# ✅ 任何食材/部位/品項都能輸入（例如：霜降牛小排、雞腿排、松阪豬、干貝、金針菇…）
# ✅ 我家有 xxx / 隨便一句話：Gemini 抽食材（盡量保留原本寫法）+ 產生至少 3 道食譜
# ✅ + 或 開啟按鈕選單：叫出 Quick Reply
# ✅ - 或 用完食材：開啟移除選單；也可直接輸入 - 食材1 食材2
# ✅ 每道菜 1 張示意圖（Imagen）
# ✅ 做法 N：每一步 1 張示意圖 + 翻頁
#
# 必要環境變數（Render / 本機）：
# - CHANNEL_SECRET
# - CHANNEL_ACCESS_TOKEN
# - GEMINI_API_KEY   (或 GOOGLE_API_KEY)
# - PUBLIC_BASE_URL  例：https://fridge-helper.onrender.com  （必須 https，LINE 才顯示圖）
#
# requirements.txt 至少：
# flask
# gunicorn
# line-bot-sdk
# google-genai
# =========================================================


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
# Google GenAI (Gemini + Imagen)
# ---------------------
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("缺少 GEMINI_API_KEY（請在 Render / 本機設定環境變數）")

client = genai.Client(api_key=GEMINI_API_KEY)

TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "imagen-4.0-generate-001").strip()

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")
MAX_KEEP_IMAGES = int(os.getenv("MAX_KEEP_IMAGES", "200"))
MAX_STEP_IMAGES = int(os.getenv("MAX_STEP_IMAGES", "12"))  # 做法圖一次最多先生成幾步


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
    # 用於比對/去重：去空白、全小寫
    return re.sub(r"\s+", "", (s or "")).strip().lower()


def _img_bytes_from_generated_image(generated_image) -> Optional[bytes]:
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

    # LINE 要顯示圖片必須是 https 的公開網址
    if not PUBLIC_BASE_URL.startswith("https://"):
        return None
    return f"{PUBLIC_BASE_URL}/static/generated/{fname}"


def generate_image_url(prompt: str) -> Optional[str]:
    prompt = (prompt or "").strip()
    if not prompt:
        return None

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
# 冰箱：用「normalized -> 顯示字串」保留你輸入的原樣（部位也保留）
user_fridge_map = defaultdict(dict)  # user_id -> {norm: display}
recent_recipes: Dict[str, List[Dict[str, Any]]] = {}     # user_id -> list[recipe dict] (至少 3)
last_used_ings: Dict[str, List[str]] = {}                # user_id -> list[str]  上次用的食材（顯示字串）
last_titles = defaultdict(list)                          # user_id -> list[str]  上次菜名（避開）
step_view_state: Dict[str, Dict[str, Any]] = {}          # user_id -> {recipe_idx, recipe_name, steps, img_urls, page}


def fridge_list(user_id: str) -> List[str]:
    return list(user_fridge_map[user_id].values())


def fridge_text(user_id: str) -> str:
    items = fridge_list(user_id)
    return "你的冰箱目前：" + ("、".join(items) if items else "（空的）")


def add_to_fridge(user_id: str, items: List[str]) -> List[str]:
    """回傳實際新增的（顯示字串）"""
    added: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        nx = _norm_token(x)
        if not nx:
            continue
        if nx not in user_fridge_map[user_id]:
            user_fridge_map[user_id][nx] = x  # 保留原本寫法（部位/品牌/品項）
            added.append(x)
    return added


def clear_fridge(user_id: str):
    user_fridge_map[user_id].clear()


def remove_from_fridge(user_id: str, items: List[str]) -> List[str]:
    """
    從冰箱移除 items（支援模糊：牛小排 可以移除 霜降牛小排）
    回傳實際移除到的（顯示字串）
    """
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
            # 完全相等 / 互為包含 皆視為同一個（讓部位更好移除）
            if t == k or (t in k) or (k in t):
                removed.append(disp)
                user_fridge_map[user_id].pop(k, None)
                break

    return removed


# =========================================================
# 快速把使用者輸入拆成「候選食材」（保底抽取，不限制任何食材）
# =========================================================
SEPS = r"[\s、,，;；/｜|]+"


def heuristic_extract_ingredients(text: str) -> List[str]:
    """
    保底抽取（不限制輸入）：偵測「我家有/冰箱有/剩下/有」後面的字串，或直接整句拆詞。
    會盡量把「雞腿肉」「霜降牛小排」這種保留為一個 token（前提是使用者有用分隔符分開）
    """
    t = (text or "").strip()
    if not t:
        return []

    m = re.search(r"(我家有|冰箱裡有|冰箱有|我剩下|剩下|有)\s*(.*)$", t)
    if m:
        tail = (m.group(2) or "").strip()
        if tail:
            parts = [p.strip() for p in re.split(SEPS, tail) if p.strip()]
            return parts

    parts = [p.strip() for p in re.split(SEPS, t) if p.strip()]
    bad = {"我", "家", "有", "冰箱", "剩下", "想", "煮", "做", "可以", "幫我", "一下"}
    parts = [p for p in parts if p not in bad]
    return parts


# =========================================================
# Quick Reply（按鈕）
# =========================================================
COMMON_INGS = ["雞肉", "牛肉", "豬肉", "雞蛋", "洋蔥", "大蒜", "蔥"]  # 只做快捷，不限制輸入


def make_quickreply_menu() -> QuickReply:
    items = []
    for ing in COMMON_INGS[:6]:
        items.append(QuickReplyButton(action=MessageAction(label=f"+{ing}", text=f"加入 {ing}")))

    items.append(QuickReplyButton(action=MessageAction(label="🍳 推薦", text="推薦")))
    items.append(QuickReplyButton(action=MessageAction(label="🔁 換食譜", text="換食譜")))
    items.append(QuickReplyButton(action=MessageAction(label="➖ 用完", text="-")))
    items.append(QuickReplyButton(action=MessageAction(label="⬅ 上一頁", text="上一頁")))
    items.append(QuickReplyButton(action=MessageAction(label="下一頁 ➡", text="下一頁")))
    items.append(QuickReplyButton(action=MessageAction(label="📦 查看冰箱", text="查看冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="🗑 清空", text="清空冰箱")))
    return QuickReply(items=items)


def make_remove_quickreply(user_id: str) -> QuickReply:
    """
    顯示「點一下就移除」：最多 10 個目前冰箱食材 + 幾個功能
    """
    items = []
    current = fridge_list(user_id)[:10]
    for ing in current:
        items.append(QuickReplyButton(action=MessageAction(label=f"➖{ing}", text=f"- {ing}")))

    items.append(QuickReplyButton(action=MessageAction(label="🍳 推薦", text="推薦")))
    items.append(QuickReplyButton(action=MessageAction(label="🔁 換食譜", text="換食譜")))
    items.append(QuickReplyButton(action=MessageAction(label="📦 查看冰箱", text="查看冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="➕ 按鈕選單", text="+")))
    return QuickReply(items=items)


# =========================================================
# Gemini：抽食材 + 生成食譜（至少 3 道）
# =========================================================
def gemini_generate_recipes(
    user_input: str,
    fridge_items: List[str],
    avoid_titles_in: List[str],
    n_recipes: int = 3,
) -> Dict[str, Any]:
    """
    回傳：
    {
      "ingredients": ["盡量保留使用者原寫法/部位名稱"],
      "recipes": [
        {"name":..., "summary":..., "ingredients":[...], "steps":[...], "image_prompt": "...English..."}
      ]
    }
    """
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
- ingredients 請「盡量保留使用者輸入的寫法與部位名稱」，不要自動把『霜降牛小排』改成『牛肉』；除非使用者本來就只寫『牛肉』
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

    # 少於 n_recipes 則補問（最多補 2 次）
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

    # ingredients 去重（保留原字串）
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
# Flex：食譜卡 & 步驟卡
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
                {
                    "type": "button",
                    "style": "primary",
                    "color": "#1DB446",
                    "action": {"type": "message", "label": f"看做法({rank})", "text": f"做法 {rank}"},
                }
            ],
        },
    }

    if image_url:
        bubble["hero"] = {
            "type": "image",
            "url": image_url,
            "size": "full",
            "aspectRatio": "16:9",
            "aspectMode": "cover",
        }

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
    }
    if image_url:
        bubble["hero"] = {
            "type": "image",
            "url": image_url,
            "size": "full",
            "aspectRatio": "16:9",
            "aspectMode": "cover",
        }
    return bubble


def steps_to_flex(step_items: List[Dict[str, str]], page: int, page_size: int = 5) -> FlexSendMessage:
    total = len(step_items)
    start = page * page_size
    end = min(start + page_size, total)

    bubbles = []
    for i in range(start, end):
        bubbles.append(step_to_bubble(i + 1, step_items[i]["text"], step_items[i].get("image_url")))

    return FlexSendMessage(
        alt_text=f"料理步驟圖（{start+1}-{end}/{total}）",
        contents={"type": "carousel", "contents": bubbles},
    )


# =========================================================
# 推薦 / 換食譜（每道菜 1 圖）
# =========================================================
def reply_recipes(user_id: str, reply_token: str, user_text: str, force_same_ingredients: bool = False):
    try:
        if force_same_ingredients:
            base_ings = last_used_ings.get(user_id) or fridge_list(user_id)
            if not base_ings:
                line_api.reply_message(
                    reply_token,
                    TextSendMessage(
                        text="你還沒有可用食材～先輸入：『我家有 霜降牛小排 洋蔥』或用『加入 雞腿排』加入吧！",
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

        # 把 Gemini 抽到的食材加入冰箱（保留部位/原寫法）
        extracted = data.get("ingredients") or []
        extracted = [str(x).strip() for x in extracted if str(x).strip()]
        if extracted:
            add_to_fridge(user_id, extracted)
        else:
            # 如果 Gemini 沒回 ingredients，就用保底拆詞把「疑似食材」加進去
            fallback = heuristic_extract_ingredients(user_text)
            if fallback:
                add_to_fridge(user_id, fallback)

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
                img_prompt = f"A high-quality photorealistic food photo of {name}, plated nicely, natural lighting, shallow depth of field, no text"

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
        step_view_state.pop(user_id, None)

        text_msg = TextSendMessage(
            text=(
                f"✅ 使用/記錄食材：{'、'.join(use_ings) if use_ings else '（未偵測到）'}\n"
                f"{fridge_text(user_id)}\n\n"
                "我給你 3 個選項～\n"
                "📌 看做法（含步驟圖）：輸入『做法 1/2/3』\n"
                "🔁 不喜歡：輸入/按『換食譜』再換一批\n"
                "➖ 用完食材：輸入『- 雞腿排』或直接輸入『-』叫出移除選單\n"
                "➕ 叫出按鈕：輸入『+』或『開啟按鈕選單』"
            ),
            quick_reply=make_quickreply_menu(),
        )
        flex_msg = FlexSendMessage(
            alt_text="推薦料理（含示意圖）",
            contents={"type": "carousel", "contents": bubbles},
        )
        line_api.reply_message(reply_token, [text_msg, flex_msg])

    except Exception as e:
        line_api.reply_message(
            reply_token,
            TextSendMessage(
                text=(
                    f"Google 生成時出錯：{type(e).__name__}: {e}\n\n"
                    "你可以試：\n"
                    "1) 我家有 霜降牛小排 洋蔥\n"
                    "2) 加入 雞腿排 洋蔥\n"
                    "3) 推薦\n\n"
                    "（若看到 API key leaked/403：請換新的 GEMINI_API_KEY，並更新 Render 環境變數）"
                ),
                quick_reply=make_quickreply_menu(),
            ),
        )


# =========================================================
# 做法：每步驟一張圖（Imagen），支援翻頁
# =========================================================
def reply_steps_with_images(user_id: str, reply_token: str, recipe_idx: int):
    if user_id not in recent_recipes:
        line_api.reply_message(
            reply_token,
            TextSendMessage(text="你還沒有推薦清單～先輸入食材或『推薦』。", quick_reply=make_quickreply_menu()),
        )
        return

    recipes = recent_recipes[user_id]
    if not (0 <= recipe_idx < len(recipes)):
        line_api.reply_message(
            reply_token,
            TextSendMessage(text="這個編號不在清單內～請輸入『做法 1/2/3』。", quick_reply=make_quickreply_menu()),
        )
        return

    recipe = recipes[recipe_idx]
    recipe_name = recipe.get("name", f"料理 {recipe_idx+1}")
    steps = recipe.get("steps") or []
    if not isinstance(steps, list) or not steps:
        line_api.reply_message(
            reply_token,
            TextSendMessage(text=f"《{recipe_name}》沒有步驟內容。你可以按『換食譜』換一批。", quick_reply=make_quickreply_menu()),
        )
        return

    cache = step_view_state.get(user_id)
    if cache and cache.get("recipe_idx") == recipe_idx and cache.get("steps") and cache.get("img_urls"):
        cache["page"] = 0
        step_items = [{"text": t, "image_url": u} for t, u in zip(cache["steps"], cache["img_urls"])]

        header = TextSendMessage(
            text=f"《{recipe_name}》步驟示意圖（第 1 頁）\n輸入『下一頁/上一頁』翻頁。",
            quick_reply=make_quickreply_menu(),
        )
        flex = steps_to_flex(step_items, page=0, page_size=5)
        line_api.reply_message(reply_token, [header, flex])
        return

    try:
        step_objs = gemini_steps_with_prompts(recipe_name, steps)
    except Exception as e:
        line_api.reply_message(
            reply_token,
            TextSendMessage(text=f"步驟圖 prompt 產生失敗：{type(e).__name__}: {e}", quick_reply=make_quickreply_menu()),
        )
        return

    step_objs = step_objs[:max(1, MAX_STEP_IMAGES)]

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
                p = f"Photorealistic instructional cooking image showing a step in action for {recipe_name}, hands, utensils, ingredients, kitchen, natural lighting, no text"
            url = generate_image_url(p)
        except Exception:
            url = None
        img_urls.append(url)

    if not step_texts:
        line_api.reply_message(
            reply_token,
            TextSendMessage(text=f"《{recipe_name}》步驟整理失敗，請按『換食譜』或再試一次『做法 {recipe_idx+1}』。", quick_reply=make_quickreply_menu()),
        )
        return

    step_view_state[user_id] = {
        "recipe_idx": recipe_idx,
        "recipe_name": recipe_name,
        "steps": step_texts,
        "img_urls": img_urls,
        "page": 0,
    }

    step_items = [{"text": t, "image_url": u} for t, u in zip(step_texts, img_urls)]
    header = TextSendMessage(
        text=(
            f"《{recipe_name}》步驟示意圖（第 1 頁）\n"
            f"（我先幫你把前 {len(step_items)} 步做成圖）\n"
            "輸入『下一頁/上一頁』翻頁。"
        ),
        quick_reply=make_quickreply_menu(),
    )
    flex = steps_to_flex(step_items, page=0, page_size=5)
    line_api.reply_message(reply_token, [header, flex])


def reply_step_page(user_id: str, reply_token: str, delta: int):
    cache = step_view_state.get(user_id)
    if not cache:
        line_api.reply_message(
            reply_token,
            TextSendMessage(text="你還沒有開啟任何步驟圖～先輸入『做法 1』。", quick_reply=make_quickreply_menu()),
        )
        return

    step_items = [{"text": t, "image_url": u} for t, u in zip(cache["steps"], cache["img_urls"])]
    total = len(step_items)
    page_size = 5
    max_page = max(0, (total - 1) // page_size)

    new_page = int(cache.get("page", 0)) + int(delta)
    new_page = max(0, min(new_page, max_page))
    cache["page"] = new_page

    flex = steps_to_flex(step_items, page=new_page, page_size=page_size)
    msg = TextSendMessage(
        text=f"《{cache.get('recipe_name','料理')}》步驟示意圖（第 {new_page+1} 頁）",
        quick_reply=make_quickreply_menu(),
    )
    line_api.reply_message(reply_token, [msg, flex])


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
    return "OK"


@handler.add(FollowEvent)
def handle_follow(event: FollowEvent):
    welcome = (
        "嗨～我是冰箱清理小幫手（Google 版）！\n\n"
        "✅ 你可以輸入任何食材/部位：\n"
        "例如：『我家有 霜降牛小排 雞腿排 洋蔥』\n\n"
        "✅ 或輸入『加入 雞腿排』存進冰箱\n"
        "✅ 輸入『推薦』生成 3 道菜\n"
        "✅ 不喜歡按『換食譜』\n"
        "✅ 看做法（含步驟圖）：輸入『做法 1』\n"
        "✅ 用完食材：輸入『- 雞腿排』或輸入『-』叫出移除選單\n"
        "✅ 叫出按鈕：輸入『+』或『開啟按鈕選單』"
    )
    line_api.reply_message(event.reply_token, TextSendMessage(text=welcome, quick_reply=make_quickreply_menu()))


@handler.add(MessageEvent, message=TextMessage)
def handle_text(event: MessageEvent):
    user_id = event.source.user_id
    text = (event.message.text or "").strip()

    # ---------- 開啟按鈕選單 ----------
    if text in {"+", "開啟按鈕選單", "按鈕選單", "選單", "menu", "MENU"}:
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="這是按鈕選單～你可以快速加入/推薦/換食譜/移除食材 👇",
                quick_reply=make_quickreply_menu(),
            ),
        )
        return

    # ---------- 用完食材：- 移除 ----------
    if text in {"-", "用完食材", "移除食材", "刪食材", "減食材"}:
        if not fridge_list(user_id):
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text="你的冰箱目前是空的～不用移除囉！", quick_reply=make_quickreply_menu()),
            )
            return
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text="你可以點下面按鈕移除已用完的食材，或直接輸入：- 霜降牛小排 洋蔥",
                quick_reply=make_remove_quickreply(user_id),
            ),
        )
        return

    m_minus = re.match(r"^-\s*(.+)$", text)  # 支援 -雞腿排 / - 雞腿排 洋蔥
    if m_minus:
        raw = m_minus.group(1).strip()
        parts = [p.strip() for p in re.split(SEPS, raw) if p.strip()]
        if not parts:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="請輸入：- 雞腿排 洋蔥（可一次移除多個）",
                    quick_reply=make_remove_quickreply(user_id),
                ),
            )
            return
        removed = remove_from_fridge(user_id, parts)
        if removed:
            step_view_state.pop(user_id, None)
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"已移除：{'、'.join(removed)}\n{fridge_text(user_id)}", quick_reply=make_quickreply_menu()),
            )
        else:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"我沒有在冰箱裡找到：{'、'.join(parts)}\n{fridge_text(user_id)}", quick_reply=make_quickreply_menu()),
            )
        return

    # ---------- 翻頁 ----------
    if text in {"下一頁", "下一", "next"}:
        reply_step_page(user_id, event.reply_token, delta=+1)
        return
    if text in {"上一頁", "上一", "prev"}:
        reply_step_page(user_id, event.reply_token, delta=-1)
        return

    # ---------- 查看冰箱 ----------
    if text in {"查看冰箱", "冰箱", "我的冰箱"}:
        line_api.reply_message(event.reply_token, TextSendMessage(text=fridge_text(user_id), quick_reply=make_quickreply_menu()))
        return

    # ---------- 清空冰箱（✅修掉你原本換行造成的 SyntaxError） ----------
    if text in {"清空冰箱", "清空", "重置冰箱", "清空全部"}:
        clear_fridge(user_id)
        recent_recipes.pop(user_id, None)
        last_used_ings.pop(user_id, None)
        last_titles.pop(user_id, None)
        step_view_state.pop(user_id, None)

        line_api.reply_message(
            event.reply_token,
            TextSendMessage(text="🗑 已清空冰箱！\n你的冰箱目前：（空的）", quick_reply=make_quickreply_menu()),
        )
        return

    # ---------- 加入食材：支援「加入 xxx」或「加 xxx」 ----------
    m_add = re.match(r"^(加入|加)\s*(.+)$", text)
    if m_add:
        raw = (m_add.group(2) or "").strip()
        parts = [p.strip() for p in re.split(SEPS, raw) if p.strip()]
        if not parts:
            line_api.reply_message(event.reply_token, TextSendMessage(text="請輸入：加入 雞腿排 洋蔥", quick_reply=make_quickreply_menu()))
            return
        added = add_to_fridge(user_id, parts)
        msg = f"✅ 已加入：{'、'.join(added)}\n{fridge_text(user_id)}" if added else f"這些已經在冰箱裡了～\n{fridge_text(user_id)}"
        line_api.reply_message(event.reply_token, TextSendMessage(text=msg, quick_reply=make_quickreply_menu()))
        return

    # ---------- 做法 N ----------
    m_steps = re.match(r"^(做法)\s*(\d+)\s*$", text)
    if m_steps:
        idx = int(m_steps.group(2)) - 1
        reply_steps_with_images(user_id, event.reply_token, recipe_idx=idx)
        return

    # ---------- 換食譜 / 推薦 ----------
    if text in {"換食譜", "換", "重新推薦", "再推薦"}:
        reply_recipes(user_id, event.reply_token, user_text=text, force_same_ingredients=True)
        return

    if text in {"推薦", "給我食譜", "食譜", "煮什麼", "今天煮什麼"}:
        reply_recipes(user_id, event.reply_token, user_text=text, force_same_ingredients=False)
        return

    # ---------- 其他任何輸入：一律當作「你想用這句話來推薦」 ----------
    reply_recipes(user_id, event.reply_token, user_text=text, force_same_ingredients=False)


if __name__ == "__main__":
    # Render 用 gunicorn 啟動時不會跑到這裡；本機測試才會用到
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
