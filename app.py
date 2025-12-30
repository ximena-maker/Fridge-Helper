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
# ✅ 你輸入一句話：我家有 雞肉 洋蔥 -> Gemini 抽食材 + 產生「至少 3 道」食譜
# ✅ 不喜歡：輸入/按「換食譜」 -> 同一批食材換一批，並盡量避開上一輪菜名
# ✅ 每道菜：有一張示意圖（Imagen 生成）
# ✅ 做法 N：顯示「每一步的示意圖 + 步驟文字」（Imagen 生成），支援「上一頁/下一頁」
#
# 必要環境變數（Render / 本機）：
# - CHANNEL_SECRET
# - CHANNEL_ACCESS_TOKEN
# - GEMINI_API_KEY   (⚠️ 不能用被標記 leaked 的 key)
# - PUBLIC_BASE_URL  例：https://fridge-helper.onrender.com   （步驟圖/料理圖要能被 LINE 以 https 讀取）
#
# 可選環境變數：
# - GEMINI_TEXT_MODEL   預設 gemini-2.5-flash
# - IMAGE_MODEL         預設 imagen-4.0-generate-001
# - MAX_KEEP_IMAGES     預設 120（static/generated 保留張數）
# - MAX_STEP_IMAGES     預設 10（每次做法最多先生成幾步的圖）
# =========================================================


# ---------------------
# LINE keys
# ---------------------
def load_line_keys(filepath: str = "keys.txt") -> Dict[str, str]:
    channel_secret = os.getenv("CHANNEL_SECRET")
    channel_access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
    if channel_secret and channel_access_token:
        return {
            "CHANNEL_SECRET": channel_secret,
            "CHANNEL_ACCESS_TOKEN": channel_access_token,
        }

    p = Path(__file__).with_name(filepath)
    if not p.exists():
        raise RuntimeError("缺少 LINE CHANNEL_SECRET / CHANNEL_ACCESS_TOKEN（請設定環境變數或提供 keys.txt）")

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

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")  # 必須 https 才能讓 LINE 顯示圖片
MAX_KEEP_IMAGES = int(os.getenv("MAX_KEEP_IMAGES", "120"))
MAX_STEP_IMAGES = int(os.getenv("MAX_STEP_IMAGES", "10"))  # 做法圖一次最多先生成幾步


# ---------------------
# Flask static for generated images
# ---------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
GEN_DIR = STATIC_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_old_images():
    """避免磁碟越來越大：保留最近 MAX_KEEP_IMAGES 張"""
    try:
        files = sorted(GEN_DIR.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[MAX_KEEP_IMAGES:]:
            try:
                p.unlink()
            except:
                pass
    except:
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
    """
    google-genai 的 generate_images 回傳結構可能是：
      generated_images[i].image.image_bytes (bytes or base64 str)
    這裡做防呆處理
    """
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
    """存到 static/generated，回傳 https 可公開 URL（需要 PUBLIC_BASE_URL）"""
    cleanup_old_images()

    fname = f"{uuid.uuid4().hex}.png"
    fpath = GEN_DIR / fname
    with fpath.open("wb") as f:
        f.write(img_bytes)

    if not PUBLIC_BASE_URL.startswith("https://"):
        # 沒設定 PUBLIC_BASE_URL 或不是 https -> LINE 會顯示不了圖
        return None

    return f"{PUBLIC_BASE_URL}/static/generated/{fname}"


def generate_image_url(prompt: str) -> Optional[str]:
    """Imagen 生成圖片 → 存檔 → 回傳 URL"""
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
user_fridge_list = defaultdict(list)     # user_id -> ["雞肉","洋蔥",...]
user_fridge_norm = defaultdict(set)      # user_id -> {"雞肉","洋蔥"...(lower/norm)}
recent_recipes = {}                      # user_id -> 最近一輪 3 道 recipes (dict list)
last_used_ings = {}                      # user_id -> 上一次生成時使用的食材（list）
last_titles = defaultdict(list)          # user_id -> 上一次生成的菜名（list）

# 步驟圖快取：user_id -> {recipe_idx:int, recipe_name:str, steps:[str], img_urls:[str], page:int}
step_view_state = {}


def add_to_fridge(user_id: str, items: List[str]):
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        nx = _norm_token(x)
        if nx and nx not in user_fridge_norm[user_id]:
            user_fridge_norm[user_id].add(nx)
            user_fridge_list[user_id].append(x)


def clear_fridge(user_id: str):
    user_fridge_list[user_id] = []
    user_fridge_norm[user_id] = set()


def fridge_text(user_id: str) -> str:
    items = user_fridge_list[user_id]
    return "你的冰箱目前：" + ("、".join(items) if items else "（空的）")


# =========================================================
# Quick Reply
# =========================================================
COMMON_INGS = ["雞肉", "牛肉", "豬肉", "雞蛋", "洋蔥", "大蒜", "蔥", "花椰菜", "馬鈴薯", "番茄"]

def make_quickreply_menu() -> QuickReply:
    items = []

    # 7 個常見食材 + 6 個功能 = 13（保守不超）
    for ing in COMMON_INGS[:7]:
        items.append(QuickReplyButton(action=MessageAction(label=f"+{ing}", text=f"加入 {ing}")))

    items.append(QuickReplyButton(action=MessageAction(label="🍳 推薦", text="推薦")))
    items.append(QuickReplyButton(action=MessageAction(label="🔁 換食譜", text="換食譜")))
    items.append(QuickReplyButton(action=MessageAction(label="⬅ 上一頁", text="上一頁")))
    items.append(QuickReplyButton(action=MessageAction(label="下一頁 ➡", text="下一頁")))
    items.append(QuickReplyButton(action=MessageAction(label="📦 查看冰箱", text="查看冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="🗑 清空", text="清空冰箱")))

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
      "ingredients": [...],
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

【目前冰箱已記錄食材】
{ "、".join(fridge_items) if fridge_items else "（空）" }

【要求 JSON 格式】
{{
  "ingredients": ["抽取/推斷到的食材（中文，去掉數量與單位，去重）"],
  "recipes": [
    {{
      "name": "菜名（中文）",
      "summary": "一句話介紹（中文）",
      "ingredients": ["關鍵食材（中文）"],
      "steps": ["步驟1（中文）","步驟2（中文）", "...至少 5 步"],
      "image_prompt": "English prompt for a photorealistic food photo of this dish, plated nicely, natural lighting, shallow depth of field, no text"
    }}
  ]
}}

【規則】
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

    # 防呆：保證至少 n_recipes
    recipes = data.get("recipes") or []
    if not isinstance(recipes, list):
        recipes = []

    # 如果少於 n_recipes，補問一次（最多補 2 次）
    tries = 0
    while len(recipes) < n_recipes and tries < 2:
        tries += 1
        prompt2 = f"""
只輸出 JSON（不要任何其他文字）。
用這些食材生成「剛好 {n_recipes} 道」recipes（同上格式），且避開菜名：{avoid_titles_in + [r.get("name","") for r in recipes]}
食材：{sorted(set(fridge_items + (data.get("ingredients") or [])))}
"""
        resp2 = client.models.generate_content(
            model=TEXT_MODEL,
            contents=prompt2,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7),
        )
        d2 = _safe_json_loads(getattr(resp2, "text", "") or "")
        r2 = (d2.get("recipes") or []) if isinstance(d2, dict) else []
        if isinstance(r2, list):
            # 合併去重（以 name）
            seen = {(_norm_token(r.get("name", ""))) for r in recipes}
            for r in r2:
                nm = _norm_token((r or {}).get("name", ""))
                if nm and nm not in seen:
                    recipes.append(r)
                    seen.add(nm)

    recipes = recipes[:n_recipes]
    data["recipes"] = recipes

    # ingredients 清理去重
    ings = data.get("ingredients") or []
    if not isinstance(ings, list):
        ings = []
    ings2 = []
    seen2 = set()
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
    """
    回傳：
    [
      {"text":"中文步驟(更清楚)", "image_prompt":"English prompt ... (instructional, no text)"}
    ]
    """
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

    out = []
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
        ing_text = "、".join([str(x) for x in ings[:10] if str(x).strip()]) + ("…" if len(ings) > 10 else "")
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
# 核心：推薦 / 換食譜（每道菜 1 圖）
# =========================================================
def reply_recipes(user_id: str, reply_token: str, user_text: str, force_same_ingredients: bool = False):
    """
    force_same_ingredients=True：換食譜（用 last_used_ings，並避開 last_titles）
    """
    try:
        if force_same_ingredients:
            base_ings = last_used_ings.get(user_id) or user_fridge_list[user_id]
            if not base_ings:
                line_api.reply_message(
                    reply_token,
                    TextSendMessage(
                        text="你還沒有可用食材～先輸入：『我家有 雞肉 洋蔥』或用『加入 雞肉』加入吧！",
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
                fridge_items=user_fridge_list[user_id],
                avoid_titles_in=[],
                n_recipes=3,
            )

        extracted = data.get("ingredients") or []
        if isinstance(extracted, list) and extracted:
            add_to_fridge(user_id, [str(x) for x in extracted])

        use_ings = list(user_fridge_list[user_id])
        last_used_ings[user_id] = use_ings

        recipes = data.get("recipes") or []
        if not isinstance(recipes, list) or len(recipes) < 3:
            raise RuntimeError("Gemini 沒產出足夠的食譜（少於 3 道）")

        # 料理圖片：每道 1 張
        bubbles = []
        titles = []
        final_recipes = []

        for i, r in enumerate(recipes[:3], start=1):
            if not isinstance(r, dict):
                continue
            name = r.get("name", f"料理 {i}")
            titles.append(name)

            # Imagen 成品圖 prompt（英文）
            img_prompt = (r.get("image_prompt") or "").strip()
            if not img_prompt:
                img_prompt = f"A high-quality photorealistic food photo of {name}, plated nicely, natural lighting, shallow depth of field, no text"

            dish_img_url = None
            try:
                dish_img_url = generate_image_url(img_prompt)
            except:
                dish_img_url = None

            bubbles.append(recipe_to_bubble(i, r, dish_img_url))
            final_recipes.append(r)

        if len(final_recipes) < 3:
            raise RuntimeError("食譜資料格式異常（不足 3 道有效食譜）")

        recent_recipes[user_id] = final_recipes
        last_titles[user_id] = titles

        # 換菜後，步驟頁狀態清掉（避免翻頁顯示上一道）
        step_view_state.pop(user_id, None)

        text_msg = TextSendMessage(
            text=(
                f"✅ 使用食材：{'、'.join(use_ings) if use_ings else '（未偵測到）'}\n"
                f"{fridge_text(user_id)}\n\n"
                "我給你 3 個選項～\n"
                "📌 看做法（含步驟圖）：輸入『做法 1/2/3』\n"
                "🔁 不喜歡：輸入/按『換食譜』再換一批"
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
                    "1) 我家有 雞肉 洋蔥\n"
                    "2) 加入 雞肉 洋蔥\n"
                    "3) 推薦\n\n"
                    "（若看到 API key leaked/403：請換一把新的 GEMINI_API_KEY，並更新 Render 環境變數）"
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

    # 如果同一道菜已做過步驟圖：直接顯示第 1 頁
    cache = step_view_state.get(user_id)
    if cache and cache.get("recipe_idx") == recipe_idx and cache.get("steps") and cache.get("img_urls"):
        page = 0
        cache["page"] = page
        step_items = [{"text": t, "image_url": u} for t, u in zip(cache["steps"], cache["img_urls"])]

        header = TextSendMessage(
            text=f"《{recipe_name}》步驟示意圖（第 1 頁）\n輸入『下一頁/上一頁』翻頁。",
            quick_reply=make_quickreply_menu(),
        )
        flex = steps_to_flex(step_items, page=page, page_size=5)
        line_api.reply_message(reply_token, [header, flex])
        return

    # 先讓 Gemini 幫每一步生成英文 prompt（教學圖）
    try:
        step_objs = gemini_steps_with_prompts(recipe_name, steps)
    except Exception as e:
        line_api.reply_message(
            reply_token,
            TextSendMessage(text=f"步驟圖 prompt 產生失敗：{type(e).__name__}: {e}", quick_reply=make_quickreply_menu()),
        )
        return

    # 產步驟圖（限制最多 MAX_STEP_IMAGES，避免太慢/太燒額度）
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
            # 如果 prompt 空，就給一個保底
            if not p:
                p = f"Photorealistic instructional cooking image showing step in action for {recipe_name}, hands, utensils, ingredients, kitchen, natural lighting, no text"
            url = generate_image_url(p)
        except:
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

    new_page = cache.get("page", 0) + delta
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
        "✅ 直接輸入一句話：\n"
        "例如：『我家有 雞肉 洋蔥 花椰菜』\n\n"
        "✅ 或輸入『加入 雞肉』把食材存進冰箱\n"
        "✅ 輸入『推薦』用冰箱食材生成 3 道菜\n"
        "✅ 不喜歡按『換食譜』再換一批\n"
        "✅ 看做法（含步驟圖）：輸入『做法 1』"
    )
    line_api.reply_message(
        event.reply_token,
        TextSendMessage(text=welcome, quick_reply=make_quickreply_menu()),
    )


@handler.add(MessageEvent, message=TextMessage)
def handle_text(event: MessageEvent):
    user_id = event.source.user_id
    text = (event.message.text or "").strip()

    # ---------- 翻頁 ----------
    if text in {"下一頁", "下一", "next"}:
        reply_step_page(user_id, event.reply_token, delta=+1)
        return
    if text in {"上一頁", "上一", "prev"}:
        reply_step_page(user_id, event.reply_token, delta=-1)
        return

    # ---------- 查看/清空 ----------
    if text in {"查看冰箱", "冰箱", "我的冰箱"}:
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(text=fridge_text(user_id), quick_reply=make_quickreply_menu()),
        )
        return
    if text in {"清空冰箱", "清空", "重置冰箱"}:
        clear_fridge(user_id)
        step_view_state.pop(user_id, None)
        line_api.reply_message(
            event.reply_token,
            TextSendMessage(text="已清空～\n" + fridge_text(user_id), quick_reply=make_quickreply_menu()),
        )
        return

    # ---------- 加入食材 ----------
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
            TextSendMessage(text=f"已加入：{'、'.join(parts)}\n{fridge_text(user_id)}", quick_reply=make_quickreply_menu()),
        )
        return

    # ---------- 推薦 ----------
    if text in {"推薦", "推薦料理", "煮什麼", "做什麼", "想煮"}:
        if not user_fridge_list[user_id]:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="你的冰箱還是空的～先輸入：『我家有 雞肉 洋蔥』或『加入 雞肉』",
                    quick_reply=make_quickreply_menu(),
                ),
            )
            return
        reply_recipes(user_id, event.reply_token, user_text="請用我的冰箱食材生成食譜", force_same_ingredients=False)
        return

    # ---------- 換食譜 ----------
    if text in {"換食譜", "換", "換一批", "不喜歡", "再給我別的"}:
        if not (last_used_ings.get(user_id) or user_fridge_list[user_id]):
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text="你還沒生成過食譜～先輸入食材或按『推薦』。", quick_reply=make_quickreply_menu()),
            )
            return
        reply_recipes(user_id, event.reply_token, user_text="換食譜", force_same_ingredients=True)
        return

    # ---------- 做法 N（含步驟圖） ----------
    if text.startswith("做法"):
        m = re.search(r"\d+", text)
        if not m:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text="請輸入：做法 1 / 做法 2 / 做法 3", quick_reply=make_quickreply_menu()),
            )
            return
        idx = int(m.group()) - 1
        reply_steps_with_images(user_id, event.reply_token, recipe_idx=idx)
        return

    # ---------- 一般句子：交給 Gemini 抓食材 + 生成 3 道 ----------
    reply_recipes(user_id, event.reply_token, user_text=text, force_same_ingredients=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
