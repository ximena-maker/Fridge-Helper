import os
import re
import json
import uuid
import base64
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

from flask import Flask, request, abort, send_from_directory

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

# Google GenAI SDK
from google import genai
from google.genai import types


# =========================================================
# 冰箱清理小幫手（LINE Bot）- 全部改用 Google Gemini/Imagen
# 功能：
# 1) 你輸入一句話：我家有 雞肉 洋蔥 -> Gemini 抽食材 + 生成 3 道食譜
# 2) 可用按鈕加入食材：加入 雞肉 / 加入 洋蔥 ...
# 3) 「換食譜」：同一批食材再生另一組 3 道（避開上一輪菜名）
# 4) 每道食譜都有示意圖：用 Imagen 產生，存到 /static/generated，Flex 顯示 URL
# =========================================================


# ---------------------
# LINE channel keys
# ---------------------
def load_line_keys(filepath: str = "keys.txt"):
    """
    讀取 LINE 金鑰：
    1) 優先讀環境變數 CHANNEL_SECRET / CHANNEL_ACCESS_TOKEN
    2) 其次讀 keys.txt
    """
    channel_secret = os.getenv("CHANNEL_SECRET")
    channel_access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
    if channel_secret and channel_access_token:
        return {"CHANNEL_SECRET": channel_secret, "CHANNEL_ACCESS_TOKEN": channel_access_token}

    p = Path(__file__).with_name(filepath)
    if not p.exists():
        raise RuntimeError("錯誤：缺少 LINE CHANNEL_SECRET / CHANNEL_ACCESS_TOKEN（請設定環境變數或提供 keys.txt）")

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
# Google GenAI client
# ---------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("缺少 GEMINI_API_KEY（請在 Render / 本機環境變數設定）")

client = genai.Client(api_key=GEMINI_API_KEY)

TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "imagen-4.0-generate-001").strip()

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
# 例：https://fridge-helper.onrender.com
if not PUBLIC_BASE_URL.startswith("https://"):
    # LINE 需要 HTTPS 圖片 URL，請務必設定成 https 網址
    #（不直接 raise，避免你本機測試時中斷）
    pass


# ---------------------
# Flask static for generated images
# ---------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
GEN_DIR = STATIC_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

# 清理舊圖（避免磁碟越來越大）——保留最近 N 張
MAX_KEEP_IMAGES = int(os.getenv("MAX_KEEP_IMAGES", "120"))


def cleanup_old_images():
    try:
        files = sorted(GEN_DIR.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[MAX_KEEP_IMAGES:]:
            try:
                p.unlink()
            except:
                pass
    except:
        pass


# ---------------------
# 使用者冰箱（記憶：存在記憶體，重啟就清空）
# ---------------------
user_fridge = defaultdict(list)  # user_id -> list[str]（保留原字串，顯示比較自然）
user_fridge_norm = defaultdict(set)  # user_id -> set[str]（去重用）
recent_recipes = {}  # user_id -> list[dict]（上一輪 3 道）
last_used_ings = {}  # user_id -> list[str]
last_titles = defaultdict(list)  # user_id -> list[str]（上一輪菜名）


def norm_token(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip().lower()


def fridge_list_text(user_id: str) -> str:
    items = user_fridge[user_id]
    return "你的冰箱目前：" + ("、".join(items) if items else "（空的）")


def add_to_fridge(user_id: str, items):
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        nx = norm_token(x)
        if nx and nx not in user_fridge_norm[user_id]:
            user_fridge_norm[user_id].add(nx)
            user_fridge[user_id].append(x)


def clear_fridge(user_id: str):
    user_fridge[user_id] = []
    user_fridge_norm[user_id] = set()


# ---------------------
# Quick Reply buttons
# ---------------------
COMMON_INGS = ["雞肉", "牛肉", "豬肉", "雞蛋", "洋蔥", "大蒜", "蔥", "花椰菜", "馬鈴薯", "番茄"]

def make_quickreply_menu():
    items = []

    # 常用食材（前 8 個）
    for ing in COMMON_INGS[:8]:
        items.append(QuickReplyButton(action=MessageAction(label=f"+{ing}", text=f"加入 {ing}")))

    # 系統功能
    items.append(QuickReplyButton(action=MessageAction(label="🍳 推薦", text="推薦")))
    items.append(QuickReplyButton(action=MessageAction(label="🔁 換食譜", text="換食譜")))
    items.append(QuickReplyButton(action=MessageAction(label="📦 查看冰箱", text="查看冰箱")))
    items.append(QuickReplyButton(action=MessageAction(label="🗑 清空", text="清空冰箱")))

    return QuickReply(items=items)


# ---------------------
# Google (Gemini) - 抽食材 + 生成食譜（至少 3 道）
# ---------------------
def _safe_json_loads(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except:
        # 嘗試從回覆中抓第一段 JSON
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                return None
        return None


def gemini_extract_and_generate(user_text: str, fridge_items: list[str], avoid_titles: list[str], n_recipes: int = 3):
    """
    回傳 dict:
    {
      "ingredients": [ ... ]  # 本次理解到的食材（會合併冰箱）
      "recipes": [
        {
          "name": "...",
          "summary": "...",
          "ingredients": ["...","..."],
          "steps": ["...","..."],
          "image_prompt": "English prompt..."
        }, ...
      ]
    }
    """
    # Imagen 官方文件：prompt 英文較穩（Imagen 也標示英語）:contentReference[oaicite:3]{index=3}
    # 所以 image_prompt 我要求 Gemini 幫我們產英文 prompt
    system = (
        "你是料理助理。你的任務：\n"
        "1) 從使用者輸入中抽取食材（可以理解中文口語）。\n"
        "2) 根據可用食材，生成食譜選項。\n"
        "3) 回覆必須是「純 JSON」(application/json)，不要加任何額外文字。\n"
        "4) 生成的食譜要務實、家常、可操作。\n"
    )

    avoid_txt = "、".join(avoid_titles[:12]) if avoid_titles else ""
    fridge_txt = "、".join(fridge_items) if fridge_items else ""

    prompt = f"""
使用者輸入：{user_text}

目前冰箱已記錄食材：{fridge_txt}

請輸出 JSON，格式如下（欄位名必須一致）：
{{
  "ingredients": ["從使用者輸入+冰箱推斷的可用食材（去重）"],
  "recipes": [
    {{
      "name": "菜名（中文）",
      "summary": "一句話介紹（中文）",
      "ingredients": ["需要的食材（中文，盡量只列關鍵食材）"],
      "steps": ["步驟1（中文）","步驟2（中文）"],
      "image_prompt": "English prompt for a photorealistic food photo of this dish, plated nicely, natural lighting, shallow depth of field"
    }}
  ]
}}

規則：
- recipes 請輸出「剛好 {n_recipes} 道」。
- 不要輸出多餘欄位。
- 步驟至少 5 步。
- 菜名不要太雷同。
{"- 避免使用這些菜名或過於相近的菜名：" + avoid_txt if avoid_txt else ""}
"""

    resp = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[system, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    data = _safe_json_loads(getattr(resp, "text", None))
    if not isinstance(data, dict):
        raise RuntimeError("Gemini 回覆不是有效 JSON")

    # 基本修正
    data.setdefault("ingredients", [])
    data.setdefault("recipes", [])
    return data


# ---------------------
# Google (Imagen) - 生成圖片並存到 /static/generated
# ---------------------
def _get_image_bytes_from_generated_image(generated_image):
    """
    SDK 有時是 bytes、有時是 base64 字串，這裡做防呆。
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
        # 可能是 base64
        try:
            return base64.b64decode(b)
        except:
            return None

    return None


def generate_image_url_for_recipe(recipe_name: str, image_prompt: str):
    """
    用 Imagen 產圖，存檔後回傳可公開 https url（PUBLIC_BASE_URL/static/generated/xxx.png）
    """
    cleanup_old_images()

    prompt = (image_prompt or "").strip()
    if not prompt:
        prompt = f"A high-quality photorealistic food photo of {recipe_name}, plated nicely, natural lighting, shallow depth of field"

    resp = client.models.generate_images(
        model=IMAGE_MODEL,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            # 你也可以加 aspect_ratio，但不同版本命名可能不同；先保守不加
        ),
    )

    gen_list = getattr(resp, "generated_images", None) or []
    if not gen_list:
        return None

    img_bytes = _get_image_bytes_from_generated_image(gen_list[0])
    if not img_bytes:
        return None

    fname = f"{uuid.uuid4().hex}.png"
    fpath = GEN_DIR / fname
    with fpath.open("wb") as f:
        f.write(img_bytes)

    if not PUBLIC_BASE_URL.startswith("https://"):
        # 本機沒設 PUBLIC_BASE_URL 時，就先回 None（避免 LINE 收到不合法 URL）
        return None

    return f"{PUBLIC_BASE_URL}/static/generated/{fname}"


# ---------------------
# Flex Message bubble（加 hero 圖片）
# ---------------------
def recipe_to_bubble(rank: int, recipe: dict, image_url: str | None):
    title = recipe.get("name", f"料理 {rank}")
    summary = recipe.get("summary", "")
    ings = recipe.get("ingredients", [])
    if isinstance(ings, list):
        ing_text = "、".join(ings[:10]) + ("…" if len(ings) > 10 else "")
    else:
        ing_text = "—"

    bubble = {
        "type": "bubble",
        "size": "mega",
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {"type": "text", "text": f"{rank}. {title}", "wrap": True, "weight": "bold", "size": "lg"},
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

    # hero image（若有）
    if image_url:
        bubble["hero"] = {
            "type": "image",
            "url": image_url,
            "size": "full",
            "aspectRatio": "16:9",
            "aspectMode": "cover",
        }

    # body 補資訊
    if summary:
        bubble["body"]["contents"].append({"type": "text", "text": summary, "wrap": True, "size": "sm"})
    bubble["body"]["contents"].append({"type": "text", "text": f"🧾 食材：{ing_text}", "wrap": True, "size": "sm"})

    return bubble


# ---------------------
# 組裝推薦訊息（至少 3 道 + 換食譜）
# ---------------------
def build_and_reply_recipes(user_id: str, reply_token: str, user_text: str, force_same_ingredients: bool = False):
    """
    force_same_ingredients=True：用 last_used_ings 直接換一批，不重新抽取
    """
    try:
        if force_same_ingredients and user_id in last_used_ings and last_used_ings[user_id]:
            # 換食譜：用同一批食材
            base_ings = last_used_ings[user_id]
            data = gemini_extract_and_generate(
                user_text=f"請用同一批食材換一組新食譜：{ '、'.join(base_ings) }",
                fridge_items=base_ings,
                avoid_titles=last_titles[user_id],
                n_recipes=3,
            )
        else:
            data = gemini_extract_and_generate(
                user_text=user_text,
                fridge_items=user_fridge[user_id],
                avoid_titles=[],
                n_recipes=3,
            )

        ing_list = data.get("ingredients", [])
        if isinstance(ing_list, list) and ing_list:
            add_to_fridge(user_id, ing_list)

        # 這輪使用的食材（用冰箱全量）
        use_ings = list(user_fridge[user_id])
        last_used_ings[user_id] = use_ings

        recipes = data.get("recipes", [])
        if not isinstance(recipes, list) or len(recipes) < 3:
            raise RuntimeError("Gemini 沒產出足夠的食譜（少於 3 道）")

        # 生成圖片（逐道）
        bubbles = []
        final_recipes = []
        titles = []
        for i, r in enumerate(recipes[:3], start=1):
            name = r.get("name", f"料理 {i}")
            titles.append(name)

            img_url = None
            try:
                img_url = generate_image_url_for_recipe(name, r.get("image_prompt", ""))
            except:
                img_url = None

            bubbles.append(recipe_to_bubble(i, r, img_url))
            final_recipes.append(r)

        recent_recipes[user_id] = final_recipes
        last_titles[user_id] = titles

        text_msg = TextSendMessage(
            text=(
                f"✅ 使用食材：{'、'.join(use_ings) if use_ings else '（未偵測到）'}\n"
                f"{fridge_list_text(user_id)}\n\n"
                "我先給你 3 個選項～\n"
                "📌 看做法：輸入『做法 1』\n"
                "🔁 不喜歡：按『換食譜』再換一批"
            ),
            quick_reply=make_quickreply_menu(),
        )

        flex_msg = FlexSendMessage(
            alt_text="推薦料理（含示意圖）",
            contents={"type": "carousel", "contents": bubbles},
        )

        line_api.reply_message(reply_token, [text_msg, flex_msg])

    except Exception as e:
        # 常見：403 key leaked / 401 無權限 / JSON 格式不對
        line_api.reply_message(
            reply_token,
            TextSendMessage(
                text=(
                    f"Google 解析或產圖時出錯了：{type(e).__name__}: {e}\n\n"
                    "你可以先試：\n"
                    "1) 我家有 雞肉 洋蔥\n"
                    "2) 加入 雞肉 洋蔥\n"
                    "3) 推薦\n\n"
                    "（如果是 API key 問題：請換一把新的 GEMINI_API_KEY，並在 Render 環境變數更新）"
                ),
                quick_reply=make_quickreply_menu(),
            ),
        )


# ---------------------
# Flask
# ---------------------
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
        if m and user_id in recent_recipes:
            idx = int(m.group()) - 1
            if 0 <= idx < len(recent_recipes[user_id]):
                r = recent_recipes[user_id][idx]
                steps = r.get("steps", [])
                if isinstance(steps, list):
                    steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
                else:
                    steps_text = str(steps) if steps else "（沒有步驟內容）"

                line_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=f"《{r.get('name','(未命名)')}》\n\n{steps_text}", quick_reply=make_quickreply_menu()),
                )
                return

        line_api.reply_message(
            event.reply_token,
            TextSendMessage(text="找不到對應的編號耶～先讓我推薦一次，再輸入『做法 1』喔。", quick_reply=make_quickreply_menu()),
        )
        return

    # ---------- 冰箱管理 ----------
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

    # ---------- 手動加入 ----------
    m_add = re.match(r"^(?:加入|加|新增)[:：\s]+(.+)$", text)
    if m_add:
        raw = m_add.group(1)
        parts = re.split(r"[\s、,，;；/]+", raw)
        parts = [p.strip() for p in parts if p.strip()]
        add_to_fridge(user_id, parts)

        line_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"已加入：{'、'.join(parts)}\n{fridge_list_text(user_id)}", quick_reply=make_quickreply_menu()),
        )
        return

    # ---------- 推薦（用冰箱） ----------
    if text in {"推薦", "推薦料理", "煮什麼", "做什麼", "想煮"}:
        if not user_fridge[user_id]:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="你的冰箱還是空的～先輸入：『我家有 雞肉 洋蔥』或『加入 雞肉』",
                    quick_reply=make_quickreply_menu(),
                ),
            )
            return

        build_and_reply_recipes(user_id, event.reply_token, user_text="請用我的冰箱食材生成食譜", force_same_ingredients=False)
        return

    # ---------- 換食譜 ----------
    if text in {"換食譜", "換", "換一批", "不喜歡", "再給我別的"}:
        if user_id not in last_used_ings or not last_used_ings[user_id]:
            line_api.reply_message(
                event.reply_token,
                TextSendMessage(text="你還沒生成過食譜～先輸入食材或按『推薦』。", quick_reply=make_quickreply_menu()),
            )
            return
        build_and_reply_recipes(user_id, event.reply_token, user_text="換食譜", force_same_ingredients=True)
        return

    # ---------- 一般句子：交給 Gemini 抓食材 + 生成 3 道 ----------
    build_and_reply_recipes(user_id, event.reply_token, user_text=text, force_same_ingredients=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
