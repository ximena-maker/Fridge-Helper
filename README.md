# 🍳 Fridge Helper - 清冰箱小助手

> 一個基於 LINE Bot 的智慧料理推薦系統，使用 Google Gemini AI 和 Imagen 圖像生成技術，幫助使用者有效運用冰箱食材，減少食物浪費。

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![LINE Bot SDK](https://img.shields.io/badge/LINE%20Bot%20SDK-3.x-00B900.svg)](https://github.com/line/line-bot-sdk-python)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Imagen](https://img.shields.io/badge/Imagen-4.0-4285F4.svg)](https://cloud.google.com/imagen)

---

## 📋 目錄

- [專案簡介](#專案簡介)
- [核心功能](#核心功能)
- [系統架構](#系統架構)
- [技術棧](#技術棧)
- [環境設置](#環境設置)
- [安裝與部署](#安裝與部署)
- [使用說明](#使用說明)
- [程式模組說明](#程式模組說明)
- [資料結構](#資料結構)
- [專案特色](#專案特色)
- [限制與未來改進](#限制與未來改進)
- [授權與致謝](#授權與致謝)

---

## 📖 專案簡介

**清冰箱小助手**是一個創新的 LINE Bot 應用程式，旨在解決家庭食材浪費問題。透過先進的 AI 技術，系統能夠：

- 🤖 **智慧識別食材**：自然語言處理，從使用者輸入中自動提取食材
- 🍽️ **AI 食譜生成**：根據現有食材生成多道家常料理
- 🖼️ **視覺化呈現**：為每道料理和步驟生成精美的 AI 圖片
- 🔄 **智慧推薦**：不喜歡可一鍵換食譜，避免重複推薦

---

## ✨ 核心功能

### 1. 🥘 智慧食材管理

- **自然語言識別**：輸入「我家有 雞肉 洋蔥 花椒菜」，AI 自動提取食材
- **虛擬冰箱系統**：手動加入「加入 雞肉 洋蔥」到冰箱
- **便捷管理**：查看冰箱、清空冰箱功能

### 2. 🍲 AI 食譜生成

- **多道料理推薦**：每次生成至少 3 道不同的家常料理
- **完整食譜內容**：菜名、簡介、食材清單、詳細步驟（至少 5 步）
- **首道示意圖**：使用 Imagen 4.0 為每道料理生成精美成品圖
- **智慧換譜**：輸入「換食譜」基於相同食材生成全新料理組合

### 3. 📸 視覺化步驟指導

- **步驟圖生成**：每個烹飪步驟配有 AI 生成的示意圖
- **分頁瀏覽**：每頁 5 個步驟，支援「上一頁/下一頁」瀏覽
- **圖文並茂**：步驟文字 + 示意圖，看圖就懂如何操作

### 4. 🟢 Quick Reply 快捷選單

- **快速加入食材**：7 個常見食材按鈕（雞肉、牛肉、豬肉等）
- **功能快捷鍵**：推薦、換食譜、查看冰箱、清空冰箱
- **翻頁按鈕**：上一頁、下一頁

---

## 🏛️ 系統架構

### 整體架構圖

```
使用者輸入 (自然語言)
        ↓
  LINE Bot Webhook
        ↓
   Flask 後端處理
        ↓
   ├── Gemini API (食材提取 + 食譜生成)
   ├── Imagen API (料理圖片 + 步驟圖生成)
   └── 虛擬冰箱狀態管理 (記憶體)
        ↓
  Flex Message 卡片呈現
        ↓
   LINE 回傳給使用者
```

### 核心流程

1. **食材輸入** → Gemini 識別食材 → 儲存到虛擬冰箱
2. **推薦料理** → Gemini 生成 3 道食譜 → Imagen 生成成品圖
3. **查看做法** → Gemini 生成步驟 prompt → Imagen 生成步驟圖
4. **瀏覽步驟** → 分頁顯示，支援翻頁操作

---

## 🛠️ 技術棧

### 主要技術

| 技術 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.x | 主要開發語言 |
| **Flask** | 3.0+ | Web 框架，處理 LINE Webhook |
| **LINE Bot SDK** | 3.x | LINE Bot API 整合 |
| **Google Gemini** | 2.5 Flash | 文字生成、食譜創作、食材識別 |
| **Google Imagen** | 4.0 | 高品質圖像生成 |

### 主要依賴套件

```txt
Flask==3.0.3
line-bot-sdk==3.14.0
google-genai==0.5.0
```

---

## ⚙️ 環境設置

### 必要環境變數

```bash
# LINE Bot 憑證
CHANNEL_SECRET=your_line_channel_secret
CHANNEL_ACCESS_TOKEN=your_line_channel_access_token

# Google AI API 金鑰
GEMINI_API_KEY=your_gemini_api_key

# 公開網址（必須為 HTTPS，讓 LINE 能讀取生成的圖片）
PUBLIC_BASE_URL=https://your-app.onrender.com
```

### 可選環境變數

```bash
# AI 模型配置
GEMINI_TEXT_MODEL=gemini-2.5-flash  # 預設值
IMAGE_MODEL=imagen-4.0-generate-001  # 預設值

# 圖片管理
MAX_KEEP_IMAGES=120  # 保留圖片數量上限
MAX_STEP_IMAGES=10   # 每次做法最多生成幾步的圖

# 伺服器埠號
PORT=5000  # 預設值
```

### 環境變數設定方式

**方式一**：創建 `.env` 檔案（本機開發）

```bash
# .env
CHANNEL_SECRET=your_secret
CHANNEL_ACCESS_TOKEN=your_token
GEMINI_API_KEY=your_api_key
PUBLIC_BASE_URL=https://your-domain.com
```

**方式二**：設定系統環境變數（Render / 雲端部署）

在 Render Dashboard 中直接設定環境變數。

**方式三**：創建 `keys.txt` 檔案（仅供開發測試）

```txt
CHANNEL_SECRET=your_secret
CHANNEL_ACCESS_TOKEN=your_token
```

---

## 🚀 安裝與部署

### 本機安裝

#### 1. 克隆專案

```bash
git clone https://github.com/ximena-maker/Fridge-Helper.git
cd Fridge-Helper
```

#### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

#### 3. 設定環境變數

參考上方「環境設置」章節，選擇一種方式設定。

#### 4. 執行程式

```bash
python app.py
```

伺服器會在 `http://0.0.0.0:5000` 啟動。

#### 5. 設定 LINE Webhook

1. 進入 [LINE Developers Console](https://developers.line.biz/)
2. 設定 Webhook URL: `https://your-domain.com/callback`
3. 啟用 Webhook

---

### Render 部署（推薦）

#### 步驟

1. **創建 Render 帳號**：前往 [Render](https://render.com/) 註冊

2. **創建 Web Service**：
   - 點擊 "New +" → "Web Service"
   - 連接你的 GitHub 儲存庫

3. **配置設定**：
   - **Name**: `fridge-helper`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`

4. **設定環境變數**：
   在 Render Dashboard 中添加：
   - `CHANNEL_SECRET`
   - `CHANNEL_ACCESS_TOKEN`
   - `GEMINI_API_KEY`
   - `PUBLIC_BASE_URL` (例：`https://fridge-helper.onrender.com`)

5. **部署**：點擊 "Create Web Service"，Render 會自動部署

6. **設定 LINE Webhook**：
   - 複製 Render 生成的 HTTPS URL
   - 在 LINE Developers 設定 Webhook: `https://your-app.onrender.com/callback`

---

## 📖 使用說明

### 基本指令

| 指令 | 說明 | 範例 |
|------|------|------|
| `我家有 [食材]` | 輸入食材並生成食譜 | 我家有 雞肉 洋蔥 大蒜 |
| `加入 [食材]` | 將食材加入冰箱 | 加入 雞肉 洋蔥 |
| `推薦` | 用目前冰箱食材推薦料理 | 推薦 |
| `換食譜` | 用相同食材換一批新食譜 | 換食譜 |
| `做法 N` | 查看第 N 道料理的詳細步驟 | 做法 1 |
| `下一頁` / `上一頁` | 翻頁查看步驟 | 下一頁 |
| `查看冰箱` | 查看目前冰箱內容 | 查看冰箱 |
| `清空冰箱` | 清空所有食材 | 清空冰箱 |
| `+` 或 `開啟按鈕選單` | 顯示 Quick Reply 選單 | + |

### 使用流程範例

1. **加入 LINE Bot 好友**
   - 掃描 QR Code 或搜尋 Bot ID

2. **輸入食材**
   - 輸入：「我家有 雞肉 洋蔥 番茄」
   - Bot 會自動識別食材並生成 3 道料理

3. **查看推薦**
   - Bot 回傳 3 道料理（含成品圖）
   - 每道料理有「看做法」按鈕

4. **選擇做法**
   - 輸入：「做法 1」
   - 查看詳細步驟（含示意圖）

5. **瀏覽步驟**
   - 使用「下一頁」、「上一頁」瀏覽更多步驟

6. **更換菜單**
   - 輸入：「換食譜」
   - 獲得新的料理建議

7. **使用快捷選單**
   - 輸入：「+」
   - 顯示 Quick Reply 按鈕，快速加入食材或操作功能

---

## 📦 程式模組說明

### 核心模組架構

```python
app.py                    # 主程式
├── LINE Bot 初始化
├── Google GenAI 初始化
├── Flask 路由設定
└── 事件處理
```

### 主要功能模組

#### 1. 食材管理模組

```python
add_to_fridge(user_id, items)  # 將食材加入虛擬冰箱
clear_fridge(user_id)          # 清空冰箱
fridge_text(user_id)           # 查看目前冰箱內容
```

#### 2. AI 生成模組

```python
gemini_generate_recipes()      # 使用 Gemini 生成食譜
gemini_steps_with_prompts()    # 為每個步驟生成圖片 prompt
generate_image_url(prompt)     # 使用 Imagen 生成圖片並返回 URL
```

#### 3. 訊息處理模組

```python
reply_recipes()                # 推薦食譜並生成料理圖
reply_steps_with_images()      # 顯示步驟及示意圖
reply_step_page()              # 處理步驟翻頁
```

#### 4. Flex Message 模組

```python
recipe_to_bubble()             # 將食譜轉換為卡片樣式
step_to_bubble()               # 將步驟轉換為卡片樣式
steps_to_flex()                # 組合多個步驟卡片
```

#### 5. Quick Reply 模組

```python
make_quickreply_menu()         # 生成 Quick Reply 快捷選單
```

---

## 🗃️ 資料結構

### 使用者狀態管理

```python
# 使用者冰箱食材列表
user_fridge_list = {
    "user_id": ["雞肉", "洋蔥", ...]
}

# 最近推薦的食譜
recent_recipes = {
    "user_id": [{recipe_1}, {recipe_2}, {recipe_3}]
}

# 步驟瀏覽狀態
step_view_state = {
    "user_id": {
        "recipe_idx": 0,
        "recipe_name": "料理名稱",
        "steps": ["步驟1", "步驟2", ...],
        "img_urls": ["url1", "url2", ...],
        "page": 0
    }
}
```

### 食譜格式

```json
{
  "ingredients": ["雞肉", "洋蔥", "大蒜"],
  "recipes": [
    {
      "name": "宮保雞丁",
      "summary": "經典川菜，香辣開胃",
      "ingredients": ["雞肉", "花生", "辣椒"],
      "steps": [
        "雞肉切丁，醜製15分鐘",
        "熱鍋爆香蔥藑蒜..."
      ],
      "image_prompt": "A photorealistic food photo of Kung Pao Chicken..."
    }
  ]
}
```

---

## 🌟 專案特色

### 1. 🤖 全 AI 驅動

- 從食材識別、食譜創作到圖片生成，全程使用 Google AI 技術
- Gemini 2.5 Flash 提供快速且精準的文字生成
- Imagen 4.0 生成高品質的料理和步驟圖片

### 2. 🎨 視覺化體驗

- 不只有文字食譜，還有精美的料理成品圖
- 每個步驟配有示意圖，看圖就懂如何操作
- Flex Message 卡片設計，美觀易用

### 3. 🧠 智慧避重

- 換食譜時會自動避開已推薦過的菜名
- 確保每次推薦都有新意，不會重複
- 智慧識別食材，去重並標準化

### 4. 🔧 彈性擴展

- 模組化設計，易於添加新功能
- 可輕鬆整合資料庫（PostgreSQL / MongoDB）
- 支援多種環境變數設定方式

### 5. ✨ 用戶友善

- Quick Reply 快捷按鈕，讓操作更便利
- 自然語言交互，無需記憶複雜指令
- 分頁顯示，不會一次顯示太多資訊

---

## ⚠️ 限制與未來改進

### 當前限制

- ✅ 使用記憶體儲存狀態，重啟後資料會清空
- ✅ 圖片生成較耗時（每張約 3-5 秒）
- ✅ 步驟圖生成數量有上限（預設 10 步）
- ✅ API 調用有額度限制

### 未來改進計畫

- ☐ 加入資料庫持久化（PostgreSQL / MongoDB）
- ☐ 實作圖片快取機制，減少重複生成
- ☐ 支援多語言（英文、日文）
- ☐ 加入過敏原標記功能
- ☐ 營養成分分析
- ☐ 烹飪計時器功能
- ☐ 使用者食譜收藏功能
- ☐ 社群分享功能

---

## 📝 授權與致謝

### 授權

本專案僅供學習與研究使用。

### 致謝

- **Google Gemini API** 提供強大的文字生成能力
- **Google Imagen API** 提供高品質圖像生成
- **LINE Messaging API** 提供便利的聊天機器人平台
- **Flask 框架** 提供輕量級 Web 服務

### 開發者

- 開發者：ximena-maker
- GitHub：[https://github.com/ximena-maker](https://github.com/ximena-maker)

---

## 💬 聯繫與支援

如有任何問題或建議，歡迎：

- 🐞 提交 [Issue](https://github.com/ximena-maker/Fridge-Helper/issues)
- 🔧 提交 [Pull Request](https://github.com/ximena-maker/Fridge-Helper/pulls)

---

## 📚 相關連結

- [LINE Developers](https://developers.line.biz/)
- [Google AI for Developers](https://ai.google.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Render](https://render.com/)

---

<div align="center">

**🍳 讓清冰箱小助手幫你輕鬆清理冰箱，做出美味料理！**

⭐ 如果這個專案對你有幫助，歡迎給個 Star 支持！

</div>
