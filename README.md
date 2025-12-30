# Linebot_NLP_Project
清冰箱小助手

## 專案簡介

「清冰箱小助手」是一個基於 LINE Bot 的智慧料理推薦助手，使用 Google Gemini AI 和 Imagen 圖像生成技術，幫助使用者有效運用冰箱食材，減少食物浪費。

### 核心功能

✅ **智慧食材識別**  
- 只需輸入一句話（例如：「我家有 雞肉 洋蔥 花椰菜」），AI 自動提取食材
- 支援手動加入食材到虛擬冰箱

✅ **AI 食譜生成**  
- 根據現有食材生成至少 3 道不同的家常料理
- 每道料理包含：菜名、簡介、食材清單、詳細步驟
- 使用 Imagen 為每道料理生成精美成品示意圖

✅ **一鍵換食譜**  
- 不喜歡目前推薦？輸入「換食譜」即可基於相同食材生成全新料理組合
- AI 會自動避開已推薦過的菜名，確保多樣性

✅ **視覺化步驟指導**  
- 選擇任一料理查看詳細做法
- 每個烹飪步驟都配有 AI 生成的示意圖
- 支援分頁瀏覽（每頁 5 個步驟），輕鬆翻頁查看

✅ **快速操作選單**  
- 提供 Quick Reply 快捷按鈕
- 快速加入常見食材（雞肉、牛肉、豬肉等）
- 一鍵查看冰箱、清空冰箱、換食譜等功能

## 技術架構

### 主要技術棧

- **後端框架**: Flask
- **LINE Bot SDK**: line-bot-sdk
- **AI 模型**: 
  - Google Gemini 2.5 Flash（文字生成、食譜創作）
  - Imagen 4.0（圖像生成）
- **開發語言**: Python 3.x

### 系統架構

```
使用者輸入
    ↓
LINE Bot Webhook
    ↓
Flask 後端處理
    ↓
├─ Gemini API（食材提取 + 食譜生成）
├─ Imagen API（料理圖片 + 步驟示意圖）
└─ 狀態管理（記憶體存儲）
    ↓
Flex Message 呈現
    ↓
回覆給使用者
```

### 核心模組

1. **食材管理模組**
   - `add_to_fridge()`: 將食材加入虛擬冰箱
   - `clear_fridge()`: 清空冰箱
   - `fridge_text()`: 查看目前冰箱內容

2. **AI 生成模組**
   - `gemini_generate_recipes()`: 使用 Gemini 生成食譜
   - `gemini_steps_with_prompts()`: 為每個步驟生成圖片 prompt
   - `generate_image_url()`: 使用 Imagen 生成圖片並返回 URL

3. **訊息處理模組**
   - `reply_recipes()`: 推薦食譜並生成料理圖
   - `reply_steps_with_images()`: 顯示步驟及示意圖
   - `reply_step_page()`: 處理步驟翻頁

4. **Flex Message 模組**
   - `recipe_to_bubble()`: 將食譜轉換為卡片樣式
   - `step_to_bubble()`: 將步驟轉換為卡片樣式
   - `steps_to_flex()`: 組合多個步驟卡片

## 環境設置

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

### 安裝步驟

1. **克隆專案**
```bash
git clone https://github.com/ximena-maker/Fridge-Helper.git
cd Fridge-Helper
```

2. **安裝依賴**
```bash
pip install -r requirements.txt
```

3. **設定環境變數**
   - 方式一：創建 `.env` 檔案
   - 方式二：直接設定系統環境變數
   - 方式三（開發用）：創建 `keys.txt` 檔案
     ```
     CHANNEL_SECRET=your_secret
     CHANNEL_ACCESS_TOKEN=your_token
     ```

4. **執行程式**
```bash
python app.py
```

5. **設定 LINE Webhook**
   - 進入 LINE Developers Console
   - 設定 Webhook URL: `https://your-domain.com/callback`
   - 啟用 Webhook

## 部署指南

### Render 部署（推薦）

1. 在 Render 創建新的 Web Service
2. 連接 GitHub 儲存庫
3. 設定環境變數（參考上方「必要環境變數」）
4. Render 會自動偵測 `requirements.txt` 並安裝依賴
5. 啟動命令：`python app.py`
6. 複製生成的 HTTPS URL 並設定到 LINE Webhook

### 注意事項

⚠️ **重要**: 
- 必須使用 HTTPS URL，否則 LINE 無法載入生成的圖片
- 確保 `PUBLIC_BASE_URL` 設定正確
- 若 API Key 被標記為 leaked，需立即更換新的 Key
- 圖片會儲存在 `static/generated` 資料夾，定期清理避免磁碟爆滿

## 使用說明

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

### 使用流程

1. **加入 LINE Bot 好友**
2. **輸入食材**：「我家有 雞肉 洋蔥 番茄」
3. **查看推薦**：Bot 回傳 3 道料理（含成品圖）
4. **選擇做法**：輸入「做法 1」查看詳細步驟
5. **瀏覽步驟**：使用「下一頁」查看更多步驟
6. **更換菜單**：輸入「換食譜」獲得新的料理建議

## 資料結構

### 使用者狀態管理

```python
# 使用者冰箱食材列表
user_fridge_list = {"user_id": ["雞肉", "洋蔥", ...]}

# 最近推薦的食譜
recent_recipes = {"user_id": [{recipe_1}, {recipe_2}, {recipe_3}]}

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
        "雞肉切丁，醃製15分鐘",
        "熱鍋爆香蔥薑蒜..."
      ],
      "image_prompt": "A photorealistic food photo of Kung Pao Chicken..."
    }
  ]
}
```

## 專案特色

🌟 **全 AI 驅動**  
從食材識別、食譜創作到圖片生成，全程使用 Google AI 技術

🌟 **視覺化體驗**  
不只有文字食譜，還有精美的料理成品圖和步驟示意圖

🌟 **智慧避重**  
換食譜時會自動避開已推薦過的菜名，確保多樣性

🌟 **彈性擴展**  
模組化設計，易於添加新功能（如食材過敏提醒、營養分析等）

🌟 **用戶友善**  
Quick Reply 快捷按鈕讓操作更便利

## 限制與改進方向

### 當前限制

- 使用記憶體存儲狀態，重啟後資料會清空
- 圖片生成較耗時（每張約 3-5 秒）
- 步驟圖生成數量有上限（預設 10 步）
- API 調用有額度限制

### 未來改進

- [ ] 加入資料庫持久化（PostgreSQL / MongoDB）
- [ ] 實作圖片快取機制
- [ ] 支援多語言（英文、日文）
- [ ] 加入過敏原標記
- [ ] 營養成分分析
- [ ] 烹飪計時器功能
- [ ] 使用者食譜收藏功能
- [ ] 社群分享功能

## 開發團隊

開發者：ximena-maker  
GitHub：[https://github.com/ximena-maker](https://github.com/ximena-maker)

## 授權

本專案僅供學習與研究使用。

## 致謝

- Google Gemini API 提供強大的文字生成能力
- Google Imagen API 提供高品質圖像生成
- LINE Messaging API 提供便利的聊天機器人平台
- Flask 框架提供輕量級 Web 服務

---

如有任何問題或建議，歡迎提交 Issue 或 Pull Request！
