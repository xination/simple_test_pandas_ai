# pandas_ai

✨ 一個輕量、互動式的 pandas AI 小工具，主要提供兩個公開函式：

- `setup_ai()`
- `ask_ai()`

## 🚀 專案特色

- 🤖 預設使用 LM Studio 相容 API（`backend="lmstudio"`）
- 🧠 預設模型為 `google/gemma-3-4b`
- 🌐 預設 `base_url` 為 `http://192.168.40.1:1234/v1`
- 🏠 支援透過 OpenAI 相容介面串接 LM Studio
- 📡 串流輸出在 `setup_ai()` 階段設定，且預設啟用
- 🐍 串流實作只依賴 Python 標準函式庫

## 🧩 最小使用範例

```python
from pandas_ai import setup_ai, ask_ai

setup_ai()
result = ask_ai("顯示前 5 筆資料", df)
print(result)
```

```python
from pandas_ai import setup_ai, ask_ai

setup_ai(
    backend="lmstudio",
    base_url="http://127.0.0.1:1234/v1",
    model="local-model",
    stream=False,
)
print(ask_ai("依照 user_id 合併 df0 和 df1", [df0, df1]))
```

## 📡 串流輸出說明

- `stream=True` 時，`ask_ai()` 仍然會回傳最終結果。
- `stream_output=True` 可把串流片段直接印到標準輸出。
- `stream_parse_code=True` 可在文字串流時過濾掉 ` ```python ` 與結尾 ` ``` `。
- 也可以傳入自訂的 `stream_handler` 來接手處理串流內容。
- 預設 handler 是靜默的，因此在 REPL 裡不會重複輸出相同內容。
- 如果你想讓串流顯示慢一點、方便肉眼觀察，可以設定 `stream_delay`。

```python
setup_ai(stream=True, stream_output=True, stream_delay=0.03)
result = ask_ai("顯示前 5 筆資料", df)
```

```python
setup_ai(stream=True, stream_output=True, stream_parse_code=True)
result = ask_ai("顯示前 5 筆資料", df)
```

```python
setup_ai(
    stream=True,
    stream_output=True,
    color="\033[102m",  # None 表示不加顏色
)
result = ask_ai("顯示前 5 筆資料", df)
```

- `color=None` 時，不做任何 ANSI 著色。
- `color="\033[102m"` 這類值可讓內建 stdout 串流輸出帶顏色。
- `stream_parse_code=True` 只影響串流顯示，不會改變最後 `ask_ai()` 的回傳值。
- 如果你有啟用 `enable_ai_result_displayhook()`，非串流情況下 REPL 自動顯示的最終結果也會套用同一個顏色。
- 如果你是自己寫 `print(ask_ai(...))`，顏色不會自動介入，因為 `ask_ai()` 仍維持回傳純文字，避免污染可複製的程式碼字串。

## 🖥️ REPL 小技巧

如果你是在一般 Python 互動模式中執行，想看到串流輸出、又不希望最終結果被再次自動回顯，可以先啟用一次：

```python
from pandas_ai import enable_ai_result_displayhook

enable_ai_result_displayhook()
```

## 🧾 JSON 輸出注意事項

當 `output_format="json"` 時，串流中的部分片段不保證是合法 JSON。系統會在完整回應組裝完成後，才進行解析。

## 🔒 安全聲明

- `pandas_ai` 只會回傳模型生成的文字結果或 Python / pandas 程式碼字串。
- `pandas_ai` 本身 **不會自動執行** 生成的 code。
- 🧑‍💻 建議你在 REPL、notebook 或 script 中 **先人工檢查** 生成內容，再決定是否執行。
- 🛡️ 在 air-gap 或本機模型情境下，這樣的使用方式會比自動執行更安全、也更容易追蹤問題。

## 🧭 `dfs=None` 行為說明

- 當你沒有顯式傳入 `dfs=` 時，`ask_ai()` 會嘗試從目前呼叫堆疊中尋找名為 `df` 的 dataframe。
- 如果有找到，就會把它當作預設資料框使用。
- 如果沒有找到，會拋出錯誤，要求你明確提供 `dfs=`。
- ✅ 在一般 REPL / notebook 使用情境下，如果你前面已經讀入資料並使用 `df` 這個變數名稱，通常可以直接呼叫 `ask_ai("...")`。
- 🧭 如果你同時操作多個 dataframe，或想讓分析對象更明確，則建議顯式傳入 `dfs=`，例如 `ask_ai("...", dfs=df)`。

## 🛠️ 開發說明

更多架構、目前狀態與後續規劃，請參考 `DEVELOPMENT.md`。

## 🎬 Demo

- 🏘️ Chicago housing 風格範例：`examples/chicago_housing_demo.py`
- 🤗 本機 Hugging Face 串流範例：`examples/example2_local_qwen_stream.py`

```bash
python examples/chicago_housing_demo.py
```

這個 demo 會建立一個小型的記憶體內資料集，欄位包含 `community_area`、`housing_type`、`sqft` 與 `list_price`，再請模型生成 pandas 程式碼，完成簡單的房屋資料分析任務。
