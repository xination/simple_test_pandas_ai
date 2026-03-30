# 開發筆記

## 目前範圍

這個 workspace 目前包含一份全新實作的輕量互動式 pandas AI helper，主程式位於 `pandas_ai/`。

公開 API：

- `setup_ai(..., color=None, stream=True, stream_output=False, stream_delay=0.0, stream_handler=None)`
- `ask_ai(text, dfs=None, output_format="text")`

參考 repo：

- `repo_inspect/` 是原始 `DashyDashOrg/pandas-llm` 專案的 clone，目前僅用於比較與參考。

## 目前架構

核心套件結構：

- `pandas_ai/api.py`
  - 對外公開的進入點。
  - 保存 singleton session。
- `pandas_ai/interactive.py`
  - 一個可選的小工具，用來在一般 Python REPL 中避免 `AIResult` 被重複回顯。
- `pandas_ai/session.py`
  - 負責協調 prompt 建立、backend 呼叫、可選的串流副作用，以及回應解析。
- `pandas_ai/config.py`
  - 處理預設值與環境變數解析。
- `pandas_ai/schema.py`
  - 正規化 dataframe 輸入，並建立 dataframe schema context。
  - 支援 `dfs=None`、單一 dataframe、多個 dataframe 序列，或名稱對 dataframe 的 mapping。
- `pandas_ai/prompting.py`
  - 根據 dataframe schema 與使用者需求建立 user prompt。
- `pandas_ai/parsing.py`
  - 從純文字或 fenced code block 中擷取程式碼。
- `pandas_ai/backends/anthropic.py`
  - Anthropic Messages API backend。
- `pandas_ai/backends/openai_compat.py`
  - 給 LM Studio 與類似本機服務使用的 OpenAI-compatible backend。

## 目前狀態

已完成：

- 預設 backend 為透過 OpenAI-compatible HTTP API 的 `lmstudio`。
- 預設 LM Studio base URL 為 `http://192.168.40.1:1234/v1`。
- 預設 LM Studio model 為 `google/gemma-3-4b`。
- 仍保留 Anthropic `claude` backend 作為可選替代。
- 預設已啟用串流模式，會透過 handler 輸出 chunk，同時仍回傳最終結果。
- `stream_output=True` 會使用內建的 stdout handler。
- `stream_delay` 可刻意放慢 chunk 處理速度，方便 demo 與人工確認。
- 支援環境變數：
  - `PANDAS_AI_BACKEND`
  - `PANDAS_AI_MODEL`
  - `PANDAS_AI_TIMEOUT`
  - `PANDAS_AI_BASE_URL`
  - `PANDAS_AI_SYSTEM_PROMPT`
  - `ANTHROPIC_API_KEY`
- 當 `output_format="text"` 時，`ask_ai()` 會回傳可直接複製貼上的程式碼文字。
- 串流是在 `setup_ai()` 階段設定，並可預設將部分 chunk 輸出到 stdout。
- `setup_ai(..., stream_handler=callable)` 可自訂串流 chunk 的處理方式，而不改變 `ask_ai()` 的回傳型別。
- `output_format="json"` 會先組裝完整回應，最後才解析 JSON。
- `ask_ai(dfs=None)` 會嘗試從 Python 呼叫堆疊中尋找呼叫端的 `df`。
- 單元測試涵蓋 config、backend payload 結構、prompt 組裝、parsing，以及 dataframe 正規化。

目前限制：

- 除了最小模組結構外，尚未加入 packaging metadata。
- 尚未提供對話記憶。
- 不會自動執行程式碼，也沒有 sandbox。
- 尚未實作本機 transformer backend。
- 開發時所使用的環境沒有安裝 `pandas`，因此測試使用 dataframe-like 的 dummy objects。

範例：

- `examples/chicago_housing_demo.py` 提供一個小型的 Chicago housing 風格記憶體內資料集，以及一個簡單的 `ask_ai()` 範例呼叫。

## 測試

主要使用的測試指令：

```bash
python -m unittest tests.test_pandas_ai -v
```

額外執行過的語法驗證：

```bash
python -m py_compile pandas_ai/__init__.py pandas_ai/api.py pandas_ai/config.py pandas_ai/errors.py pandas_ai/schema.py pandas_ai/prompting.py pandas_ai/parsing.py pandas_ai/session.py pandas_ai/backends/__init__.py pandas_ai/backends/base.py pandas_ai/backends/anthropic.py pandas_ai/backends/openai_compat.py tests/test_pandas_ai.py
```

## 下一輪建議

當切換到已安裝 `pandas` 的 Linux workspace 後，下一步較有價值的工作如下：

1. 以真實 pandas DataFrame 測試取代或補強目前的 dummy dataframe 測試。
2. 增加一個適用於 REPL / IPython 的小型 end-to-end 範例腳本。
3. 如果這個專案要提供安裝，加入 packaging 檔案。
4. 決定 `ask_ai(dfs=None)` 的 stack inspection 是否適合作為長期設計，還是應改為明確的 namespace 參數。
5. 如果 air-gap 使用情境是第一級需求，加入可選的本機 backend 實作。

## Git / Workspace 備註

- 目前的 workspace root 是此 repository 的本機 clone。
- `repo_inspect/` 屬於第三方參考程式碼，除非是刻意同步或比較行為，否則不應修改。
