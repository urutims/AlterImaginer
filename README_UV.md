# uv での環境構築

## 前提

- **Git** と **uv** がインストールされていること  
  - 確認: `git --version` / `uv --version`

## 重要: 実行するディレクトリ

uv は **プロジェクトルート外** の path 依存を正しく扱えないため、  
**必ず `retouch_engine` と同じ階層（RJP_2025）で** `uv sync` を実行してください。

`pyproject.toml` と `.python-version` は **RJP_2025** フォルダに配置してあります。

```powershell
cd "g:\マイドライブ\授業内\RJP_2025"
uv sync
```

- `.venv` は RJP_2025 に自動作成されます。
- 実行例:
  - 収集アプリ: `uv run python collector_app/dpg_app.py`
  - 学習: `uv run python trainer/train.py --data dataset --out artifacts`
  - 推論UI: `uv run streamlit run infer_app/app.py`

詳細は `ENV_SETUP_UV_GIT_JA.txt` を参照してください。

## 依存の更新

- 依存を追加・変更したら:
  1. `pyproject.toml` を編集
  2. `uv lock`
  3. `uv sync` で動作確認
  4. `pyproject.toml` と `uv.lock` をコミット
