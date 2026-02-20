# uv での環境構築

## 前提

- **Git** と **uv** がインストールされていること  
  - 確認: `git --version` / `uv --version`

## このフォルダ（alter_imagineer）で使う場合

`pyproject.toml` と `.python-version` はこのフォルダにあります。  
`retouch_engine` は親ディレクトリを参照する設定になっています。

```powershell
cd "g:\マイドライブ\授業内\RJP_2025\alter_imagineer"
uv sync
```

- `.venv` はこのフォルダに自動作成されます。
- 実行例（このフォルダにある collector_app）:
  - 収集アプリ: `uv run python collector_app/dpg_app.py`
- 学習（trainer）・推論（infer_app）はリポジトリルート側にある場合は、リポジトリルートで uv を実行し、`ENV_SETUP_UV_GIT_JA.txt` の手順に従ってください。

## リポジトリルートで使う場合

`pyproject.toml` をリポジトリルート（`retouch_engine` と同じ階層）に置いている場合は、そこで uv を実行します。

1. リポジトリルートに `pyproject.toml` があることを確認する。
2. `pyproject.toml` の `[tool.uv.sources]` を次のようにする:
   ```toml
   retouch_engine = { path = "retouch_engine", editable = true }
   ```
3. リポジトリルートで:
   ```powershell
   uv sync
   uv run python collector_app/dpg_app.py
   ```

詳細は `ENV_SETUP_UV_GIT_JA.txt` を参照してください。

## 依存の更新

- 依存を追加・変更したら:
  1. `pyproject.toml` を編集
  2. `uv lock`
  3. `uv sync` で動作確認
  4. `pyproject.toml` と `uv.lock` をコミット
