# 仕様書: Alter Imagniner MVP

## 0. 目的
ユーザー好みのレタッチを **教師あり学習 (パラメータ回帰)** で再現する。
MVPでは以下を満たすこと:

- 画像 (JPG) を入力し、6つのレタッチパラメータ $\theta$ を適用して出力JPGを生成できる
- 同じエンジン (同じ処理) で以下が一貫して動作する
  - データ収集 (手動スライダーで after + $\theta$ 保存)
  - 学習 (画像 $\rightarrow$ $\theta$ を回帰)
  - 推論 (新規画像 $\rightarrow$ $\theta$ $\rightarrow$ after を生成)

## 1. スコープ

### 1.1 対象 (実装する機能)
- レタッチ (グローバルのみ、部分補正なし)
- パラメータ (6つ)
  - Exposure (EV)
  - Contrast
  - Gamma (Midtone)
  - Saturation
  - Temp (WB)
  - Tint (WB)
- 成果物 (3つ)
  - 学習用データ収集レタッチアプリ (Streamlit)
  - 学習システム (PyTorch)
  - 推論アプリ (Streamlit): 入力 $\rightarrow$ 出力 (予測$\theta$表示付き)

### 1.2 非対象 (MVPでやらない)
- トーンカーブ、LUT
- ローカル補正 (マスク、人物補正)
- シャープ、ノイズ除去
- RAW対応 (JPGのみ)
- バッチ処理、履歴、プリセット管理 (最低限のみ)

## 2. 全体アーキテクチャ

### 2.1 構成
- retouch_engine/: 画像処理 (Pillow/OpenCV + numpy) ※共通ライブラリ
- collector_app/: データ収集UI (Streamlit)
- trainer/: 学習スクリプト (PyTorch)
- infer_app/: 推論UI (Streamlit)

### 2.2 データフロー
- 収集アプリ: before.jpg + $\theta$ $\rightarrow$ after.jpg 生成 $\rightarrow$ 保存
- 学習: before.jpg を入力、$\theta$ を正解ラベルとして回帰学習
- 推論: before.jpg $\rightarrow$ $\theta_{pred}$ $\rightarrow$ 同じエンジンで after_pred.jpg

## 3. データ仕様

### 3.1 ディレクトリ構成 (固定)
```
dataset/
  before/         # 入力画像
  after/          # 手動レタッチ後の出力
  params/         # 画像ごとのパラメータjson
  index.csv       # 全サンプル一覧 (学習用)
```
- 保存先は常に dataset/ をルートとする (collector_appの保存先は固定)

### 3.2 サンプルID
- id は4桁ゼロパディング (例: 0001)
- 採番ルール: dataset/ 以下の既存IDの最大値 + 1
- 欠番は埋めず、連番の最大値からのみ増やす

ファイル名:
- before/0001.jpg
- after/0001.jpg
- params/0001.json

### 3.3 params JSON schema (必須)
params/{id}.json
```json
{
  "exposure_ev": 0.65,
  "contrast": 1.12,
  "gamma": 0.95,
  "saturation": 1.08,
  "temp": 0.20,
  "tint": -0.10
}
```

### 3.4 index.csv (必須)
```
id,before_path,after_path,exposure_ev,contrast,gamma,saturation,temp,tint
0001,before/0001.jpg,after/0001.jpg,0.65,1.12,0.95,1.08,0.20,-0.10
```

## 4. レタッチパラメータ $\theta$ の定義

### 4.1 パラメータ一覧と推奨レンジ (UIスライダー)
- exposure_ev: [-2.0, +2.0] (EV)
- contrast: [0.70, 1.30]
- gamma: [0.70, 1.30] (1.0で無変化)
- saturation: [0.70, 1.30]
- temp: [-1.0, +1.0] (WBの青 ⇔ 黄)
- tint: [-1.0, +1.0] (WBの緑 ⇔ マゼンタ)

### 4.2 WBの実装方針 (MVP)
- temp と tint を内部的にRGBゲインへ変換して適用
- 係数 (0.10, 0.05) はMVPでは固定し、学習・推論・収集で共通にする

例:
```
gain_r = 1 + 0.10*temp - 0.05*tint
gain_g = 1 + 0.10*tint
gain_b = 1 - 0.10*temp - 0.05*tint
```

### 4.3 モデル出力のレンジ変換 (共通)
- tanh 出力 ([-1, 1]) を各パラメータの実レンジへ線形スケーリング
- clampは行わない (端の表現力を優先)
- trainer と infer で必ず同一の関数を共有する

```
real = (tanh + 1) / 2 * (max - min) + min
```

## 5. 画像処理エンジン仕様 (retouch_engine)

### 5.1 入出力
- 入力: uint8 RGB (0-255) JPG
- 内部処理: float32 RGB (0-1)
- 出力: uint8 RGB (0-255) JPG
- 色管理: 入力JPGにICCプロファイルがある場合はsRGBへ変換してから処理する
  - ICCが無い場合はsRGBとして扱う

### 5.2 処理順 (固定)
1. sRGB $\rightarrow$ Linear RGB
2. WB (RGBゲイン)
3. Exposure (EV)
4. Contrast
5. Gamma (Midtone)
6. Saturation
7. クリップ [0, 1]
8. Linear $\rightarrow$ sRGB

### 5.3 色空間変換 (必須)
- sRGB $\rightarrow$ linear (近似でOK、標準式推奨)
- linear $\rightarrow$ sRGB (逆変換)

### 5.4 各処理の数式 (MVP実装)
WB (linear RGB):
```
rgb_lin *= [gain_r, gain_g, gain_b]
```

Exposure (EV):
```
rgb_lin *= 2 ** exposure_ev
```

Contrast (linear RGB):
```
luma = 0.2126*R + 0.7152*G + 0.0722*B
rgb_lin = (rgb_lin - luma) * contrast + luma
```

Gamma (linear RGB):
```
gamma は 1.0 が無変化
rgb_lin = rgb_lin ** (1 / gamma)
```

Saturation (固定: HSV):
```
RGB -> HSV
S *= saturation
S = clamp(S, 0, 1)
HSV -> RGB
```

### 5.5 API (Python関数)
- retouch_engine/engine.py
- 関数: apply_retouch(img_rgb_u8: np.ndarray, params: dict) -> np.ndarray
- params はJSONと同じキー
- 入出力はRGBのuint8

## 6. データ収集アプリ仕様 (collector_app / Streamlit)

### 6.1 目的
- ユーザーが好みでスライダー調整し、after画像とparamsを保存する
- 1枚あたり10〜20秒で処理できるUI

### 6.2 UI要件
- 画像アップロード (JPG)
- スライダー6本 (上記レンジ)
- プレビュー表示 (2カラム)
  - 左: Before
  - 右: After (現在スライダーの結果)
- 保存ボタン
  - 次のIDを採番して保存
  - before/{id}.jpg
  - after/{id}.jpg (エンジン出力)
  - params/{id}.json
  - index.csv に追記

### 6.3 保存仕様
- before画像は入力そのままを保存 (リサイズしない)
- after画像はエンジン出力 (同解像度)
- index.csv は存在しなければ新規作成
- 非JPGはsRGBに変換してJPG保存する (拡張子は .jpg に統一)

### 6.4 エラーハンドリング
- 保存失敗時はメッセージ表示
- 中途半端なファイルを避けるため、一時ファイルに保存後、全て成功した時点でリネーム

## 7. 学習システム仕様 (trainer / PyTorch)

### 7.1 目的
- before.jpg から $\theta$ (6次元) を回帰するモデルを学習
- 学習済み重みを保存して推論アプリで利用

### 7.2 入力データ
- index.csv を読み込む
- 入力: before_path の画像
- 正解: CSVの exposure_ev, contrast, gamma, saturation, temp, tint

### 7.3 前処理
- 画像サイズ: 短辺256にリサイズ (アスペクト比維持)、中央224にセンタークロップ
- 正規化: ImageNet平均分散 (pretrainedを使うため)
- データ拡張 (MVP最小): なし or 軽いランダムクロップ程度

### 7.4 出力制約 (重要)
- モデル出力は [-1, 1] にして、後段でレンジにマップ
- 例: tanh 出力 $\rightarrow$ denorm() で実レンジへ (4.3の線形スケーリング)
- denormは retouch_engine と同じ定義の関数を共有

### 7.5 モデル
- Backbone: ResNet18 pretrained=True
- Head: Linear(512->128) + ReLU + Linear(128->6) + tanh

### 7.6 損失関数
- MVPは **パラメータ回帰 (MSE or SmoothL1)** で開始
- loss = SmoothL1Loss(theta_pred, theta_gt)

追加オプション (後回し可):
- after_pred = apply_retouch(before, theta_pred)
- loss_img = L1(after_pred, after_gt)

### 7.7 学習設定 (デフォルト)
- optimizer: AdamW
- lr: 1e-4
- epochs: 30 (まずは10でもOK)
- split: train/val = 8/2 (固定seed)

### 7.8 出力成果物
- artifacts/
  - model.pt (state_dict)
  - config.json (レンジ・画像サイズ・前処理情報)
  - metrics.json (val loss)

## 8. 推論アプリ仕様 (infer_app / Streamlit)

### 8.1 目的
- 入力JPG $\rightarrow$ $\theta_{pred}$ $\rightarrow$ after_pred を生成し表示・保存
- ユーザーが「一発レタッチ」の使用感を確認できる

### 8.2 UI要件
- 画像アップロード
- 推論ボタン
- 3カラム表示
  - Before
  - AI After
  - (オプション) 手動After (同一IDがある場合のみ)
- $\theta_{pred}$ の表示 (数値テーブル)
- 保存ボタン: output/{timestamp}_after.jpg 等

### 8.3 推論処理
- 学習と同じ前処理 (リサイズ・正規化・センタークロップ)
- $\theta_{pred}$ を実レンジに復元
- retouch_engineで apply_retouch (原寸画像に適用してOK)
- 注意: モデル入力は縮小でも、出力は原寸に $\theta$ を適用する

## 9. 評価仕様 (最低限)

### 9.1 定性評価 (必須)
- 推論アプリで Before / AI After を並べて確認
- 学習データがある場合は Before / GT After / AI After の比較

### 9.2 定量評価 (任意)
- val loss (パラメータ)
- 画像損失 (導入した場合のみ)

## 10. 受け入れ条件 (Acceptance Criteria)

### データ収集アプリ
- 画像1枚を読み込み、6スライダーでプレビューが更新される
- 保存ボタンで before/ after/ params/ index.csv が一貫して作られる
- 連番IDが重複しない

### 学習
- `python trainer/train.py --data dataset --out artifacts` で学習完了
- artifacts/model.pt が生成される
- val loss が初期より低い値になる

### 推論
- infer_app で画像を入れるとAI Afterが表示され、保存できる
- 予測$\theta$が表示される

## 11. 実装メモ (LLM向け指針)
- retouch_engine は collector / trainer / infer で必ず共有 (同一コード)
- パラメータのレンジ定義は configとして一箇所に集約
- numpy + OpenCV で高速に。Pillowは入出力に使用してもよい
- StreamlitはUI最小: 完成度より回転速度
