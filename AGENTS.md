# SDAR Pure OPD 注釈ミラー作業規約

## 対象

- Repository: `asayu255/SDAR`
- 読み取り専用 source branch: `claude/pure-opd-multitask`
- 成果物 work branch: `codex/explain-pure-opd-multitask`
- 固定 source commit: `10cacaf6fb2cd4eea971b06d3e73ada868b611f4`

## 変更範囲

- 元ソースは変更しない。
- 注釈版は `docs/code_explanation/annotated/` に元のパスを保って配置する。
- 検証ツールは `tools/code_annotation/` に配置する。
- 追加する解説コメントには必ず `[EXPLAIN]` を含める。
- 注釈コメントを除去した結果が元ファイルと一致することを、commit 前に検証する。

## 禁止事項

- source branch、元コード、既存テスト、実験設定の変更
- PR 作成・更新・コメント
- merge、rebase、reset、force push
- formatter による全体変更、import 整理、変数名変更、元コメント削除
- 未検証 commit、coverage の捏造、推測の事実化

## Git

- work branch への通常の fast-forward push のみ許可する。
- ユーザー由来の未コミット変更が見つかった場合は、破棄・stash・上書きせず停止する。
- source branch が work branch の ancestor でなくなった場合は、状態を記録して停止する。

## 完了判定

- `docs/code_explanation/manifest.json` に `pending` がない。
- completed ファイルの source preservation と annotation coverage が 100%。
- 対応する syntax 検証が成功し、baseline からテスト状態が悪化していない。
- architecture 文書、各種 index、ambiguity/dormant-config report が完成している。
- 全変更が work branch に commit 済みで、PR は作成されていない。
