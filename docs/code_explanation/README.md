# SDAR Pure OPD 日本語注釈ミラー

このディレクトリは、`claude/pure-opd-multitask` の固定コミットを読み取り専用の基準として作成する、日本語解説付きソースミラーです。

- 元コードは変更しません。
- 注釈版は `annotated/` 以下に元のパスを保って配置します。
- 追加コメントは `[EXPLAIN]` で識別します。
- `tools/code_annotation/` の検証ツールで、元コード保持・注釈範囲・構文を確認します。
- Pure OPD の実効経路と、共有コード上に存在するだけの非実効経路を区別します。

進捗と再開位置は [STATUS.md](STATUS.md)、対象判定は `manifest.json` / `manifest.csv`、source 固定値は `SOURCE_COMMIT` を参照してください。
