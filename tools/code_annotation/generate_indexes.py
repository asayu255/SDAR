"""固定 source と manifest から横断索引を再生成する。"""

from __future__ import annotations

import ast
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

from common import DOC_ROOT, ROOT, decode_text, load_manifest, read_source_bytes


INDEX_ROOT = DOC_ROOT / "indexes"


def md_link(path: str, line: int | None = None) -> str:
    suffix = f"#L{line}" if line else ""
    return f"[`{path}`](../../{path}{suffix})"


def write(name: str, title: str, body: list[str]) -> None:
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    (INDEX_ROOT / name).write_text(
        "\n".join([f"# {title}", "", *body]).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )


def source_text(path: str) -> str | None:
    text, _ = decode_text(read_source_bytes(path))
    return text


def file_index(rows: list[dict]) -> None:
    body = [
        "manifestに記録した全tracked fileを、対象判定と実行関連度で検索する索引。",
        "",
        "| original | language | relevance | status | annotated / reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        target = (
            f"[mirror](../{Path(row['annotated_path']).relative_to('docs/code_explanation').as_posix()})"
            if row["status"] == "completed"
            else row.get("skip_reason", "")
        )
        body.append(
            f"| `{row['original_path']}` | {row['language']} | "
            f"{row['execution_relevance']} | {row['status']} | {target} |"
        )
    write("file_index.md", "ファイル索引", body)


class DefinitionVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.stack: list[str] = []
        self.classes: list[tuple[str, int]] = []
        self.functions: list[tuple[str, int, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualified = ".".join([*self.stack, node.name])
        self.classes.append((qualified, node.lineno))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        qualified = ".".join([*self.stack, node.name])
        args = [argument.arg for argument in node.args.posonlyargs + node.args.args]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        self.functions.append((qualified, node.lineno, f"{kind}({', '.join(args)})"))
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node, "def")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node, "async def")


def symbol_indexes(rows: list[dict]) -> None:
    classes: list[tuple[str, str, int]] = []
    functions: list[tuple[str, str, int, str]] = []
    for row in rows:
        if row["language"] != "python" or row["status"] != "completed":
            continue
        text = source_text(row["original_path"])
        if text is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text, filename=row["original_path"])
        except SyntaxError:
            continue
        visitor = DefinitionVisitor(row["original_path"])
        visitor.visit(tree)
        classes.extend((name, row["original_path"], line) for name, line in visitor.classes)
        functions.extend(
            (name, row["original_path"], line, signature)
            for name, line, signature in visitor.functions
        )

    class_body = [
        "Python sourceのclass定義。nested classはqualified nameで表示する。",
        "",
        "| class | definition |",
        "|---|---|",
        *[
            f"| `{name}` | {md_link(path, line)} |"
            for name, path, line in sorted(classes, key=lambda item: (item[0].lower(), item[1], item[2]))
        ],
    ]
    function_body = [
        "Python sourceのfunction/method定義。signatureは引数名だけを簡略表示する。",
        "",
        "| function | kind / args | definition |",
        "|---|---|---|",
        *[
            f"| `{name}` | `{signature}` | {md_link(path, line)} |"
            for name, path, line, signature in sorted(
                functions, key=lambda item: (item[0].lower(), item[1], item[2])
            )
        ],
    ]
    write("class_index.md", "クラス索引", class_body)
    write("function_index.md", "関数・メソッド索引", function_body)


def config_index(rows: list[dict]) -> None:
    scientific = [
        ("`actor_rollout_ref.actor.loss_type`", "`topk_kl`", "studentの実効蒸留loss"),
        ("`actor_rollout_ref.actor.teacher_kl_loss_coef`", "run script値", "scalar loss係数"),
        ("`actor_rollout_ref.actor.pg_loss_coef`", "`0.0`注入", "policy-gradientを非実効化"),
        ("`actor_rollout_ref.actor.entropy_coeff`", "`0.0`注入", "entropy項を非実効化"),
        ("`actor_rollout_ref.actor.use_kl_loss`", "`false`注入", "通常KL lossを非実効化"),
        ("`actor_rollout_ref.actor.use_sdl_loss`", "`false`注入", "SDLを非実効化"),
        ("`actor_rollout_ref.actor.use_sdar_loss`", "`false`注入", "SDARを非実効化"),
        ("`algorithm.use_kl_in_reward`", "`false`注入", "reward KLを非実効化"),
        ("`data.task_balance.*`", "3 task / 15 prompts", "task-balanced sampler"),
        ("`trainer.total_training_steps`", "`300`", "run length"),
    ]
    body = [
        "Pure OPDの科学的設定と、repository内の設定ファイルを分けて示す。",
        "",
        "## Pure OPDの主要実効設定",
        "",
        "| key | effective value | role |",
        "|---|---|---|",
        *[f"| {key} | {value} | {role} |" for key, value, role in scientific],
        "",
        "Hydraの順序はbase config → CLI override → `main_opd.py`のPure OPD注入 → "
        "expected-config検証である。micro-batchやengine tuningは性能設定であり、"
        "科学的意図を固定するexpectationsとは別扱いになる。",
        "",
        "## 設定ファイル一覧",
        "",
        "| file | type | relevance |",
        "|---|---|---|",
    ]
    config_languages = {"yaml", "toml", "ini", "config", "json", "jsonc"}
    for row in rows:
        if row["status"] == "completed" and (
            row["language"] in config_languages
            or row["original_path"].endswith((".sh", ".bash", ".zsh"))
        ):
            body.append(
                f"| {md_link(row['original_path'])} | {row['file_type']} | "
                f"{row['execution_relevance']} |"
            )
    write("config_index.md", "設定索引", body)


ENV_PATTERNS = (
    re.compile(r"""(?:os\.getenv|os\.environ\.get)\(\s*["']([A-Z][A-Z0-9_]*)["'](?:\s*,\s*([^)]+))?"""),
    re.compile(r"""os\.environ\[\s*["']([A-Z][A-Z0-9_]*)["']\s*\]"""),
    re.compile(r"""\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}"""),
    re.compile(r"""^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=([^\s#;]+)""", re.MULTILINE),
)


def environment_index(rows: list[dict]) -> None:
    found: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "completed":
            continue
        text = source_text(row["original_path"])
        if text is None:
            continue
        for pattern in ENV_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1)
                default = (
                    (match.group(2) or "").strip()
                    if match.re.groups >= 2
                    else ""
                )
                line = text.count("\n", 0, match.start()) + 1
                item = (row["original_path"], line, default)
                if item not in found[name]:
                    found[name].append(item)
    body = [
        "sourceから静的抽出した環境変数。空のdefaultは必須値、直接index、"
        "または静的解析で既定値を確定できない呼び出しを表す。",
        "",
        "| variable | default / assigned expression | locations |",
        "|---|---|---|",
    ]
    for name in sorted(found):
        entries = found[name]
        defaults = sorted({default.replace("|", "\\|") for _, _, default in entries if default})
        locations = ", ".join(md_link(path, line) for path, line, _ in entries[:12])
        if len(entries) > 12:
            locations += f", … (+{len(entries) - 12})"
        body.append(f"| `{name}` | {'; '.join(defaults) or '—'} | {locations} |")
    write("environment_variable_index.md", "環境変数索引", body)


def tensor_shape_index() -> None:
    rows = [
        ("`input_ids`", "`(B, S)`", "integer token id", "student/teacher共通入力"),
        ("`responses`", "`(B, R)`", "integer token id", "student on-policy response"),
        ("`attention_mask`", "`(B, S)`", "bool/int mask", "paddingを除外"),
        ("`response_mask`", "`(B, R)`", "bool/int mask", "有効response tokenだけを集約"),
        ("`teacher_topk_ids`", "`(B, R, K)`", "float32→long", "padding後にround/longへ復元"),
        ("`teacher_topk_logprobs`", "`(B, R, K)`", "floating", "teacher top-k log-softmax"),
        ("`teacher_logsumexp`", "`(B, R)`", "floating", "tail mass算出の正規化項"),
        ("`student_topk_logprobs`", "`(B, R, K)`", "floating", "teacher token ID位置をgather"),
        ("`teacher_kld`", "`(B, R)`", "floating", "top-k support + tail bucket KL"),
        ("`task_ids`", "`(B,)`", "long", "worker側per-task row mask"),
        ("`indices` / `unpad_input`", "`(nnz,)`", "long", "padded tokenからrmpadへの写像"),
        ("`input_ids_rmpad`", "`(1, nnz)`", "integer token id", "FlashAttention varlen入力"),
        ("`logits_rmpad`", "`(1, nnz, V)`", "floating", "pad_input前のmodel出力"),
    ]
    body = [
        "`B`: turn-row batch、`S`: prompt+response sequence、`R`: response長、"
        "`K`: teacher top-k、`V`: vocabulary、`nnz`: paddingを除いたtoken数。",
        "",
        "| tensor | shape | dtype | meaning |",
        "|---|---|---|---|",
        *[f"| {name} | {shape} | {dtype} | {meaning} |" for name, shape, dtype, meaning in rows],
        "",
        "teacher token IDはvocabulary indexのためbf16では256を超える整数を全て厳密表現できない。"
        "DP auto-padding時はfloat32 bufferを経由し、scatter後に`round().long()`でindexへ戻す。",
    ]
    write("tensor_shape_index.md", "Tensor shape索引", body)


def call_graph() -> None:
    body = [
        "Pure OPD production pathを中心にした実効call graph。破線相当の「実験」は"
        "固定sourceでproduction接続が確認できない経路。",
        "",
        "```mermaid",
        "flowchart TD",
        '  A["run_multitask_qwen3.sh"] --> B["main_opd.py"]',
        '  B --> C["TaskRunner.run"]',
        '  C --> D["OPDRayTrainer.fit"]',
        '  D --> E["TrajectoryCollector.multi_turn_loop"]',
        '  E --> F["MultiTaskEnvironmentManager.reset / step"]',
        '  D --> G["compute_teacher_log_probs (batch mutation)"]',
        '  G --> H["task-specific ref worker groups"]',
        '  H --> I["teacher_topk_ids / logprobs / logsumexp"]',
        '  D --> J["actor_rollout_wg.update_actor"]',
        '  J --> K["DataParallelPPOActor.update_policy"]',
        '  K --> L["compute_topk_kl"]',
        '  L --> M["teacher KL scalar loss"]',
        '  N["async_rollout_core.collect_async"] -. "test/scaffolding only" .-> E',
        "```",
        "",
        "Pure OPD thin loopは通常PPOのcritic value、advantage/returns、reward-KL、"
        "policy-gradient phaseを通らない。`compute_teacher_log_probs`はTensorを返すAPIではなく、"
        "入力DataProtoへteacher signalをscatterする副作用境界である。",
    ]
    write("call_graph.md", "Pure OPD call graph", body)


def main() -> None:
    rows = load_manifest()
    file_index(rows)
    symbol_indexes(rows)
    config_index(rows)
    environment_index(rows)
    tensor_shape_index()
    call_graph()
    summary = {
        "files": len(rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "indexes": 7,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
