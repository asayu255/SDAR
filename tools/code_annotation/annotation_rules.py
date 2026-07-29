"""言語別に、安全にコメントを挿入できる論理行を抽出する。"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path


SPECIAL_SYMBOLS = {
    "compute_teacher_log_probs": (
        "task_name で分割した row を各 teacher worker group へ送り、"
        "得られた teacher signal を元の global row 順で入力 batch に書き戻す。"
    ),
    "update_actor": (
        "student actor の micro-batch forward、teacher 分布との KL、"
        "gradient accumulation、optimizer step を実行する。"
    ),
    "fit": (
        "rollout 生成から teacher forward、actor 更新、検証、checkpoint までの"
        "学習 phase を順序付ける trainer loop である。"
    ),
    "compute_topk_kl": (
        "teacher top-k token support と tail bucket 上で student/teacher KL を"
        "token 単位に計算し、response mask で集約する。"
    ),
    "TaskBalancedSampler": (
        "task ごとの index 集合から同数を選び、distributed batch 内の"
        "task 構成を均衡させる sampler を定義する。"
    ),
    "MultiTaskEnvironmentManager": (
        "mixed batch を task 別 manager へ分配し、reset/step 結果を"
        "元の global batch 順へ復元する。"
    ),
}


def _symbol_from_line(line: str) -> str | None:
    match = re.search(r"\b(?:class|def|async\s+def)\s+([A-Za-z_]\w*)", line)
    return match.group(1) if match else None


def explain_python(line: str) -> str:
    stripped = line.strip()
    symbol = _symbol_from_line(stripped)
    if symbol in SPECIAL_SYMBOLS:
        return SPECIAL_SYMBOLS[symbol]
    if stripped.startswith(("import ", "from ")):
        return (
            "このモジュールで使用する型・設定・分散処理または Tensor 操作の"
            "依存関係を読み込む。"
        )
    if stripped.startswith(("async def ", "def ")):
        return (
            f"`{symbol}` の入力を検証・変換し、呼び出し元が使用する結果または"
            "副作用を生成する処理単位を定義する。"
        )
    if stripped.startswith("class "):
        return (
            f"`{symbol}` として状態と関連処理をまとめ、worker・trainer・"
            "dataset などの責務境界を定義する。"
        )
    if stripped.startswith("@"):
        return "直後の定義へ decorator を適用し、呼び出し規約または実行時属性を付与する。"
    if stripped.startswith(("if ", "elif ")):
        return "実行時の設定または状態を評価し、成立する経路だけを選択する。"
    if stripped.startswith("else"):
        return "直前の条件が成立しない場合の代替経路を実行する。"
    if stripped.startswith(("for ", "async for ")):
        return "batch、micro-batch、task または token の要素を順に処理する。"
    if stripped.startswith("while "):
        return "終了条件を満たすまで rollout または状態更新を反復する。"
    if stripped.startswith(("with ", "async with ")):
        return "context manager で resource、autocast、no_grad または session の寿命を限定する。"
    if stripped.startswith("try"):
        return "失敗し得る処理を開始し、後続の例外処理へ制御を接続する。"
    if stripped.startswith(("except ", "except:")):
        return "発生した例外を捕捉し、fallback、cleanup または文脈付き再送出を行う。"
    if stripped.startswith("finally"):
        return "成功・失敗にかかわらず resource 解放または状態復元を実行する。"
    if stripped.startswith("return"):
        return "計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。"
    if stripped.startswith("yield"):
        return "現在の要素を逐次呼び出し元へ渡し、反復状態を保持する。"
    if stripped.startswith("raise"):
        return "不正な設定・shape・task または実行状態を例外として明示する。"
    if stripped.startswith("assert "):
        return "後続処理が依存する shape、dtype、設定または分散条件を検証する。"
    if stripped.startswith(("del ", "global ", "nonlocal ")):
        return "後続処理で参照する状態の寿命またはスコープを明示的に調整する。"
    if stripped.startswith(("break", "continue", "pass")):
        return "現在の分岐または反復の制御を明示する。"
    if re.match(r"[A-Za-z_][\w.\[\], ]*\s*(?::[^=]+)?=", stripped):
        return (
            "後続の計算・routing・mask・metric で参照する値を構築し、"
            "現在のスコープまたは batch に保持する。"
        )
    if "(" in stripped:
        return (
            "必要な引数と現在の状態を渡して処理を呼び出し、戻り値または"
            "batch への副作用を次の段階へ接続する。"
        )
    return "この論理行で現在の処理ブロックに必要な状態または制御を定義する。"


def python_statement_lines(text: str) -> set[int]:
    starts: set[int] = set()
    at_start = True
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for token in tokens:
            token_type, token_text, start, _, _ = token
            if token_type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
                tokenize.NL,
            }:
                continue
            if token_type == tokenize.NEWLINE:
                at_start = True
                continue
            if at_start and token_text != ";":
                starts.add(start[0])
                at_start = False
    except (IndentationError, tokenize.TokenError):
        return set()
    return starts


def yaml_statement_lines(lines: list[str]) -> set[int]:
    result: set[int] = set()
    scalar_indent: int | None = None
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if scalar_indent is not None:
            if stripped and indent > scalar_indent:
                continue
            scalar_indent = None
        if not stripped or stripped.startswith("#"):
            continue
        result.add(number)
        if re.search(r"(?:^|:\s*)[>|][+-]?\s*(?:#.*)?$", stripped):
            scalar_indent = indent
    return result


def shell_statement_lines(lines: list[str]) -> set[int]:
    result: set[int] = set()
    heredoc_end: str | None = None
    continued = False
    paren_depth = 0
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if heredoc_end:
            if stripped == heredoc_end:
                heredoc_end = None
            continue
        if not stripped or stripped.startswith("#"):
            continued = stripped.endswith("\\") if stripped else continued
            continue
        if not continued and paren_depth <= 0:
            result.add(number)
        match = re.search(r"<<-?\s*['\"]?([A-Za-z_]\w*)['\"]?", line)
        if match:
            heredoc_end = match.group(1)
        paren_depth += line.count("(") - line.count(")")
        continued = line.rstrip().endswith("\\")
    return result


def markdown_statement_lines(lines: list[str]) -> set[int]:
    result: set[int] = set()
    in_fence = False
    previous_blank = True
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            if not in_fence:
                result.add(number)
            in_fence = not in_fence
            previous_blank = False
            continue
        if in_fence or not stripped or stripped.startswith("<!--"):
            previous_blank = not stripped
            continue
        if (
            previous_blank
            or stripped.startswith(("#", "-", "*", "+", ">", "|"))
            or re.match(r"\d+[.)]\s", stripped)
        ):
            result.add(number)
        previous_blank = False
    return result


def generic_statement_lines(lines: list[str], comment_prefixes: tuple[str, ...]) -> set[int]:
    result = set()
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(comment_prefixes):
            continue
        result.add(number)
    return result


def eligible_lines(language: str, text: str) -> set[int]:
    lines = text.splitlines()
    if language == "python":
        return python_statement_lines(text)
    if language in {"yaml", "toml", "ini", "config"}:
        return yaml_statement_lines(lines)
    if language in {"shell", "make"}:
        return shell_statement_lines(lines)
    if language in {"markdown", "rst", "text"}:
        return markdown_statement_lines(lines)
    if language in {"json", "jsonc"}:
        return generic_statement_lines(lines, ("//",))
    return generic_statement_lines(lines, ("//", "/*", "*"))


def explanation_for(language: str, line: str, path: str) -> str:
    if language == "python":
        return explain_python(line)
    stripped = line.strip()
    if language == "shell":
        if "=" in stripped and not stripped.startswith(("if ", "for ", "while ")):
            return "実行設定または後続 command が参照する環境値・引数を定義する。"
        return "実験起動、環境準備または検証 command の一段階を実行する。"
    if language in {"yaml", "toml", "ini", "config"}:
        return "この設定項目の値を、entry point または worker が読み取る構成として定義する。"
    if language in {"markdown", "rst", "text"}:
        if stripped.startswith("#"):
            return "以下の節で扱う設計・実行経路・制約の範囲を示す見出しである。"
        return "この段落は実装の意図、利用条件または検証上の注意を説明する。"
    if language in {"json", "jsonc"}:
        return "この JSON 要素は実行時設定またはデータ構造の一部を保持する。"
    return "この行は現在の処理ブロックで使用する宣言、状態更新または制御を表す。"


def comment_line(language: str, indent: str, explanation: str) -> str:
    explanation = " ".join(explanation.split())
    if language == "markdown":
        return f"{indent}<!-- [EXPLAIN] {explanation} -->"
    if language == "rst":
        return f"{indent}.. [EXPLAIN] {explanation}"
    if language in {
        "javascript",
        "typescript",
        "c",
        "cpp",
        "java",
        "go",
        "rust",
        "protobuf",
        "json",
        "jsonc",
    }:
        return f"{indent}// [EXPLAIN] {explanation}"
    return f"{indent}# [EXPLAIN] {explanation}"


def make_annotated_text(language: str, text: str, path: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    plain_lines = text.splitlines()
    eligible = eligible_lines(language, text)
    output: list[str] = []
    inserted = 0
    for number, original in enumerate(lines, 1):
        content = original.rstrip("\n")
        newline = original[len(content) :]
        if number in eligible:
            indent = content[: len(content) - len(content.lstrip())]
            explanation = explanation_for(language, content, path)
            output.append(comment_line(language, indent, explanation) + (newline or "\n"))
            inserted += 1
        output.append(original)
    if not lines and text == "":
        return "", 0
    if len(lines) < len(plain_lines):
        raise AssertionError("line accounting mismatch")
    return "".join(output), inserted
