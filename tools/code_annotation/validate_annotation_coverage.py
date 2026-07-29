"""注釈mapを基準に、行数ではなく意味blockの説明状態を検証する。"""

from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

from annotation_map_utils import load_annotation_map
from common import DOC_ROOT, ROOT, load_manifest


BLOCK_TYPES = (
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
)

PRIORITY_PATHS = {
    "A": [
        "examples/opd_trainer/run_multitask_qwen3.sh",
        "examples/opd_trainer/expected_multitask_config.yaml",
        "verl/trainer/main_opd.py",
        "verl/trainer/ppo/opd_ray_trainer.py",
        "verl/workers/fsdp_workers.py",
        "verl/workers/actor/dp_actor.py",
        "verl/trainer/ppo/core_algos.py",
        "verl/trainer/config/ppo_trainer.yaml",
    ],
    "B": [
        "examples/data_preprocess/prepare_sdar_multitask.py",
        "verl/trainer/main_ppo.py",
        "verl/utils/dataset/rl_dataset.py",
        "agent_system/environments/env_manager.py",
        "agent_system/environments/resume.py",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "agent_system/multi_turn_rollout/utils.py",
        "agent_system/multi_turn_rollout/async_rollout_core.py",
    ],
    "C": [
        "verl/single_controller/base/decorator.py",
        "verl/single_controller/ray/base.py",
        "verl/workers/sharding_manager/__init__.py",
        "verl/workers/sharding_manager/base.py",
        "verl/workers/sharding_manager/fsdp_sglang.py",
        "verl/workers/sharding_manager/fsdp_ulysses.py",
        "verl/workers/sharding_manager/fsdp_vllm.py",
        "verl/workers/sharding_manager/megatron_sglang.py",
        "verl/workers/sharding_manager/megatron_vllm.py",
        "verl/utils/gpu_profiler.py",
        "verl/trainer/ppo/metric_utils.py",
    ],
    "D": [
        "agent_system/environments/env_package/alfworld/envs.py",
        "agent_system/environments/env_package/search/envs.py",
        "agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py",
        "agent_system/environments/env_package/webshop/envs.py",
    ],
}

SELF_EXPLANATORY_REVIEW = {
    "verl/workers/sharding_manager/__init__.py":
        "公開symbolを持たないpackage markerで、実行blockがない。",
}


def _direct_annotation(ranges: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start <= annotation_start <= end for annotation_start, _ in ranges)


def _block_rows(path: str, language: str, ranges: list[tuple[int, int]]) -> list[dict]:
    source_lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    module_end = max(1, len(source_lines))
    module_explained = bool(ranges)
    rows = [{
        "original_path": path,
        "block_kind": "module",
        "block_name": path,
        "source_start_line": 1,
        "source_end_line": module_end,
        "coverage_status": "explained" if module_explained else "self_explanatory",
    }]
    if language != "python":
        return rows

    tree = ast.parse("\n".join(source_lines), filename=path)

    def visit(node: ast.AST, parent_covered: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, BLOCK_TYPES):
                start = int(child.lineno)
                end = int(getattr(child, "end_lineno", start))
                direct = _direct_annotation(ranges, start, end)
                if direct:
                    status = "explained"
                elif parent_covered:
                    status = "covered_by_parent_comment"
                else:
                    status = "self_explanatory"
                rows.append({
                    "original_path": path,
                    "block_kind": type(child).__name__,
                    "block_name": getattr(child, "name", f"{type(child).__name__}@{start}"),
                    "source_start_line": start,
                    "source_end_line": end,
                    "coverage_status": status,
                })
                visit(child, parent_covered or direct)
            else:
                visit(child, parent_covered)

    visit(tree, module_explained)
    return rows


def main() -> None:
    annotations = load_annotation_map()
    ranges_by_path: dict[str, list[tuple[int, int]]] = defaultdict(list)
    priorities_by_path: dict[str, set[str]] = defaultdict(set)
    for row in annotations:
        ranges_by_path[row["original_path"]].append(
            (int(row["source_start_line"]), int(row["source_end_line"]))
        )
        if row.get("priority"):
            priorities_by_path[row["original_path"]].add(row["priority"])

    target_rows = [
        row for row in load_manifest()
        if row["status"] == "completed" and row["language"] != "notebook"
    ]
    block_rows: list[dict] = []
    failures: list[str] = []
    for row in target_rows:
        try:
            block_rows.extend(
                _block_rows(
                    row["original_path"],
                    row["language"],
                    ranges_by_path[row["original_path"]],
                )
            )
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append(f"{row['original_path']}: {exc}")

    coverage_path = DOC_ROOT / "block_coverage.jsonl"
    coverage_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in block_rows),
        encoding="utf-8",
        newline="\n",
    )

    review_rows = []
    for priority, paths in PRIORITY_PATHS.items():
        for path in paths:
            if priority in priorities_by_path[path]:
                status = "reviewed"
                reason = "semantic annotation mapにpriority block解説あり"
            elif path in SELF_EXPLANATORY_REVIEW:
                status = "self_explanatory"
                reason = SELF_EXPLANATORY_REVIEW[path]
            else:
                status = "needs_review"
                reason = "priority block解説が未登録"
                failures.append(f"{priority}: {path}: {reason}")
            review_rows.append({
                "priority": priority,
                "original_path": path,
                "review_status": status,
                "reason": reason,
                "annotation_count": sum(
                    row.get("priority") == priority
                    for row in annotations
                    if row["original_path"] == path
                ),
            })

    (DOC_ROOT / "semantic_review.json").write_text(
        json.dumps(review_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    status_counts = Counter(row["coverage_status"] for row in block_rows)
    review_counts = Counter(row["review_status"] for row in review_rows)
    needs_review = status_counts.get("needs_review", 0) + review_counts.get("needs_review", 0)
    print(
        f"blocks={len(block_rows)} explained={status_counts.get('explained', 0)} "
        f"covered_by_parent={status_counts.get('covered_by_parent_comment', 0)} "
        f"self_explanatory={status_counts.get('self_explanatory', 0)} "
        f"needs_review={needs_review}"
    )
    for priority in PRIORITY_PATHS:
        expected = len(PRIORITY_PATHS[priority])
        reviewed = sum(
            row["priority"] == priority and row["review_status"] != "needs_review"
            for row in review_rows
        )
        print(f"priority_{priority}={reviewed}/{expected}")
    for failure in failures:
        print(f"FAIL {failure}")
    raise SystemExit(1 if failures or needs_review else 0)


if __name__ == "__main__":
    main()
