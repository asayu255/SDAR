# [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
"""
RLSD (Reinforcement Learning with Self-Distillation) utilities.

This module provides skill-based privileged information loading and
token-level advantage computation for the RLSD algorithm.
All task types and keyword mappings are loaded from skill_mapping.json,
making this module environment-agnostic.
"""

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import json
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import os
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from typing import Dict, List, Optional

# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
import torch
# [EXPLAIN] このモジュールで使用する型・設定・分散処理または Tensor 操作の依存関係を読み込む。
from omegaconf import OmegaConf


# [EXPLAIN] `load_skill_mapping` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_skill_mapping(skills_dir: str) -> dict:
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    mapping_path = os.path.join(skills_dir, "skill_mapping.json")
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with open(mapping_path, "r") as f:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return json.load(f)


# [EXPLAIN] `load_skill_content` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def load_skill_content(skills_dir: str, skill_mapping: dict) -> Dict[str, str]:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Load all skill markdown files into a dict keyed by skill name."""
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    contents = {}
    # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
    for skill_name, filename in skill_mapping["skill_files"].items():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        filepath = os.path.join(skills_dir, filename)
        # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
        with open(filepath, "r") as f:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            contents[skill_name] = f.read().strip()
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return contents


# [EXPLAIN] `_normalize_task_name` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _normalize_task_name(task_name: Optional[str]) -> Optional[str]:
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if task_name is None:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return None
    # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
    task_name = str(task_name).lower()
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "alfworld" in task_name:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "alfworld"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "webshop" in task_name:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "webshop"
    # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
    if "search" in task_name:
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "search"
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return task_name


# [EXPLAIN] `_to_plain_dict` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def _to_plain_dict(value) -> dict:
    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else dict(value)


# [EXPLAIN] `SkillProvider` として状態と関連処理をまとめ、worker・trainer・dataset などの責務境界を定義する。
class SkillProvider:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """Loads and caches skill files, provides privileged info per task type.

    Everything is driven by skill_mapping.json:
      - skill_files: maps skill names to markdown filenames
      - task_to_skill: maps task type strings to skill names
      - task_keywords (optional): ordered list of (task_type -> keywords) for
        inferring task type from prompt text. Keywords are matched in order;
        the first task type whose ALL keywords appear in the text wins.
    """

    # [EXPLAIN] `__init__` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def __init__(self, skills_dir: str, skill_all: bool = False, skills_dirs: Optional[Dict[str, str]] = None):
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skills_dir = skills_dir
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skill_all = skill_all

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.task_skill_providers = {}
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.default_provider = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if skills_dirs:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            skills_dirs = _to_plain_dict(skills_dirs)
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.task_skill_providers = {
                _normalize_task_name(task_name): SkillProvider(skills_dir=task_skills_dir, skill_all=skill_all)
                for task_name, task_skills_dir in skills_dirs.items()
            }
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.default_provider = next(iter(self.task_skill_providers.values()))
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.skill_mapping = self.default_provider.skill_mapping
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.skill_contents = self.default_provider.skill_contents
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.task_to_skill = self.default_provider.task_to_skill
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self.task_keywords = self.default_provider.task_keywords
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skill_mapping = load_skill_mapping(skills_dir)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.skill_contents = load_skill_content(skills_dir, self.skill_mapping)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.task_to_skill = self.skill_mapping["task_to_skill"]
        # task_keywords: ordered dict of task_type -> list of keywords (all must match)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        self.task_keywords: Dict[str, List[str]] = self.skill_mapping.get("task_keywords", {})

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.skill_all:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            self._all_skills_text = self._build_all_skills_text()

    # [EXPLAIN] `_provider_for_task` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _provider_for_task(self, task_name: Optional[str]):
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not self.task_skill_providers:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        normalized = _normalize_task_name(task_name)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.task_skill_providers.get(normalized, self.default_provider)

    # [EXPLAIN] `_build_all_skills_text` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _build_all_skills_text(self) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Concatenate general_skills + all task-specific skills."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        general = self.skill_contents.get("general_skills", "")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parts = [general]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for skill_name, content in self.skill_contents.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if skill_name != "general_skills":
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                parts.append(content)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "\n\n".join(parts)

    # [EXPLAIN] `_get_skill_text` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def _get_skill_text(self, task_type: Optional[str]) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Assemble general_skills + task-specific skill text."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        general = self.skill_contents.get("general_skills", "")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parts = [general]
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if task_type:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mapped_name = self.task_to_skill.get(task_type)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mapped_name and mapped_name in self.skill_contents:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                parts.append(self.skill_contents[mapped_name])
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "\n\n".join(parts)

    # [EXPLAIN] `get_privileged_info` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_privileged_info(self, gamefile: str) -> str:
        # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
        """Return skills for a gamefile path (task type appears as substring)."""
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.task_skill_providers:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.default_provider.get_privileged_info(gamefile)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.skill_all:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._all_skills_text
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        matched_task = None
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task_type in self.task_to_skill:
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if task_type in gamefile:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                matched_task = task_type
                # [EXPLAIN] 現在の分岐または反復の制御を明示する。
                break
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_skill_text(matched_task)

    # [EXPLAIN] `get_privileged_info_from_prompt` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_privileged_info_from_prompt(self, prompt_text: str) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Infer task type from prompt text using keyword rules from skill_mapping.json.

        Uses ``any`` matching: a task type is matched if ANY of its keywords
        appear in the prompt.  When multiple task types match, all of their
        skill texts are concatenated (general_skills is included only once).
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.task_skill_providers:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.default_provider.get_privileged_info_from_prompt(prompt_text)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.skill_all:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._all_skills_text
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        text_lower = prompt_text.lower()
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        matched_tasks = []
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task_type, keywords in self.task_keywords.items():
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if keywords and any(kw in text_lower for kw in keywords):
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                matched_tasks.append(task_type)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if not matched_tasks:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._get_skill_text(None)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        general = self.skill_contents.get("general_skills", "")
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        parts = [general]
        # [EXPLAIN] batch、micro-batch、task または token の要素を順に処理する。
        for task_type in matched_tasks:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            mapped_name = self.task_to_skill.get(task_type)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if mapped_name and mapped_name in self.skill_contents:
                # [EXPLAIN] 必要な引数と現在の状態を渡して処理を呼び出し、戻り値またはbatch への副作用を次の段階へ接続する。
                parts.append(self.skill_contents[mapped_name])
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return "\n\n".join(parts)

    # [EXPLAIN] `get_privileged_info_from_data_source` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_privileged_info_from_data_source(self, data_source: str, prompt_text: str) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Infer task type using data_source field and prompt content.

        Matching rules:
          - data_source == 'popqa' -> entity_attribute_lookup
          - data_source in ('nq', 'triviaqa') -> direct_retrieval
          - prompt contains 'which' + 'or' (without 'for') -> compare
          - data_source == 'hotpotqa' -> multi_hop_reasoning
          - otherwise -> general skills only (unknown)
        """
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.task_skill_providers:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.default_provider.get_privileged_info_from_data_source(data_source, prompt_text)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if self.skill_all:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self._all_skills_text

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        task_type = None
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if data_source == "popqa":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_type = "entity_attribute_lookup"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif data_source in ("nq", "triviaqa"):
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_type = "direct_retrieval"
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        elif data_source == "hotpotqa":
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            task_type = "multi_hop_reasoning"
        # [EXPLAIN] 直前の条件が成立しない場合の代替経路を実行する。
        else:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            text_lower = prompt_text.lower()
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if "which" in text_lower and "or" in text_lower and "for" not in text_lower:
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                task_type = "compare"
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            elif data_source in ("2wikimultihopqa", "musique", "bamboogle"):
                # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
                task_type = "multi_hop_reasoning"

        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self._get_skill_text(task_type)

    # [EXPLAIN] `get_privileged_info_for_sample` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
    def get_privileged_info_for_sample(
        self,
        task_name: Optional[str] = None,
        gamefile: Optional[str] = None,
        data_source: Optional[str] = None,
        prompt_text: str = "",
    ) -> str:
        # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
        """Return skill text, routing by task_name when multitask skills are configured."""
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        provider = self._provider_for_task(task_name)
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if provider is not self:
            # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
            normalized_task = _normalize_task_name(task_name)
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if gamefile is not None:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return provider.get_privileged_info(str(gamefile))
            # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
            if normalized_task == "search" and data_source is not None:
                # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
                return provider.get_privileged_info_from_data_source(str(data_source), prompt_text)
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return provider.get_privileged_info_from_prompt(prompt_text)

        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if gamefile is not None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.get_privileged_info(str(gamefile))
        # [EXPLAIN] 実行時の設定または状態を評価し、成立する経路だけを選択する。
        if data_source is not None:
            # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
            return self.get_privileged_info_from_data_source(str(data_source), prompt_text)
        # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
        return self.get_privileged_info_from_prompt(prompt_text)


# [EXPLAIN] `compute_rlsd_token_advantage` の入力を検証・変換し、呼び出し元が使用する結果または副作用を生成する処理単位を定義する。
def compute_rlsd_token_advantage(
    seq_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    rlsd_lambda: float = 0.5,
    rlsd_clip_eps: float = 0.2,
) -> torch.Tensor:
    # [EXPLAIN] この論理行で現在の処理ブロックに必要な状態または制御を定義する。
    """
    Compute token-level RLSD advantages.

    Args:
        seq_advantages: (bs, response_length) — standard GRPO sequence-level advantage
            broadcast to all tokens (same value per sequence).
        student_log_probs: (bs, response_length) — log π_θ(y_t | x, y_<t).
        teacher_log_probs: (bs, response_length) — log π_θ(y_t | x, r, y_<t).
        response_mask: (bs, response_length) — mask for valid response tokens.
        rlsd_lambda: mixing coefficient λ. When 0, degrades to standard GRPO.
        rlsd_clip_eps: clipping bound ε_w for token weights.

    Returns:
        token_advantages: (bs, response_length) — token-level advantage Â_t.
    """
    # [EXPLAIN] context manager で resource、autocast、no_grad または session の寿命を限定する。
    with torch.no_grad():
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        first_valid = response_mask.int().argmax(dim=-1)  # (bs,)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        batch_indices = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        A_seq = seq_advantages[batch_indices, first_valid]  # (bs,)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        delta_t = teacher_log_probs - student_log_probs  # (bs, response_length)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        sign_A = torch.sign(A_seq).unsqueeze(-1)  # (bs, 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        w_t = torch.exp(sign_A * delta_t)  # (bs, response_length)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        w_t = torch.clamp(w_t, 1.0 - rlsd_clip_eps, 1.0 + rlsd_clip_eps)

        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        A_seq_expanded = A_seq.unsqueeze(-1)  # (bs, 1)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_advantages = A_seq_expanded * ((1.0 - rlsd_lambda) + rlsd_lambda * w_t)
        # [EXPLAIN] 後続の計算・routing・mask・metric で参照する値を構築し、現在のスコープまたは batch に保持する。
        token_advantages = token_advantages * response_mask

    # [EXPLAIN] 計算済みの Tensor、metric、batch または状態を呼び出し元へ返す。
    return token_advantages
