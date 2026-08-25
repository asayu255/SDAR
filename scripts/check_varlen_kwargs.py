#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Can this transformers take cu_seqlens directly, instead of syncing per layer?

With ``use_remove_padding`` the actor hands the model ``position_ids`` and lets
HF's flash-attention path work out the packed-sequence boundaries. That path
decides on the device and reads the answer on the host -- whether the sequences
are packed at all, and what the longest one is, both of which flash-attn needs
as Python ints. It is a device-to-host sync, and it happens once per layer per
forward: 28 layers, doubled by gradient checkpointing recomputing the forward
inside the backward. The trace measures ~80 D2H copies per micro-batch, each
followed by ~147 us with the device empty while the host is back in Python --
0.38% of wall, about a third of the whole ambient deficit.

The values are already computed. ``unpad_input`` returns cu_seqlens and
max_seqlen_in_batch once per micro-batch, and _flash_attention_forward will skip
the whole position_ids path if handed them explicitly. verl's own monkey_patch
has the TODO saying so. What is version-dependent is whether the kwargs survive
the trip from the model's forward down to the attention module, which is what
this checks -- run it with the environment's interpreter, not the system one:

    /path/to/envs/<env>/bin/python3 scripts/check_varlen_kwargs.py
"""

import inspect
import sys


def _report(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'NO'}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    try:
        import transformers
    except ImportError:
        print("transformers not importable: run this with the env's python3, "
              "not the system one")
        return 2

    print(f"transformers {transformers.__version__}, python {sys.version.split()[0]}\n")
    from transformers.modeling_flash_attention_utils import _flash_attention_forward as fa

    params = inspect.signature(fa).parameters
    wanted = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
    accepted = [name for name in wanted if name in params]
    entry = _report("_flash_attention_forward accepts the varlen kwargs",
                    len(accepted) == len(wanted),
                    f"has {accepted or 'none'}")

    # The sync itself, so the diagnosis is confirmed on THIS version rather than
    # assumed from another one. Both spellings are a host-side branch on a device
    # tensor; only the wording changed between releases.
    src = inspect.getsource(fa)
    syncing = "_is_packed_sequence" in src or "torch.diff(position_ids" in src
    _report("the position_ids path branches on a device value (the sync)", syncing,
            "_is_packed_sequence" if "_is_packed_sequence" in src else "torch.diff(...).all()")

    # And whether the kwargs can actually reach the attention module.
    chain = []
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3

        for name in ("Qwen3ForCausalLM", "Qwen3Model", "Qwen3DecoderLayer", "Qwen3Attention"):
            cls = getattr(qwen3, name, None)
            if cls is None:
                chain.append((name, False, "class not found"))
                continue
            body = inspect.getsource(cls.forward)
            chain.append((name, "**kwargs" in body, "forwards **kwargs" if "**kwargs" in body
                          else "does NOT forward **kwargs"))
    except ImportError as exc:
        chain.append(("qwen3", False, repr(exc)))

    through = all(ok for _, ok, _ in chain)
    for name, ok, detail in chain:
        _report(f"{name}.forward", ok, detail)

    print()
    if entry and through:
        print("=> The fix is available: pass cu_seq_lens_q/k and max_length_q/k from")
        print("   _forward_micro_batch (unpad_input already returns them) and the")
        print("   per-layer sync disappears.")
        return 0
    if not entry:
        print("=> This version cannot take them at the entry point; the fix would have")
        print("   to patch _flash_attention_forward itself rather than pass kwargs.")
    elif not through:
        broken = [name for name, ok, _ in chain if not ok]
        print(f"=> The entry point accepts them but {', '.join(broken)} drops kwargs on")
        print("   the way down, so they would not arrive. Patching that forward is the")
        print("   smaller change than reimplementing the attention path.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
