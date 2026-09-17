"""Let torch 2.8 read a DeviceMesh that torch 2.7 pickled.

THE BREAK. An FSDP shard saved by this repo carries DTensors, and a DTensor
carries its DeviceMesh by value. torch 2.7 stored the per-dimension process
groups on the mesh as ``_dim_group_infos`` -- a list of
``(tag, ranks, group_name)`` -- and 2.8 renamed that to ``_dim_group_names``,
holding just the names. Unpickling does not run ``__init__``, so a mesh from a
2.7 checkpoint arrives in a 2.8 process with the old attribute and none of the
new one, and the first collective on it dies inside torch itself:

    torch/distributed/_functional_collectives.py, _resolve_group_name
    return dmesh._dim_group_names[dim]
    AttributeError: 'DeviceMesh' object has no attribute '_dim_group_names'

WHAT THIS DOES. Gives ``DeviceMesh`` a fallback for the new name that reads the
old one: the group name is the third element of each info tuple, so nothing is
invented or guessed -- the value 2.8 wants is already in the file. Set as a
class-level ``__getattr__`` so it fires only when the instance really lacks the
attribute; a mesh built normally by 2.8 has it and never reaches here.

A no-op on any torch whose DeviceMesh already uses ``_dim_group_infos``, and on
one that stores neither (then the original AttributeError is raised, unchanged).
"""


def install() -> str:
    try:
        from torch.distributed.device_mesh import DeviceMesh
    except Exception as exc:                      # no distributed build
        return f"skipped: {type(exc).__name__}"

    if getattr(DeviceMesh, "_dim_group_names_compat", False):
        return "already installed"

    prev = getattr(DeviceMesh, "__getattr__", None)

    def __getattr__(self, name):                  # only for missing attributes
        if name == "_dim_group_names":
            infos = self.__dict__.get("_dim_group_infos")
            if infos is not None:
                # (tag, ranks, group_name) -> group_name
                names = [i[2] if isinstance(i, (tuple, list)) and len(i) >= 3 else i
                         for i in infos]
                self.__dict__["_dim_group_names"] = names
                return names
        if prev is not None:
            return prev(self, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    DeviceMesh.__getattr__ = __getattr__
    DeviceMesh._dim_group_names_compat = True
    return "installed"
