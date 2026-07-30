# Detailed Japanese Comment Reference

This directory preserves the three user-provided Python files as exact gzip+Base64 payloads.
They demonstrate the preferred annotation style: explanations are placed next to the relevant code, use concrete identifiers, and describe execution flow, data shapes, task routing, and side effects without `[EXPLAIN]` tags.

## Included files

- `ray_trainer.py`
- `main_ppo.py`
- `skillsd_ray_trainer.py`

The payload is split only to keep GitHub API writes reliable. Run:

```bash
python docs/code_explanation/reference_examples/extract_reference_examples.py
```

This materializes the files under:

```text
docs/code_explanation/reference_examples/materialized/
```

To replace the current annotated mirror with these exact files, run:

```bash
python docs/code_explanation/reference_examples/extract_reference_examples.py --install
```

The extractor verifies SHA-256 before writing. The uploaded files were also checked with `python -m py_compile` before publication.
