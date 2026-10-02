# Tracked local FLUX runtime helper

`local_runtime_fast.py` is byte-identical to the exact-weight CPU-offload helper used in the pretrained development run. Its root lookup is `Path(__file__).resolve().parents[2]`; this tracked two-level location preserves that contract. Portable runners add this folder to `sys.path`.

The original upstream model/source revisions are constants in the helper and the experiment source lock. Model weights and encoders are fetched into ignored `models/flux-action/`, and the official pinned Python implementation is installed from ignored `runtime/flux-action-src/`. The `README-original.md` preserves the earlier runtime notes; its old root-local paths are historical rather than portable invocation instructions. Use `docs/flux-so101-retry-reproduction.md` for current commands.

The slow helper remains for the retained comparison: the fast helper returned bit-identical commands for one paired raw-coordinate diagnostic. That diagnostic establishes runtime equivalence only, not task success. Both full contact-based pretrained development trials subsequently failed placement.
