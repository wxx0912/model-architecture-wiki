# Config fixtures

These small JSON files are synthetic test inputs shaped like common model configuration fields. They test parsing and checklist generation only; they are not downloaded checkpoint configs, do not pin a model release, and must not be cited as architecture evidence.

The state-dict fixtures are synthetic key manifests/indexes. They test parameter-name observation only; they do not contain weights, tensor shapes, source revisions, or execution-graph evidence. A JSON list is treated as incomplete unless wrapped as `{"keys": [...], "complete": true}`; the HF `weight_map` fixture is treated as complete by default.

The forward fixture is a synthetic observation manifest. It tests recording shape/dtype/layout only and is not evidence from a real model run.

Real pinned configuration snapshots are under `tests/fixtures/real/`; their source URL and repository commit are recorded in that directory README.
