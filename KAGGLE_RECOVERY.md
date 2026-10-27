# Kaggle Ablation Recovery Contract

This document defines the recovery procedure for future ablation runs. It is
not evidence that a Kaggle recovery smoke test has already passed.

## Output contract

Each run must use a unique namespace:

```text
/kaggle/working/momok_ablation/<dataset>/<variant>/<seed>/<config_hash>/
```

The namespace must contain:

```text
config.json
manifest.json
environment.json
dataset_manifest.json
logs/session_*.log
metrics/train.jsonl
metrics/eval.jsonl
checkpoints/checkpoint_latest.pt
checkpoints/checkpoint_best.pt
checkpoints/checkpoint_final.pt
reports/smoke_summary.json
reports/result_summary.json
reports/failure.json
reports/artifact_integrity.json
exports/resume_bundle.zip
```

## Checkpoint requirements

Checkpoints must be written atomically in the same filesystem and include
the model, optimizers, scheduler, RNG state, corpus order, variant ID,
configuration fingerprint, source commit, dataset fingerprint, and completed
epoch. Estimator state is required only when ExID is active.

The existing main-result hardening implementation is kept unchanged. The
ablation runtime must use the isolated schema in
`ablations/checkpoint_contract.py` and must not overwrite the main-result
checkpoints.

## Recovery sequence

1. Let the current Kaggle session save a checkpoint and checksum.
2. Download the output with the authenticated Kaggle CLI, for example:

   ```bash
   kaggle kernels output <owner>/<kernel-slug> -p <local-output-dir> -o
   ```

3. Verify the checkpoint checksum and `manifest.json` before reuse.
4. Publish the verified bundle as a private Kaggle Dataset version if the
   next notebook session needs `/kaggle/input` storage.
5. Attach that Dataset to the next notebook version.
6. Restore from the mounted read-only checkpoint into a new
   `/kaggle/working` output namespace.
7. Resume at `completed_epoch + 1`; never restart silently.
8. Record `RESUME_OK`, source checksum, config hash, and resumed epoch.

The exact Kaggle UI persistence behavior remains environment-dependent and
requires a real smoke test before any full run is authorized.

## Mandatory recovery gates

- checksum verification;
- dataset/source/config/variant compatibility;
- optimizer and scheduler restoration;
- Python/NumPy/Torch/CUDA RNG restoration;
- corpus ordering restoration;
- no epoch replay or skip;
- failure traceback persisted to `failure.json`;
- output isolation from all Full Model artifacts.
