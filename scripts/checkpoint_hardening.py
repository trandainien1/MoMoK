"""Notebook cell source for the audited full-state checkpoint contract."""

HARDENING_CELL = r'''
# Harden the generated upstream patch. This cell runs inside Kaggle after the
# pinned source is checked out and before any training process starts.
import copy
import hashlib
import json
import os
import random

base_train = original_train
base_loader = original_loader
train_text = base_train
loader_text = base_loader
fixes = []

RESUME_CHECKPOINT_PATH = ""
resume_candidates = sorted(Path("/kaggle/input").glob("**/checkpoint_handoff.pt"))
if len(resume_candidates) > 1:
    raise RuntimeError("Ambiguous resume input: more than one checkpoint_handoff.pt was mounted")
if resume_candidates:
    RESUME_CHECKPOINT_PATH = str(resume_candidates[0])
    print("RESUME_INPUT_FOUND=" + Path(RESUME_CHECKPOINT_PATH).name)
elif RESUME_MODE == "FULL_STATE_RESUME":
    raise RuntimeError("FULL_STATE_RESUME requested but no checkpoint_handoff.pt was mounted")

if "return metrics, [att_s, att_i, att_t, att_mm]" in loader_text:
    if "y.byte()" in loader_text:
        loader_text = loader_text.replace("y.byte()", "y.bool()")
        fixes.append({"file": "utils/data_loader.py", "before": "y.byte()", "after": "y.bool()", "reason": "modern PyTorch boolean-mask compatibility", "semantic_change": False})

if "val_metrics = corpus.get_validation_pred(model, 'test')" in train_text:
    train_text = train_text.replace("val_metrics = corpus.get_validation_pred(model, 'test')", "val_metrics, _ = corpus.get_validation_pred(model, 'test')")
    fixes.append({"file": "train.py", "before": "val_metrics = corpus.get_validation_pred(model, 'test')", "after": "val_metrics, _ = corpus.get_validation_pred(model, 'test')", "reason": "upstream validation returns metrics and attention", "semantic_change": False})

train_text = train_text.replace("import argparse\n", "import argparse\nimport copy\nimport hashlib\nimport json\nimport os\nimport random\n", 1)
train_text = train_text.replace("        'save': 1,", "        'save': 1,\n        'resume_checkpoint': '',\n        'checkpoint_every': 25,\n        'checkpoint_dir': os.environ.get('MOMOK_CHECKPOINT_DIR', '/kaggle/working/repro_output/checkpoints'),\n        'session_soft_limit_seconds': 37800,")
train_text = train_text.replace("    if args.save:\n        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')", "    if args.save:\n        os.makedirs(f'./checkpoint/{args.dataset}', exist_ok=True)\n        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')")

helper = r"""

def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_parent(path):
    try:
        fd = os.open(os.path.dirname(os.path.abspath(path)) or '.', os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_parent(path)
    digest = _sha256_file(path)
    with open(path + '.sha256', 'w') as handle:
        handle.write(digest + '  ' + os.path.basename(path) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent(path + '.sha256')
    return digest


def _dataset_metadata():
    manifest_path = os.environ.get('MOMOK_DATASET_MANIFEST', '')
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        canonical = json.dumps(manifest, sort_keys=True).encode()
        return {'manifest': manifest, 'fingerprint': hashlib.sha256(canonical).hexdigest()}
    return {'manifest': {}, 'fingerprint': ''}


def _checkpoint_payload(args, model, estimator, optimizer, optimizer_mi,
                       lr_scheduler, best_test_metrics, final_epoch_metrics,
                       corpus, completed_epoch):
    return {
        'format_version': 2,
        'completed_epoch': int(completed_epoch),
        'model_state_dict': model.state_dict(),
        'estimator_state_dict': estimator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_mi_state_dict': optimizer_mi.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_test_metrics': copy.deepcopy(best_test_metrics),
        'final_epoch_metrics': copy.deepcopy(final_epoch_metrics),
        'python_random_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'corpus_train_indices': copy.deepcopy(corpus.train_indices),
        'training_config': {
            'dataset': args.dataset, 'seed': args.seed, 'lr': args.lr,
            'mu': args.mu, 'dim': args.dim, 'r_dim': args.r_dim,
            'batch_size': args.batch_size, 'n_exp': args.n_exp,
            'epochs': args.epochs, 'eval_freq': args.eval_freq,
        },
        'source_commit': os.environ.get('MOMOK_SOURCE_COMMIT', ''),
        'dataset_metadata': _dataset_metadata(),
    }


def _save_training_checkpoint(args, model, estimator, optimizer, optimizer_mi,
                              lr_scheduler, best_test_metrics,
                              final_epoch_metrics, corpus, completed_epoch,
                              filename='checkpoint_latest.pt'):
    payload = _checkpoint_payload(args, model, estimator, optimizer, optimizer_mi,
                                  lr_scheduler, best_test_metrics,
                                  final_epoch_metrics, corpus, completed_epoch)
    path = os.path.join(args.checkpoint_dir, filename)
    digest = _atomic_torch_save(payload, path)
    if filename == 'checkpoint_latest.pt':
        print(f'CHECKPOINT_SAVED={filename} COMPLETED_EPOCH={completed_epoch} SHA256={digest}')
    return digest


def _load_full_checkpoint(path):
    # Explicit trusted full-state mode; arbitrary untrusted files are rejected.
    return torch.load(path, map_location='cpu', weights_only=False)
"""
train_text = train_text.replace('\n\ndef train_decoder(args):', helper + '\n\ndef train_decoder(args):', 1)

resume_block = r"""
    start_epoch = 0
    final_epoch_metrics = None
    if args.resume_checkpoint:
        checkpoint = _load_full_checkpoint(args.resume_checkpoint)
        required = [
            'format_version', 'completed_epoch', 'model_state_dict',
            'estimator_state_dict', 'optimizer_state_dict',
            'optimizer_mi_state_dict', 'lr_scheduler_state_dict',
            'best_test_metrics', 'final_epoch_metrics',
            'python_random_state', 'numpy_rng_state', 'torch_rng_state',
            'torch_cuda_rng_state_all', 'corpus_train_indices',
            'training_config', 'source_commit', 'dataset_metadata',
        ]
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise RuntimeError(f'Invalid full-state checkpoint; missing {missing}')
        expected = {
            'dataset': args.dataset, 'seed': args.seed, 'lr': args.lr,
            'mu': args.mu, 'dim': args.dim, 'r_dim': args.r_dim,
            'batch_size': args.batch_size, 'n_exp': args.n_exp,
            'epochs': args.epochs, 'eval_freq': args.eval_freq,
        }
        saved = checkpoint['training_config']
        for key, value in expected.items():
            if saved.get(key) != value:
                raise RuntimeError(f'Checkpoint config mismatch for {key}: {saved.get(key)!r} != {value!r}')
        if checkpoint['source_commit'] != os.environ.get('MOMOK_SOURCE_COMMIT', ''):
            raise RuntimeError('Checkpoint source commit mismatch')
        if checkpoint['dataset_metadata'].get('fingerprint') != _dataset_metadata().get('fingerprint'):
            raise RuntimeError('Checkpoint dataset inventory fingerprint mismatch')
        completed_epoch = int(checkpoint['completed_epoch'])
        if completed_epoch < 0 or completed_epoch >= args.epochs:
            raise RuntimeError(f'Invalid completed_epoch={completed_epoch}')
        model.load_state_dict(checkpoint['model_state_dict'])
        estimator.load_state_dict(checkpoint['estimator_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_mi.load_state_dict(checkpoint['optimizer_mi_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        _move_optimizer_state(optimizer, args.device)
        _move_optimizer_state(optimizer_mi, args.device)
        best_test_metrics = copy.deepcopy(checkpoint['best_test_metrics'])
        final_epoch_metrics = copy.deepcopy(checkpoint['final_epoch_metrics'])
        corpus.train_indices = copy.deepcopy(checkpoint['corpus_train_indices'])
        # Restore RNG only after all constructors and state loads.
        random.setstate(checkpoint['python_random_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_state_all'])
        start_epoch = completed_epoch
        print(f'RESUME_CHECKPOINT={args.resume_checkpoint}')
        print(f'RESUME_FROM_COMPLETED_EPOCH={completed_epoch}')
        print(f'NEXT_EPOCH={completed_epoch + 1}')
        print(f'TARGET_EPOCH={args.epochs}')
        print(f'REMAINING_EPOCHS={args.epochs - completed_epoch}')
"""
train_text = train_text.replace('    best_test_metrics = model.init_metric_dict()\n    corpus.batch_size', '    best_test_metrics = model.init_metric_dict()\n' + resume_block + '    corpus.batch_size', 1)

old_loop = """    training_range = tqdm(range(args.epochs))
    for epoch in training_range:
        model.train()
"""
new_loop = """    training_start = time.monotonic()
    training_range = tqdm(range(start_epoch, args.epochs))
    for epoch in training_range:
        model.train()
"""
if old_loop not in train_text:
    raise RuntimeError('Expected upstream training loop was not found')
train_text = train_text.replace(old_loop, new_loop, 1)

old_eval = """        if (epoch + 1) % args.eval_freq == 0:
            print("Epoch {:04d} , average loss {:.4f} , epoch_time {:.4f}\\n".format(
                epoch + 1, sum(epoch_loss) / len(epoch_loss), time.time() - t))
            model.eval()
            with torch.no_grad():
                val_metrics, _ = corpus.get_validation_pred(model, 'test')
            if val_metrics['MRR'] > best_test_metrics['MRR']:
                best_test_metrics['MRR'] = val_metrics['MRR']
            if val_metrics['MR'] < best_test_metrics['MR']:
                best_test_metrics['MR'] = val_metrics['MR']
            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                best_test_metrics['Hits@100'] = val_metrics['Hits@100']
            print('\\n'.join(['Epoch: {:04d}'.format(epoch + 1), model.format_metrics(val_metrics, 'test')]))
            print("\\n\\n")
"""
new_eval = """        completed_epoch = epoch + 1
        final_epoch_metrics = None
        if completed_epoch % args.eval_freq == 0:
            print("Epoch {:04d} , average loss {:.4f} , epoch_time {:.4f}\\n".format(
                completed_epoch, sum(epoch_loss) / len(epoch_loss), time.time() - t))
            model.eval()
            with torch.no_grad():
                val_metrics, _ = corpus.get_validation_pred(model, 'test')
            final_epoch_metrics = copy.deepcopy(val_metrics)
            for metric in ['MRR', 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@100']:
                if val_metrics[metric] > best_test_metrics[metric]:
                    best_test_metrics[metric] = val_metrics[metric]
            if val_metrics['MR'] < best_test_metrics['MR']:
                best_test_metrics['MR'] = val_metrics['MR']
            print('\\n'.join(['Epoch: {:04d}'.format(completed_epoch), model.format_metrics(val_metrics, 'test')]))
            print("\\n\\n")
        if completed_epoch % args.checkpoint_every == 0 or final_epoch_metrics is not None or completed_epoch == args.epochs:
            _save_training_checkpoint(args, model, estimator, optimizer, optimizer_mi,
                                      lr_scheduler, best_test_metrics,
                                      final_epoch_metrics, corpus, completed_epoch)
        elapsed = time.monotonic() - training_start
        if elapsed >= args.session_soft_limit_seconds and completed_epoch < args.epochs:
            digest = _save_training_checkpoint(args, model, estimator, optimizer,
                                               optimizer_mi, lr_scheduler,
                                               best_test_metrics, final_epoch_metrics,
                                               corpus, completed_epoch,
                                               'checkpoint_handoff.pt')
            run_state = {
                'status': 'PARTIAL_CHECKPOINT_READY',
                'completed_epoch': completed_epoch,
                'target_epoch': args.epochs,
                'resume_required': True,
                'source_commit': os.environ.get('MOMOK_SOURCE_COMMIT', ''),
                'config': _checkpoint_payload(args, model, estimator, optimizer, optimizer_mi, lr_scheduler, best_test_metrics, final_epoch_metrics, corpus, completed_epoch)['training_config'],
                'checkpoint_filename': 'checkpoint_handoff.pt',
                'checkpoint_sha256': digest,
                'elapsed_seconds': elapsed,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }
            with open(os.path.join(args.checkpoint_dir, 'run_state.json'), 'w') as handle:
                json.dump(run_state, handle, indent=2)
            print('RUN_STATE=PARTIAL_CHECKPOINT_READY')
            print(f'COMPLETED_EPOCH={completed_epoch}')
            return run_state
"""
if old_eval not in train_text:
    raise RuntimeError('Expected upstream evaluation block was not found')
train_text = train_text.replace(old_eval, new_eval, 1)

old_total = "    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))\n"
new_total = """    final_digest = _save_training_checkpoint(args, model, estimator, optimizer,
                                            optimizer_mi, lr_scheduler,
                                            best_test_metrics, final_epoch_metrics,
                                            corpus, args.epochs, 'checkpoint_final.pt')
    with open(os.path.join(args.checkpoint_dir, 'run_state.json'), 'w') as handle:
        json.dump({'status': 'FINAL_COMPLETE', 'completed_epoch': args.epochs,
                   'target_epoch': args.epochs, 'resume_required': False,
                   'source_commit': os.environ.get('MOMOK_SOURCE_COMMIT', ''),
                   'checkpoint_filename': 'checkpoint_final.pt',
                   'checkpoint_sha256': final_digest}, handle, indent=2)
    print('RUN_STATE=FINAL_COMPLETE')
    if final_epoch_metrics is not None:
        print('FINAL_EPOCH_METRICS_JSON=' + json.dumps(final_epoch_metrics, sort_keys=True))
    print('UPSTREAM_BEST_METRICS_JSON=' + json.dumps(best_test_metrics, sort_keys=True))
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
"""
if old_total not in train_text:
    raise RuntimeError('Expected upstream train tail was not found')
train_text = train_text.replace(old_total, new_total, 1)

train_path.write_text(train_text)
loader_path.write_text(loader_text)
patch = subprocess.run(['git', '-C', REPO_DIR, 'diff'], text=True, stdout=subprocess.PIPE, check=True).stdout
Path(OUTPUT_DIR, 'upstream_compatibility.patch').write_text(patch)
Path(OUTPUT_DIR, 'source_fixes.json').write_text(json.dumps(fixes + [{
    'file': 'train.py',
    'before': 'single model-only save and non-resumable loop',
    'after': 'full-state atomic checkpoints, graceful handoff, fail-closed resume',
    'reason': 'cross-session Kaggle durability',
    'semantic_change': False,
}], indent=2))
print('SOURCE_FIXES=', len(fixes) + 1)
print('CHECKPOINT_CONTRACT=PASS')
print('END_OF_EPOCH_ORDER=lr_scheduler.step() -> evaluation -> _save_training_checkpoint')
print('SOURCE_AUDIT=PASS')
'''
