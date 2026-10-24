import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ablations.runtime_contract import estimator_should_run
from ablations.torch_checkpoint import build_payload, checkpoint_config, restore_checkpoint, save_checkpoint
from ablations.variants import resolve_variant
from models.RelMoE import RelMoE
from models.model import *
from models.modules import MIEstimator
from utils.data_loader import *
from utils.data_util import load_data


def parse_args():
    config_args = {
        'lr': 0.0005,
        'dropout_gat': 0.3,
        'dropout': 0.3,
        'cuda': 0,
        'epochs_gat': 3000,
        'epochs': 2000,
        'weight_decay_gat': 1e-5,
        'weight_decay': 0,
        'seed': 10010,
        'model': 'RMoE',
        'num-layers': 3,
        'dim': 256,
        'r_dim': 256,
        'k_w': 10,
        'k_h': 20,
        'n_heads': 2,
        'dataset': 'DB15K',
        'pre_trained': 0,
        'encoder': 0,
        'image_features': 1,
        'text_features': 1,
        'patience': 5,
        'eval_freq': 100,
        'lr_reduce_freq': 500,
        'gamma': 1.0,
        'bias': 1,
        'neg_num': 2,
        'neg_num_gat': 2,
        'alpha': 0.2,
        'alpha_gat': 0.2,
        'out_channels': 32,
        'kernel_size': 3,
        'batch_size': 1024,
        'save': 1,
        'n_exp': 3,
        'mu': 0.0001,
        'img_dim': 256,
        'txt_dim': 256,
        'variant': 'full',
        'max_train_batches': 0,
        'resume_checkpoint': '',
        'output_dir': '',
        'run_mode': 'full',
        'metric_policy': 'legacy',
        'wandb_mode': 'disabled'
        , 'checkpoint_dir': ''
        , 'checkpoint_every': 25
        , 'wandb_run_id': ''
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val, type=type(val))
    args = parser.parse_args()
    return args

args = parse_args()
args.variant_config = resolve_variant(args.variant, n_exp=args.n_exp, mu=args.mu)
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(args.dataset)
print("Training data {:04d}".format(len(train_data[0])))

corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id)

if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)
args.entity2id = entity2id
args.relation2id = relation2id

model_name = {
    'RMoE': RelMoE
}
time.sleep(5)



def train_decoder(args):
    model = model_name[args.model](args)
    args.img_dim = model.img_dim
    args.txt_dim = model.txt_dim
    estimator = MIEstimator(args) if estimator_should_run(args.variant_config) else None
    print(str(model))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    optimizer_mi = (
        torch.optim.Adam(params=estimator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if estimator is not None
        else None
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)
        if estimator is not None:
            estimator = estimator.to(args.device)

    # Train Model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = model.init_metric_dict()
    start_epoch = 0
    if args.resume_checkpoint:
        payload = restore_checkpoint(
            args.resume_checkpoint,
            args=args,
            model=model,
            estimator=estimator,
            optimizer=optimizer,
            optimizer_mi=optimizer_mi,
            lr_scheduler=lr_scheduler,
            corpus=corpus,
        )
        best_test_metrics = payload["best_validation_metrics"]
        start_epoch = int(payload["completed_epoch"])
        print(f"RESUME_OK={args.resume_checkpoint}")
        print(f"RESUME_FROM_COMPLETED_EPOCH={start_epoch}")
        print(f"NEXT_EPOCH={start_epoch + 1}")
    checkpoint_dir = args.checkpoint_dir or args.output_dir
    final_epoch_metrics = None
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num
    training_range = tqdm(range(start_epoch, args.epochs))
    for epoch in training_range:
        model.train()
        epoch_loss = []
        epoch_mi_loss = []
        final_epoch_metrics = None
        t = time.time()
        corpus.shuffle()
        batch_limit = corpus.max_batch_num
        if args.max_train_batches:
            batch_limit = min(batch_limit, args.max_train_batches)
        for batch_num in range(batch_limit):
            # Training the KGC model
            if estimator is not None:
                estimator.eval()
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)
            output, embeddings = model.forward(train_indices)
            loss = model.loss_func(output, train_values)
            if estimator is not None:
                loss = loss + args.mu * estimator(embeddings)
            loss.backward()
            optimizer.step()
            # Train the estimator
            if estimator is not None:
                estimator.train()
                optimizer_mi.zero_grad()
                with torch.no_grad():
                    embeddings = model.get_batch_embeddings(train_indices)
                estimator_loss = estimator.train_estimator(embeddings)
                estimator_loss.backward()
                optimizer_mi.step()
            else:
                estimator_loss = torch.zeros((), device=train_indices.device)
            epoch_loss.append(loss.data.item())
            epoch_mi_loss.append(estimator_loss.item())
        training_range.set_postfix(loss="main: {:.5} mi: {:.5}".format(sum(epoch_loss), sum(epoch_mi_loss)))
        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            print("Epoch {:04d} , average loss {:.4f} , epoch_time {:.4f}\n".format(
                epoch + 1, sum(epoch_loss) / len(epoch_loss), time.time() - t))
            model.eval()
            with torch.no_grad():
                val_metrics, _ = corpus.get_validation_pred(model, 'test')
            final_epoch_metrics = dict(val_metrics)
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
            print('\n'.join(['Epoch: {:04d}'.format(epoch + 1), model.format_metrics(val_metrics, 'test')]))
            print("\n\n")

        completed_epoch = epoch + 1
        if checkpoint_dir and (
            completed_epoch % args.checkpoint_every == 0
            or final_epoch_metrics is not None
            or completed_epoch == args.epochs
        ):
            payload = build_payload(
                args,
                model,
                estimator,
                optimizer,
                optimizer_mi,
                lr_scheduler,
                best_test_metrics,
                corpus,
                completed_epoch,
                final_epoch_metrics,
            )
            digest = save_checkpoint(Path(checkpoint_dir) / "checkpoint_latest.pt", payload)
            print(f"CHECKPOINT_SAVED=checkpoint_latest.pt COMPLETED_EPOCH={completed_epoch} SHA256={digest}")


    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        if estimator is not None:
            estimator.eval()
        with torch.no_grad():
            best_test_metrics, _ = corpus.get_validation_pred(model, 'test')
    print('\n'.join(['Val set results:', model.format_metrics(best_val_metrics, 'val')]))
    print('\n'.join(['Test set results:', model.format_metrics(best_test_metrics, 'test')]))
    print("\n\n\n\n\n\n")

    if args.save:
        if checkpoint_dir:
            payload = build_payload(
                args,
                model,
                estimator,
                optimizer,
                optimizer_mi,
                lr_scheduler,
                best_test_metrics,
                corpus,
                args.epochs,
                final_epoch_metrics,
            )
            digest = save_checkpoint(Path(checkpoint_dir) / "checkpoint_final.pt", payload)
            print(f"CHECKPOINT_FINAL_SHA256={digest}")
        torch.save(model.state_dict(), f'./checkpoint/{args.dataset}/{args.model}.pth')
        print('Saved model!')


if __name__ == '__main__':
    train_decoder(args)
