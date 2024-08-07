#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
import ot
import copy
from tqdm import tqdm
from multiprocessing import Pool

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from models.gaussian_models import GaussianMLP, GaussianGCN, GaussianSheafDiffusion, SampledGaussianSheafDiffusion
from utils.heterophilic import get_dataset, get_fixed_splits
from torch.optim.lr_scheduler import ReduceLROnPlateau


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    loss = 0
    y = data.y[data.train_mask]
    a = torch.ones(out[0].size(0), device=device) / out[0].size(0)
    b = torch.ones(y[0].size(0), device=device) / y[0].size(0)
    for i in range(out.size(0)):
        M = ot.dist(out[i][:, None], y[i][:, None])
        wloss = ot.emd2(a, b, M)
        loss = loss + wloss
    #print(loss.item())
    loss.backward()
    optimizer.step()
    del out

def test(model, data, device):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask]

            loss = 0
            distances = []
            y = data.y[mask]
            a = torch.ones(pred[0].size(0), device=device) / pred[0].size(0)
            b = torch.ones(y[0].size(0), device=device) / y[0].size(0)
            for i in range(pred.size(0)):
                M = ot.dist(pred[i][None].t(), y[i][None].t())
                wloss = torch.sqrt(ot.emd2(a, b, M))
                loss += wloss
                distances.append(wloss.item())
            distances = torch.tensor(distances)
            mean = distances.mean().item()
            std = distances.std().item()
            acc = (mean, std)

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


def run_exp(args, dataset, model_class, fold):
    data = dataset
    #data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])

    model = model_class(data.edge_index, args)
    model = model.to(args['device'])

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-5,
                              patience=args['lr_decay_patience'])

    epoch = 0
    best_val_acc = test_acc = [float('inf')]
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0
    best_model = None

    for epoch in range(args['epochs']):
        train(model, optimizer, data, args['device'])

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, args['device'])
        # if fold == 0:
        #     res_dict = {
        #         f'fold{fold}_train_mean': torch.tensor(train_acc[0]),
        #         f'fold{fold}_train_loss': train_loss,
        #         f'fold{fold}_val_mean': torch.tensor(val_acc[0]),
        #         f'fold{fold}_val_loss': val_loss,
        #         f'fold{fold}_tmp_test_mean': torch.tensor(tmp_test_acc[0]),
        #         f'fold{fold}_tmp_test_loss': tmp_test_loss,
        #         f'fold{fold}_best_test_mean': torch.tensor(test_acc[0]),
        #         f'fold{fold}_best_val_mean': torch.tensor(best_val_acc[0])
        #     }
        #     wandb.log(res_dict, step=epoch)
        if fold == 0:
            res_dict = {
                f'train_mean': torch.tensor(train_acc[0]),
                f'train_loss': train_loss,
                f'val_mean': torch.tensor(val_acc[0]),
                f'val_loss': val_loss,
                f'tmp_test_mean': torch.tensor(tmp_test_acc[0]),
                f'tmp_test_loss': tmp_test_loss,
                f'best_test_mean': torch.tensor(test_acc[0]),
                f'best_val_mean': torch.tensor(best_val_acc[0])
            }
            wandb.log(res_dict, step=epoch)

        scheduler.step(best_val_acc[0])
        new_best_trigger = val_acc[0] < best_val_acc[0] #if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    #torch.save(best_model, f'../best_{model_name}_v{version}.{num_run}_models/best_model_{fold}.pt')

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: ({test_acc[0]:.4f}, {test_acc[1]:.4f})")
    print(f"Best val acc: ({best_val_acc[0]:.4f}, {best_val_acc[1]:.4f})")

    if "ODE" not in args['model']:
        if "Discrete" in args['model']:
        # Debugging for discrete models
            for i in range(len(model.sheaf_learners)):
                L_max = model.sheaf_learners[i].L.detach().max().item()
                L_min = model.sheaf_learners[i].L.detach().min().item()
                L_avg = model.sheaf_learners[i].L.detach().mean().item()
                L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
                print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

            with np.printoptions(precision=3, suppress=True):
                for i in range(0, args['layers']):
                    print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")
        
        elif args['model'] == "GaussianSheafDiffusion":
            L_max = model.sheaf_learners.L.detach().max().item()
            L_min = model.sheaf_learners.L.detach().min().item()
            L_avg = model.sheaf_learners.L.detach().mean().item()
            L_abs_avg = model.sheaf_learners.L.detach().abs().mean().item()
            print(f"Laplacian: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

    wandb.log({'best_test_mean': test_acc[0], 'best_val_mean': best_val_acc[0], 'best_epoch': best_epoch})
    #keep_running = False if test_acc[0] < args['min_acc'] else True

    return test_acc, best_val_acc, best_model#, keep_running


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # se usar a gpu: export CUBLAS_WORKSPACE_CONFIG=:4096:8
    torch.manual_seed(43)
    torch.cuda.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    np.random.seed(43)
    random.seed(43)

    if args.model == 'GaussianMLP':
        model = GaussianMLP
    elif args.model == 'GaussianGCN':
        model = GaussianGCN
    elif args.model == 'GaussianSheafDiffusion':
        model = GaussianSheafDiffusion
    elif args.model == 'SampledGaussianSheafDiffusion':
        model = SampledGaussianSheafDiffusion
    elif args.model == 'DiscreteDiagSheafDiffusion':
        model = DiscreteDiagSheafDiffusion
    elif args.model == "DiscreteBundleSheafDiffusion":
        model = DiscreteBundleSheafDiffusion
    elif args.model == "DiscreteGeneralSheafDiffusion":
        model = DiscreteGeneralSheafDiffusion

    dataset = get_dataset(args.dataset)
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # Add extra arguments
    args.sha = sha
    args.graph_size = dataset.x.size(0)
    args.input_dim = dataset.x.size(1)
    args.dist_dim = dataset.dist_dim
    args.num_samples = dataset.num_samples
    args.samples_dim = dataset.samples_dim
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    print(f"Running with wandb account: {args.entity}")
    print(args)

    def run(model_class, num_run, dataset):
        # Set the seed for each run
        torch.manual_seed(num_run)
        torch.cuda.manual_seed(num_run)
        torch.cuda.manual_seed_all(num_run)
        np.random.seed(num_run)
        random.seed(num_run)

        mean_results = []
        std_results = []
        best_models = []
        for fold in tqdm(range(args.folds)):
            test_acc, best_val_acc, best_model = run_exp(wandb.config, dataset, model_class, fold)
            mean_results.append([test_acc[0], best_val_acc[0]])
            std_results.append([test_acc[1], best_val_acc[1]])
            best_models.append(best_model)

        fold = mean_results.index(min(mean_results, key=lambda x: x[1]))
        best_model = best_models[fold]

        if not os.path.exists(f'../best_{args.model}_models'):
            os.makedirs(f'../best_{args.model}_models')

        torch.save(best_model, f'../best_{args.model}_models/{wandb.run.name}.pt')
        artifact = wandb.Artifact(f'{wandb.run.name}', type='model')
        artifact.add_file(f'../best_{args.model}_models/{wandb.run.name}.pt')
        wandb.log_artifact(artifact)

        print(f'\n(Best model) Test acc: {mean_results[fold][0]:.4f} +/- {std_results[fold][0]:.4f}, Val acc: {mean_results[fold][1]:.4f} +/- {std_results[fold][1]:.4f}\n')

        wandb.log({'fold_best_test_mean': mean_results[fold][0], 'fold_best_val_mean': mean_results[fold][1]})

        test_acc_mean, val_acc_mean = np.mean(mean_results, axis=0)
        test_acc_std, val_acc_std = np.std(mean_results, axis=0)

        wandb_results = {'final_test_mean': test_acc_mean, 'final_test_std': test_acc_std, 'final_val_mean': val_acc_mean, 'final_val_std': val_acc_std}

        model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
        #print(f'{model_name} on {args.dataset} | SHA: {sha}')
        print(f'{model_name} on {args.dataset}')
        print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f} +/- {val_acc_std:.4f}')
        print('----------------------------------------')

        return wandb_results

    wandb.init(config=vars(args), entity=args.entity)
    wandb_results = run(model, args.seed, dataset)
    wandb.log(wandb_results)
    wandb.finish()