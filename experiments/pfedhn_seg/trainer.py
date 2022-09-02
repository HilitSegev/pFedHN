import argparse
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data
from tqdm import trange

from experiments.pfedhn_seg.models import CNNHyper, CNNTarget
from experiments.pfedhn_seg.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool

ALLOWED_DATASETS = ['promise12', 'medical_segmentation_decathlon', 'nci_isbi_2013']


def dice_loss_3d(pred_3d, label_3d):
    return 2 * (((pred_3d > 0) * (label_3d > 0)).sum().item() + 1) / (
            (pred_3d > 0).sum().item() + (label_3d > 0).sum().item() + 1)


def eval_model(nodes, hnet, net, criteria, device, split):
    curr_results = evaluate(nodes, hnet, net, criteria, device, split=split)
    avg_loss = (sum([node_res['total'] * node_res['loss'] for node_res in curr_results]) / sum(
        [node_res['total'] for node_res in curr_results]))
    avg_dice = (sum([node_res['total'] * node_res['mean_dice'] for node_res in curr_results]) / sum(
        [node_res['total'] for node_res in curr_results]))

    all_dice = [node_res['mean_dice'] for node_res in curr_results.values()]

    return curr_results, avg_loss, avg_dice, all_dice


@torch.no_grad()
def evaluate(nodes: BaseNodes, hnet, net, criteria, device, split='test'):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))

    for node_id in range(len(nodes)):  # iterating over nodes

        running_loss, running_dice, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)

            weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            net.load_state_dict(weights)
            pred = net(img.float())
            running_loss += criteria(pred, label).item()
            # TODO: update to use dice loss (and update mean)
            running_dice = (running_samples * running_dice + \
                            len(label) * np.mean([dice_loss_3d(pred[i], label[i]) for i in range(pred.shape[0])])) \
                           / (running_samples + len(label))
            # running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['mean_dice'] = running_dice
        results[node_id]['total'] = running_samples

    return results


def train(data_names: List[str], data_path: str,
          steps: int, inner_steps: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int) -> None:
    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_names, data_path, batch_size=bs)

    embed_dim = embed_dim
    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + len(nodes) / 4)

    # TODO: Define models for the segmentation problem
    # if data_name == "cifar10":
    #     hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid, n_hidden=n_hidden, n_kernels=n_kernels)
    #     net = CNNTarget(n_kernels=n_kernels)
    # elif data_name == "cifar100":
    #     hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid,
    #                     n_hidden=n_hidden, n_kernels=n_kernels, out_dim=100)
    #     net = CNNTarget(n_kernels=n_kernels, out_dim=100)
    if all([d in ALLOWED_DATASETS for d in data_names]):
        hnet = CNNHyper(len(nodes), embed_dim, hidden_dim=hyper_hid,
                        n_hidden=n_hidden, n_kernels=n_kernels, out_dim=100)
        net = CNNTarget(in_channels=15, out_channels=15, features=[64, 128, 256, 512])
    else:
        raise ValueError(f"choose data_name from {ALLOWED_DATASETS}")

    hnet = hnet.to(device)
    net = net.to(device)

    ##################
    # init optimizer #
    ##################
    embed_lr = embed_lr if embed_lr is not None else lr
    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr, momentum=0.9, weight_decay=wd
        ),
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    # TODO: is this the loss to use?
    criteria = torch.nn.BCEWithLogitsLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_dice = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)

    results = defaultdict(list)
    for step in step_iter:
        hnet.train()

        # select client at random
        node_id = random.choice(range(len(nodes)))

        # produce & load local network weights
        weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))

        # keep the BatchNorm params in the state_dict
        model_dict = net.state_dict()
        model_dict.update(weights)
        net.load_state_dict(model_dict)

        # init inner optimizer
        inner_optim = torch.optim.SGD(
            net.parameters(), lr=inner_lr, momentum=.9, weight_decay=inner_wd
        )

        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        # NOTE: evaluation on sent model
        with torch.no_grad():
            net.eval()
            batch = next(iter(nodes.test_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)
            pred = net(img.float())
            prvs_loss = criteria(pred, label)

            prvs_acc = np.mean([dice_loss_3d(pred[i], label[i]) for i in range(pred.shape[0])])
            net.train()

        # inner updates -> obtaining theta_tilda
        for i in range(inner_steps):
            net.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()

            batch = next(iter(nodes.train_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)

            pred = net(img.float())

            loss = criteria(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

            inner_optim.step()

        optimizer.zero_grad()

        final_state = net.state_dict()

        # calculating delta theta
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        # calculating phi gradient
        hnet_grads = torch.autograd.grad(
            list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
        )

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        step_iter.set_description(
            f"Step: {step + 1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_acc:.4f}"
        )

        if step % eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_dice, all_dice = eval_model(nodes, hnet, net, criteria, device,
                                                                  split="test")
            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_dice:.4f}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_dice'].append(avg_dice)

            _, val_avg_loss, val_avg_dice, _ = eval_model(nodes, hnet, net, criteria, device, split="val")
            if best_dice < val_avg_dice:
                best_dice = val_avg_dice
                best_step = step
                test_best_based_on_step = val_avg_dice
                test_best_min_based_on_step = np.min(all_dice)
                test_best_max_based_on_step = np.max(all_dice)
                test_best_std_based_on_step = np.std(all_dice)

            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_dice)
            results['best_step'].append(best_step)
            results['best_val_acc'].append(best_dice)
            results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
            results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
            results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
            results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    if step != last_eval:
        _, val_avg_loss, val_avg_dice, _ = eval_model(nodes, hnet, net, criteria, device, split="val")
        step_results, avg_loss, avg_dice, all_dice = eval_model(nodes, hnet, net, criteria, device,
                                                              split="test")
        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_dice:.4f}")

        results['test_avg_loss'].append(avg_loss)
        results['test_avg_acc'].append(avg_dice)

        if best_dice < val_avg_dice:
            best_dice = val_avg_dice
            best_step = step
            test_best_based_on_step = avg_dice
            test_best_min_based_on_step = np.min(all_dice)
            test_best_max_based_on_step = np.max(all_dice)
            test_best_std_based_on_step = np.std(all_dice)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_dice)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_dice)
        results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
        results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
        results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
        results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / f"results_{inner_steps}_inner_steps_seed_{seed}.json"), "w") as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Hypernetwork with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-names", type=List[str], default=['promise12', 'medical_segmentation_decathlon', 'nci_isbi_2013'],
        help="list of datasets to use for different clients"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    # TODO: change to 50
    parser.add_argument("--inner-steps", type=int, default=2, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="pfedhn_hetro_res", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    train(
        data_names=args.data_names,
        data_path=args.data_path,
        steps=args.num_steps,
        inner_steps=args.inner_steps,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed
    )
