import argparse
import datetime
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import List
import wandb

import numpy as np
import torch
import torch.utils.data
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import EnsureType, Activations, AsDiscrete, Compose
from tqdm import trange

from experiments.pfedhn_seg.custom_losses import DiceBCELoss, Dice
from experiments.pfedhn_seg.models import CNNHyper, CNNTarget
from experiments.pfedhn_seg.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool

ALLOWED_DATASETS = ['Promise12', 'MSD', 'NCI_ISBI', 'PROSTATEx']

# DEBUG Helpers
DEBUG_MODE = False

# logging.basicConfig(
#     filename=f'run_{str(datetime.datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")}.log',
#     level=logging.INFO)

logging.basicConfig(
    # filename=f'run_{str(datetime.datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")}.log',
    level=logging.INFO)


# def dice_loss_3d(pred_3d, label_3d):
#     return 2 * (((pred_3d > 0.5) * (label_3d > 0.5)).sum().item() + 1) / (
#             (pred_3d > 0.5).sum().item() + (label_3d > 0.5).sum().item() + 1)

def eval_model(nodes, hnet, net, criteria, device, split, use_hnet, batch_norm_layers_dict):
    curr_results = evaluate(nodes, hnet, net, criteria, device, split=split, use_hnet=use_hnet,
                            batch_norm_layers_dict=batch_norm_layers_dict)
    avg_loss = (sum([node_res['total'] * node_res['loss'] for node_res in curr_results.values()]) / sum(
        [node_res['total'] for node_res in curr_results.values()]))
    avg_dice = (sum([node_res['total'] * node_res['mean_dice'] for node_res in curr_results.values()]) / sum(
        [node_res['total'] for node_res in curr_results.values()]))

    all_dice = [node_res['mean_dice'] for node_res in curr_results.values()]

    return curr_results, avg_loss, avg_dice, all_dice


@torch.no_grad()
def evaluate(nodes: BaseNodes, hnet, net, criteria, device, split='test', use_hnet=True, batch_norm_layers_dict=None):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))

    for node_id in range(len(nodes)):  # iterating over nodes
        running_loss, running_dice, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            raise ValueError(f'Unknown split: {split}')
            # curr_data = nodes.train_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            img, label = (
                batch["image"].to(device),
                batch["label"].to(device),
            )

            weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))

            if use_hnet:
                weights_with_batch_norm = {}
                weights_with_batch_norm.update(weights)

                if node_id in batch_norm_layers_dict:
                    weights_with_batch_norm.update(batch_norm_layers_dict[node_id])

                logging.debug(f"weights are assigned!")

                model_dict = net.state_dict()
                model_dict.update(weights_with_batch_norm)
                net.load_state_dict(model_dict)

            pred = net(img)

            # reshape pred and label to be complete images
            pred = pred.view(1, -1)
            label = label.view(1, -1)

            # when looking at val and test, the batch is just patches of one image
            running_loss += DiceBCELoss()(pred, label).item()
            running_dice += Dice()(pred, label).item()
            running_samples += 1

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['mean_dice'] = running_dice / (batch_count + 1)
        results[node_id]['total'] = running_samples

    return results


def train(data_names: List[str], data_path: str,
          steps: int, inner_steps: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int, no_hn_steps: int, dropout_p: float) -> None:
    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_names, data_path, batch_size=bs)

    embed_dim = embed_dim
    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + len(nodes) / 4)

    if all([d in ALLOWED_DATASETS for d in data_names]):
        # hnet = CNNHyper(len(nodes), embed_dim, hidden_dim=hyper_hid,
        #                 n_hidden=n_hidden, n_kernels=n_kernels, out_dim=100)
        net = CNNTarget(in_channels=1, out_channels=1, features=[16, 32, 64, 128], dropout_p=dropout_p)

        last_conv_block = [
                              l for l in net.state_dict() if
                              f"ups.{max([int(k.split('.')[1]) for k in net.state_dict().keys() if 'ups' in k])}" in l
                          ] + ['final_conv.weight', 'final_conv.bias']

        hnet = CNNHyper(len(nodes), embed_dim, hidden_dim=hyper_hid, model=net, n_hidden=n_hidden,
                        out_layers=last_conv_block)
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
    # use mix of Dice and BCE loss
    criteria = DiceBCELoss()
    # eval_metric = Dice()
    transform_post = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    inferer = SlidingWindowInferer(roi_size=(160, 160, 32), sw_batch_size=4, overlap=0.25)
    eval_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_dice = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)

    losses_dict = {}
    results = defaultdict(list)
    logging.info(f"Starting training with {len(nodes)} nodes")

    batch_norm_layers_dict = {}

    use_hnet = False
    for step in step_iter:
        # check if we should use hnet
        if step == no_hn_steps:
            logging.info(f"using hnet from step {step}")
            use_hnet = True

        hnet.train()

        # select client at random
        node_id = random.choice(range(len(nodes)))

        # produce & load local network weights
        weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
        if use_hnet:
            weights_with_batch_norm = {}
            weights_with_batch_norm.update(weights)

            if node_id in batch_norm_layers_dict:
                weights_with_batch_norm.update(batch_norm_layers_dict[node_id])

            logging.debug(f"weights are assigned!")

            # keep the BatchNorm params in the state_dict
            model_dict = net.state_dict()
            model_dict.update(weights_with_batch_norm)
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
            metric = 0
            cum_loss = 0
            for i, batch in enumerate(nodes.val_loaders[node_id]):
                val_img, val_label = (
                    batch["image"].to(device),
                    batch["label"].to(device),
                )

                # Inference
                val_outputs = inferer(val_img, net)
                # compute loss for validation set
                loss_score = criteria(val_outputs, val_label)
                cum_loss += loss_score.item()

                # Compute metric for 1 batch of validation set
                val_outputs = transform_post(val_outputs)
                metric_score = eval_metric(y_pred=val_outputs, y=val_label)
                metric += metric_score.item()

            # compute mean dice over whole validation set
            metric /= len(nodes.val_loaders[node_id])
            cum_loss /= len(nodes.val_loaders[node_id])

            # ========================================================================
            # This part is evaluation of batch, from previous implementation
            # ========================================================================
            # img = np.swapaxes(img, 0, 1).float()
            # label = np.swapaxes(label, 0, 1).float()
            # pred = net(img)
            #
            # # make sure pred is 0 for the padding part
            # pred = torch.where(img == img.min().item(), pred.min().item(), pred)
            #
            # # reshape pred and label to be complete images
            # pred = pred.view(1, -1)
            # label = label.view(1, -1)
            #
            # prvs_loss = criteria(pred, label)
            # ========================================================================

            # New implementation for evaluation

            if node_id not in losses_dict:
                losses_dict[node_id] = []
            losses_dict[node_id].append(cum_loss)

            # log loss to wandb
            if (len(losses_dict[node_id]) + 1) % 5 == 0:
                if not DEBUG_MODE:
                    wandb.log(
                        {f"target_val_loss_{node_id}": float(cum_loss)},
                        step=step
                    )
                    wandb.log(
                        {f"target_val_avg_dice_{node_id}": float(metric)},
                        step=step
                    )

            # prvs_dice = eval_metric(pred, label)
            # prvs_acc = np.mean([dice_loss_3d(pred[i], label[i]) for i in range(pred.shape[0])])

            net.train()

        # inner updates -> obtaining theta_tilda
        logging.debug(f"starting with inner steps")
        for i in range(inner_steps):
            net.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()

            batch = next(iter(nodes.train_loaders[node_id]))
            img, label = (
                batch["image"].to(device),
                batch["label"].to(device),
            )

            pred = net(img)

            loss = criteria(pred, label)
            loss.backward()

            if i % 10 == 0:
                if not DEBUG_MODE:
                    wandb.log(
                        {f"target_train_loss_{node_id}": float(loss)},
                        step=step
                    )

            torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

            inner_optim.step()
        # log loss to wandb
        if (len(losses_dict[node_id]) + 1) % 5 == 0:
            if not DEBUG_MODE:
                wandb.log(
                    {f"target_train_loss_{node_id}": float(loss)},
                    step=step
                )

        optimizer.zero_grad()

        final_state = net.state_dict()

        # update batch_norm_dict
        batch_norm_layers_dict[node_id] = OrderedDict(
            {k: tensor.data for k, tensor in final_state.items() if 'running_mean' in k or 'running_var' in k})

        logging.debug("done with inner steps")

        if use_hnet:
            # calculating delta theta
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            # calculating phi gradient
            hnet_grads = torch.autograd.grad(
                list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()), allow_unused=True
            )

            # update hnet weights
            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g

            torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)

            optimizer.step()
            logging.debug("done with hnet update")
        step_iter.set_description(
            f"Step: {step + 1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_dice:.4f}"
        )

        # Eval using test set
        if step % eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_dice, all_dice = eval_model(nodes, hnet, net, criteria, device,
                                                                    split="test", use_hnet=use_hnet,
                                                                    batch_norm_layers_dict=batch_norm_layers_dict)
            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Dice: {avg_dice:.4f}")
            if not DEBUG_MODE:
                wandb.log(
                    {f"Test - AVG Dice": float(avg_dice)},
                    step=step
                )
                wandb.log(
                    {f"Test - Dice - Node {node_id}": float(node_dice) for node_id, node_dice in enumerate(all_dice)},
                    step=step
                )
            results['test_avg_loss'].append(avg_loss)
            results['test_avg_dice'].append(avg_dice)

            _, val_avg_loss, val_avg_dice, _ = eval_model(nodes, hnet, net, criteria, device, split="val",
                                                          use_hnet=use_hnet,
                                                          batch_norm_layers_dict=batch_norm_layers_dict)
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
        _, val_avg_loss, val_avg_dice, _ = eval_model(nodes, hnet, net, criteria, device, split="val",
                                                      use_hnet=use_hnet, batch_norm_layers_dict=batch_norm_layers_dict)
        step_results, avg_loss, avg_dice, all_dice = eval_model(nodes, hnet, net, criteria, device,
                                                                split="test", use_hnet=use_hnet,
                                                                batch_norm_layers_dict=batch_norm_layers_dict)
        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Dice: {avg_dice:.4f}")

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

    # save models to wandb
    torch.onnx.export(hnet, torch.tensor([node_id], dtype=torch.long).to(device), "hyper_net.onnx")
    if not DEBUG_MODE:
        wandb.save("hyper_net.onnx")

    torch.onnx.export(net, img.float(), "target_net.onnx")
    if not DEBUG_MODE:
        wandb.save("target_net.onnx")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Hypernetwork with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-names", type=List[str],
        default=['Promise12', 'MSD', 'NCI_ISBI', 'PROSTATEx'],
        help="list of datasets to use for different clients"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")
    parser.add_argument("--no-hn-steps", type=int, default=0,
                        help="number of steps without HN (only federated learning)")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=20, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")
    parser.add_argument("--dropout-p", type=float, default=0.5, help="p for dropout layers")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="pfedhn_hetro_res", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if not DEBUG_MODE:
        wandb.login()
        with wandb.init(project='pFedHN-MedicalSegmentation',
                        entity='pfedhnmed',
                        name='UNET-3D-FULL-16-32-64-128-Dropout-HNforLastConv',
                        config=args):
            config = wandb.config
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
                seed=args.seed,
                no_hn_steps=args.no_hn_steps,
                dropout_p=args.dropout_p
            )
    else:
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
            seed=args.seed,
            no_hn_steps=args.no_hn_steps,
            dropout_p=args.dropout_p
        )
