# imports and constants
import argparse
from collections import OrderedDict

print("imports and constants")
import random

import numpy as np
import torch.utils.data
import wandb
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import EnsureType, Activations, AsDiscrete, Compose

from custom_losses import *
from experiments.pfedhn_seg import monai_datasets
from experiments.pfedhn_seg.models import CNNTarget, CNNHyper

print("imports and constants done")

# ==================== CONFIG ====================
parser = argparse.ArgumentParser(
    description="Federated Hypernetwork with Lookahead experiment"
)

#############################
#       Dataset Args        #
#############################

parser.add_argument("--data-path", type=str, default="/dsi/shared/hilita/ProstateSegmentation/",
                    help="dir path for MNIST dataset")

##################################
#       Optimization args        #
##################################
parser.add_argument("--num-steps", type=int, default=10000)
parser.add_argument("--inner-steps", type=int, default=10, help="number of inner steps")

################################
#       Model Prop args        #
################################
parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
parser.add_argument("--hn-lr", type=float, default=3e-2, help="learning rate")
parser.add_argument("--dropout-p", type=float, default=0.5, help="p for dropout layers")
parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dimension")
parser.add_argument("--unet-size", type=int, default=16, help="number of channels in the first conv layer")

#############################
#       General args        #
#############################
parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
parser.add_argument("--use-hn", type=int, default=0, help="use HyperNetwork or not")
parser.add_argument("--use-hn-for-final-conv", type=int, default=1,
                    help="use HyperNetwork for the last convolution layer")
parser.add_argument("--use-hn-for-batch-norm", type=int, default=1, help="use HyperNetwork for batch normalization")

args = parser.parse_args()
assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

config = {
    'num_steps': args.num_steps,
    'inner_steps': args.inner_steps,
    'batch_size': 4,
    'data_path': args.data_path,
    'dropout_p': args.dropout_p,
    'embed_dim': args.embed_dim,  # -1,
    'hyper_hidden_dim': 100,
    'hyper_n_hidden': 20,
    'device': f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu',
    'hn_lr': args.hn_lr,
    'inner_lr': args.inner_lr,
    'use_hn': args.use_hn,
    'use_hn_for_final_conv': args.use_hn_for_final_conv,
    'use_hn_for_batch_norm': args.use_hn_for_batch_norm,
    'unet_size': args.unet_size
}

print(config)


# evaluation
def eval_model(model, dataloader):
    # define metrics
    criteria = DiceBCELoss(dice_weight=1, bce_weight=1)
    inferer = SlidingWindowInferer(roi_size=(160, 160, 32),
                                   sw_batch_size=1,
                                   overlap=0.25,
                                   device=config['device'],
                                   sw_device=config['device'],
                                   progress=False)
    eval_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    # Sigmoid and Threshold are **not** included in DiceMetric, so need to do the post_transform
    transform_post = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # loop over dataloader
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        cum_metric = 0
        cum_loss = 0
        for i, batch_data in enumerate(dataloader):
            val_img, val_label = batch_data["image"].to(config['device']), batch_data["label"].to(config['device'])
            # make prediction
            val_outputs = inferer(val_img, model)

            loss = criteria(val_outputs, val_label)
            cum_loss += loss.item()

            val_outputs = transform_post(val_outputs)
            metric_score = eval_metric(val_outputs, val_label)
            cum_metric += metric_score.item()

    # calculate metrics
    # compute mean dice over whole validation set
    mean_metric = cum_metric / len(dataloader)
    mean_loss = cum_loss / len(dataloader)

    # return metrics
    return mean_metric, mean_loss


def init_local_net(net, hnet, node_id, use_hnet, device):
    weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
    if use_hnet:
        model_dict = net.state_dict()
        model_dict.update(weights)
        net.load_state_dict(model_dict)
    return net, weights


"""
define models
"""
# UNet
target_net = CNNTarget(in_channels=1,
                       out_channels=1,
                       features=[config['unet_size'], 2 * config['unet_size'], 4 * config['unet_size']],
                       dropout_p=config['dropout_p'])

# HyperNet
last_conv_block = []
if config['use_hn_for_final_conv']:
    last_conv_block += ['final_conv.weight', 'final_conv.bias']
if config['use_hn_for_batch_norm']:
    last_conv_block += [k for k in target_net.state_dict() if 'running_' in k]

hnet = CNNHyper(n_nodes=4,
                embedding_dim=config['embed_dim'],
                model=target_net,
                out_layers=last_conv_block,
                in_channels=1,
                hidden_dim=config['hyper_hidden_dim'],
                n_hidden=config['hyper_n_hidden'])

target_net.to(config['device'])
hnet.to(config['device'])

# wandb login
wandb.login()
with wandb.init(project='pFedHN-MedicalSegmentation-Unet',
                entity='pfedhnmed',
                name=f'pFedHN-MedicalSegmentation-Unet' + f'_{"hn" if config["use_hn"] else "nohn"}',
                config=config):
    print("wandb init")

    """
    define DataLoaders for each client
    """
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for node_id, data_name in enumerate(['Promise12', 'MSD', 'NCI_ISBI', 'PROSTATEx']):
        datasets = monai_datasets.get_datasets(data_name, config['data_path'])
        train_loaders[node_id] = DataLoader(datasets[0], batch_size=config['batch_size'], shuffle=True)
        val_loaders[node_id] = DataLoader(datasets[1], batch_size=1, shuffle=False)
        test_loaders[node_id] = DataLoader(datasets[2], batch_size=1, shuffle=False)

    # optimizers
    optimizer = torch.optim.Adam(target_net.parameters(),
                                 lr=config['inner_lr'],
                                 weight_decay=1e-5)

    hn_optimizer = torch.optim.Adam(hnet.parameters(),
                                    lr=config['hn_lr'],
                                    weight_decay=1e-5)

    # criteria
    criteria = DiceBCELoss(dice_weight=0, bce_weight=1)

    # training loop
    for step in range(config['num_steps']):
        # select node
        node_id = random.choice(range(len(train_loaders)))
        train_loader = train_loaders[node_id]

        # predict using HyperNet
        target_net, predicted_weights = init_local_net(net=target_net,
                                                       hnet=hnet,
                                                       node_id=node_id,
                                                       use_hnet=config['use_hn'],
                                                       device=config['device'])

        # get initial state for later calculation
        inner_state = OrderedDict({k: tensor.data for k, tensor in predicted_weights.items()})

        # train target_net
        target_net.train()

        for inner_step in range(config['inner_steps']):
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            # get prediction
            batch = next(iter(train_loader))
            img, label = (
                batch["image"].to(config['device']),
                batch["label"].to(config['device']),
            )
            pred = target_net(img)

            # calculate loss
            loss = criteria(pred, label)

            # calculate gradients
            loss.backward()

            # update weights
            optimizer.step()

        # train HyperNet
        hn_optimizer.zero_grad()

        final_state = target_net.state_dict()

        if config['use_hn']:
            # calculating delta theta - should it be inner-final? or vice versa?
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in predicted_weights.keys()})

            # calculating phi gradient
            hnet_grads = torch.autograd.grad(
                list(predicted_weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()),
                allow_unused=True
            )

            # update hnet weights
            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g

            torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)

            hn_optimizer.step()

        # print loss on train set
        print(f"Step {step} Loss: {loss.item()}")
        wandb.log(
            {f"target_train_loss_{node_id}": float(loss)},
            step=step
        )

        # evaluate on validation dataset
        if step % 10 == 0:
            metric, loss = eval_model(target_net, val_loaders[node_id])

            # change loss to support DiceLoss
            # if loss < 0.2 and step % 100 == 0:
            #     dice_weight = criteria.dice_weight
            #     criteria = DiceBCELoss(dice_weight=dice_weight + 1, bce_weight=1)

            print(f"Step {step}: Validation Metric: {metric} Validation Loss: {loss}")
            wandb.log(
                {f"target_val_loss_{node_id}": float(loss),
                 f"target_val_avg_dice_{node_id}": float(metric)},
                step=step
            )

        # evaluate on test datasets
        if step % 50 == 0:
            test_evaluations = {}
            for node_id in test_loaders.keys():
                # load target_net
                target_net, _ = init_local_net(net=target_net,
                                               hnet=hnet,
                                               node_id=node_id,
                                               use_hnet=config['use_hn'],
                                               device=config['device'])

                metric, loss = eval_model(target_net, test_loaders[node_id])
                test_evaluations[node_id] = {'metric': float(metric),
                                             'loss': float(loss)}
                print(f"Step {step}: Test Metric: {metric} Test Loss: {loss}")
                wandb.log(
                    {f"target_test_loss_{node_id}": float(loss),
                     f"target_test_avg_dice_{node_id}": float(metric)},
                    step=step
                )

            # log avg test evaluations
            wandb.log(
                {
                    f"target_test_{key}_avg":
                        np.mean([test_evaluations[node_id][key] for node_id in test_evaluations.keys()])
                    for key in ['metric', 'loss']
                },
                step=step
            )

    # save models to wandb
    torch.onnx.export(hnet, torch.tensor([node_id], dtype=torch.long).to(config['device']), f"hyper_net_{step}.onnx")
    wandb.save(f"hyper_net_{step}.onnx")

    torch.onnx.export(target_net, img.float(), f"target_net.onnx")
    wandb.save("target_net.onnx")
