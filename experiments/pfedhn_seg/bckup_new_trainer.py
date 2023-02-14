# imports and constants
import random

import numpy as np
import torch
import torch.utils.data
import wandb
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import EnsureType, Activations, AsDiscrete, Compose
from torch.nn import BCELoss

from custom_losses import *
from experiments.pfedhn_seg import monai_datasets
from experiments.pfedhn_seg.models import CNNTarget

config = {
    'num_steps': 35000,
    'inner_steps': 25,
    'batch_size': 6,
    'data_path': '/dsi/shared/hilita/ProstateSegmentation/',
    'dropout_p': 0.5,
    'embed_dim': -1,
    'hyper_hidden_dim': 100,
    'hyper_n_hidden': 20,
    'device': 'cuda:1',

}


# evaluation
def eval_model(model, dataloader):
    # define metrics
    criteria = DiceBCELoss()
    inferer = SlidingWindowInferer(roi_size=(160, 160, 32),
                                   sw_batch_size=4,
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


"""
define models
"""
# UNet
target_net = CNNTarget(in_channels=1,
                       out_channels=1,
                       features=[16, 32, 64, 128],
                       dropout_p=config['dropout_p'])

# # HyperNet
# hnet = CNNHyper(config['embed_dim'], 1, hidden_dim=config['hyper_hidden_dim'], n_hidden=config['hyper_n_hidden'])

target_net.to(config['device'])

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

"""
train loop

0. overfitting over 1 image only from node0
1. train over real batches from node0
2. randomly select nodes
3. add HyperNet
"""

# wandb login
wandb.login()
with wandb.init(project='pFedHN-MedicalSegmentation-Overfittinng-Unet',
                entity='pfedhnmed',
                name='UNET-Overfitting-1image-1node',
                config=config):
    # optimizers
    optimizer = torch.optim.Adam(target_net.parameters(), lr=5e-3, weight_decay=1e-5)

    # criteria
    criteria = DiceBCELoss()

    # training loop
    for step in range(config['num_steps']):
        # select node
        node_id = random.choice(range(len(train_loaders)))
        train_loader = train_loaders[node_id]

        # TODO: predict using HyperNet

        # TODO: load parameters to target_net

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

        # TODO: train HyperNet

        # print loss on train set
        print(f"Step {step} Loss: {loss.item()}")
        wandb.log(
            {f"target_train_loss_{node_id}": float(loss)},
            step=step
        )

        # evaluate on validation dataset
        if step % 10 == 0:
            metric, loss = eval_model(target_net, val_loaders[node_id])
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
