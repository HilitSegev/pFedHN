import torch.utils.data
import wandb
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import EnsureType, Activations, AsDiscrete, Compose
from experiments.pfedhn_seg import monai_datasets
import argparse
from custom_losses import *
from experiments.pfedhn_seg.models import CNNTarget

# ==================== CONFIG ====================
parser = argparse.ArgumentParser(
    description="Federated Hypernetwork with Lookahead experiment"
)

#############################
#       Dataset Args        #
#############################

parser.add_argument("--data-path", type=str, default="/dsi/shared/hilita/ProstateSegmentation/",
                    help="dir path for MNIST dataset")
parser.add_argument("--data-name", type=str, default="Promise12",
                    help="dataset name")

##################################
#       Optimization args        #
##################################
parser.add_argument("--inner-steps", type=int, default=25000, help="number of inner steps")

################################
#       Model Prop args        #
################################
parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
parser.add_argument("--dropout-p", type=float, default=0.5, help="p for dropout layers")

#############################
#       General args        #
#############################
parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")

args = parser.parse_args()
assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

config = {
    'inner_steps': args.inner_steps,
    'batch_size': 4,
    'data_path': args.data_path,
    'data_name': args.data_name,
    'dropout_p': args.dropout_p,
    'device': f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu',
    'inner_lr': args.inner_lr,
}


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


"""
define models
"""
# UNet
net = CNNTarget(in_channels=1,
                out_channels=1,
                features=[16, 32, 64],
                dropout_p=config['dropout_p'])

net.to(config['device'])

# wandb login
wandb.login()
with wandb.init(project='pFedHN_Local_Unet',
                entity='pfedhnmed',
                name=f'Unet_{config["data_name"]}',
                config=config):
    print("wandb init")

    """
    define DataLoaders for each client
    """
    train_loader = {}
    val_loader = {}
    test_loader = {}
    datasets = monai_datasets.get_datasets(config["data_name"], config['data_path'])
    train_loader = DataLoader(datasets[0], batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=1, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=1, shuffle=False)

    # optimizers
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=config['inner_lr'],
                                 weight_decay=1e-5)
    # criteria
    criteria = DiceBCELoss(dice_weight=0, bce_weight=1)

    """training"""

    for inner_step in range(config['inner_steps']):
        net.train()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # get prediction
        batch = next(iter(train_loader))
        img, label = (
            batch["image"].to(config['device']),
            batch["label"].to(config['device']),
        )
        pred = net(img)

        # calculate loss
        loss = criteria(pred, label)

        # calculate gradients
        loss.backward()

        # update weights
        optimizer.step()

        # print loss on train set
        print(f"Step {inner_step} Loss: {loss.item()}")
        wandb.log(
            {f"train_loss": float(loss)},
            step=inner_step
        )
        # evaluate on validation dataset
        if inner_step % 50 == 0:
            metric, loss = eval_model(net, val_loader)

            print(f"Step {inner_step}: Validation Metric: {metric} Validation Loss: {loss}")
            wandb.log(
                {f"target_val_loss": float(loss),
                 f"target_val_avg_dice": float(metric)},
                step=inner_step
            )
        # evaluate on test datasets
        if inner_step % 250 == 0:
            test_evaluation = {}
            metric, loss = eval_model(net, test_loader)
            test_evaluation = {'metric': float(metric),
                                         'loss': float(loss)}
            print(f"Step {inner_step}: Test Metric: {metric} Test Loss: {loss}")
            wandb.log(
                {f"target_test_loss": float(loss),
                 f"target_test_avg_dice": float(metric)},
                step=inner_step
            )

