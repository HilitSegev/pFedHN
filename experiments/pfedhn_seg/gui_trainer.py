import argparse
import os
from typing import List

import streamlit as st


parser = argparse.ArgumentParser(
    description="Federated Hypernetwork with Lookahead experiment"
)

#############################
#       Dataset Args        #
#############################

parser.add_argument(
    "--data-names", type=List[str],
    default=['promise12', 'medical_segmentation_decathlon', 'nci_isbi_2013', 'prostatex'],
    help="list of datasets to use for different clients"
)
parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")

##################################
#       Optimization args        #
##################################

parser.add_argument("--num-steps", type=int, default=5000)
parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--batch-size", type=int, default=32)
# TODO: change to 50
parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")

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
parser.add_argument("--spec-norm", type=str, default=False, help="hypernet hidden dim")
parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

#############################
#       General args        #
#############################
parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
parser.add_argument("--eval-every", type=int, default=50, help="eval every X selected epochs")
parser.add_argument("--save-path", type=str, default="pfedhn_hetro_res", help="dir path for output file")
parser.add_argument("--seed", type=int, default=42, help="seed value")

args = parser.parse_args()

data = {}
with st.form(key='my_form'):
    for k, v in args.__dict__.items():
        data[k] = st.text_input(label=k, value=str(v))
    submitted = st.form_submit_button(label='Submit')

if submitted:
    command = f'python3 trainer.py ' + ' '.join([f'--{k.replace("_", "-")} {v}' for k, v in data.items() if v != str(args.__dict__[k])])
    st.write(command)
    os.system(command)

