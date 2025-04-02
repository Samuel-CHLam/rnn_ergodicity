# %%
import argparse
import math

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import special
from scipy.stats import probplot
from tqdm import tqdm

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--n_neurons', type=int, default=1000, help="number of neurons")
parser.add_argument('--n_samples', type=int, default=100, help="number of samples")
parser.add_argument('--n_iter', type=int, default=100, help="number of layers")
parser.add_argument('--device', type=str, default="cpu", help="GPU devices")


args = parser.parse_args()

N = args.n_neurons
n_sample = args.n_samples
device = args.device
k = args.n_iter

with torch.no_grad():
    x = torch.randn(5, dtype=torch.float16).to(device=device)
    W = torch.randn(5, N, dtype=torch.float16).to(device=device)
    H = x @ W
    S = torch.special.expit(H)[None, :, None]
    B = torch.randn(n_sample, N, N, dtype=torch.float16).to(device=device)

# %%
for _ in tqdm(range(k)):
    H = B @ S / math.sqrt(N)
    S = torch.special.expit(H)

H_final = H.to(device="cpu")

try:
    np.save("/math-hpc-machine-learning/univ5366", H_final)
except:
    np.save("/", H_final)
