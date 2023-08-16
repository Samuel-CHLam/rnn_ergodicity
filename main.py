import numpy as np

import torch
# import torch.mps

import math
from tqdm import tqdm
from simulation import simulate_all

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--n_neurons', type=int, default=100, help='number of neurons in the hidden layer')
parser.add_argument('--max_time', type=int, default=1000, help='number of simulation steps')
parser.add_argument('--n_paths', type=int, default=10, help='number of simulated paths')
parser.add_argument('--rolling_mean', type=bool, default=True, help='if True then return rolling mean, if False then return actual samples')
parser.add_argument('--memorize', type=bool, default=True, help='if True then include memory layer')
parser.add_argument('--store_xz', type=bool, default=False, help='if True then return the input sequence as well')
parser.add_argument('--seed', type=int, default=1234, help='seed for random number generation')
parser.add_argument('--device', type=str, default='cuda', help='device for running simulation CPU/Cuda/MPS')

args = parser.parse_args()

simulate_avg_h, simulate_avg_h2, time_arr, = simulate_all(n_neurons=args.n_neurons, 
                                                          max_time=args.max_time, 
                                                          n_paths=args.n_paths, 
                                                          rolling_mean=args.rolling_mean, 
                                                          memorize=args.memorize, 
                                                          store_xz=args.store_xz,
                                                          seed=args.seed,
                                                          use_device=args.device)

output_df = pd.DataFrame()
output_df["time_arr"] = time_arr

output_df["overall_mean_mean_h"] = simulate_avg_h.mean(axis=(1,2)).to(device="cpu").numpy()
output_df["overall_mean_std_h"] = simulate_avg_h.std(axis=(1,2)).to(device="cpu").numpy()
output_df["overall_var_mean_h"] = simulate_avg_h2.mean(axis=(1,2)).to(device="cpu").numpy()
output_df["overall_var_std_h"] = simulate_avg_h2.std(axis=(1,2)).to(device="cpu").numpy()

output_df["lower_mean_h"] = simulate_avg_h.mean(axis=1).min(axis=1).values.to(device="cpu").numpy()
output_df["upper_mean_h"] = simulate_avg_h.mean(axis=1).max(axis=1).values.to(device="cpu").numpy()
output_df["union_lower_mean_h"] = (simulate_avg_h.mean(axis=1) - 2 * simulate_avg_h.std(axis=1)).min(axis=1).values.to(device="cpu").numpy()
output_df["union_upper_mean_h"] = (simulate_avg_h.mean(axis=1) + 2 * simulate_avg_h.std(axis=1)).max(axis=1).values.to(device="cpu").numpy()

output_df["lower_var_h"] = simulate_avg_h2.mean(axis=1).min(axis=1).values.to(device="cpu").numpy()
output_df["upper_var_h"] = simulate_avg_h2.mean(axis=1).max(axis=1).values.to(device="cpu").numpy()
output_df["union_lower_var_h"] = (simulate_avg_h2.mean(axis=1) - 2 * simulate_avg_h2.std(axis=1)).min(axis=1).values.to(device="cpu").numpy()
output_df["union_upper_var_h"] = (simulate_avg_h2.mean(axis=1) + 2 * simulate_avg_h2.std(axis=1)).max(axis=1).values.to(device="cpu").numpy()

if args.rolling_mean:
    output_df.to_csv(f"./output/simulation_n_neurons_{args.n_neurons}_max_time_{args.max_time}_rolling_mean")
else:
    output_df.to_csv(f"./output/simulation_n_neurons_{args.n_neurons}_max_time_{args.max_time}_actual")

