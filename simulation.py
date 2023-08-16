import numpy as np

import torch
import torch.mps

import math
from tqdm import tqdm

def simulate_all(n_neurons : int = 100, T : int = 10, max_time : int or None = None, 
                 n_paths : int = 20, max_point : int or None = 500,
                 memorize : bool = True, rolling_mean : bool = False, store_xz : bool = False, 
                 seed : int = 1234, use_device : str = "cuda"): 

    torch.manual_seed(seed)

    if use_device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif use_device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(use_device)

    with torch.no_grad():

        # simulation dynamics
        P = torch.tensor([[math.sqrt(3)/2, -1/2], [1/2, math.sqrt(3)/2]])
        A = (P @ torch.diag(torch.tensor([1.,0.5])) @ torch.inverse(P)).to(device=device)

        def g(xz : torch.tensor) -> torch.tensor: # xz takes 2D array for individual point or 3D array
            return torch.tanh(xz @ A.T) / 2

        # setups

        if (max_time is None):
            max_time = T * n_neurons

        sample = max(max_time // max_point, 1)
        time_arr = sample * torch.arange(max_point)
        max_time = sample * max_point # actual maximum time

        print(f"case for n_neurons={n_neurons}, max_time={max_time}")

        # initialise underlying chain and neural network parameters

        if memorize == True:
            B0 = torch.rand((n_neurons, n_paths)).to(device=device)
        else: 
            B0 = torch.zeros((n_neurons, n_paths)).to(device=device)

        W0 = torch.randn((n_neurons, n_paths)).to(device=device)

        new_simulant_xz = -1 + 2 * torch.rand((n_paths,2)).to(device=device)
        old_simulant_xz = new_simulant_xz
        new_simulant_h = torch.special.expit(W0 * new_simulant_xz.T[0]).to(device=device)
        old_simulant_h = new_simulant_h

        # simulation 
        # the arrays of (x,z) is of the format (steps, paths, dimension of entry = 2)
        # the array of memories are of the format (steps, neurons, paths)
        # for each step we first step through (x,z), then step through the memories 

        if store_xz == True:
            simulate_xz = torch.zeros((max_point, n_paths, 2), dtype=torch.float).to(device=device)
            simulate_xz[0] = old_simulant_xz
        
        if rolling_mean == True:
            simulate_sum_h = old_simulant_h
            simulate_sum_h2 = (old_simulant_h ** 2)

            simulate_avg_h = torch.zeros((max_point, n_neurons, n_paths), dtype=torch.float).to(device=device)
            simulate_avg_h2 = torch.zeros((max_point, n_neurons, n_paths), dtype=torch.float).to(device=device)

            print("now simulate the neurons, storing rolling mean")

            for k in tqdm(range(1, max_time - 1)):
                new_simulant_h = torch.special.expit(W0 * old_simulant_xz.T[0] + torch.sum(B0 * old_simulant_h, axis=0) / n_neurons)
                old_simulant_h = new_simulant_h
                simulate_sum_h += old_simulant_h
                simulate_sum_h2 += old_simulant_h ** 2
                new_simulant_xz = g(old_simulant_xz) - 0.5 + torch.rand((n_paths,2), device=device)
                old_simulant_xz = new_simulant_xz

                if (k % sample == 0):
                    simulate_avg_h[(k // sample)] = simulate_sum_h / k
                    simulate_avg_h2[(k // sample)] = simulate_sum_h2 / k
                    if store_xz == True: 
                        simulate_xz[(k // sample)] = new_simulant_xz

            # del new_simulant_h, old_simulant_h, simulate_sum_h, simulate_sum_h2, new_simulant_xz, old_simulant_xz
            
            if store_xz == True:
                return simulate_avg_h, simulate_avg_h2, time_arr, simulate_xz
            else:
                return simulate_avg_h, simulate_avg_h2, time_arr

        else:
            simulate_h = torch.zeros((max_point, n_neurons, n_paths)).to(device=device)

            print("now simulate the neurons, storing history")

            for k in tqdm(range(1, max_time - 1)):
                new_simulant_h = torch.special.expit(W0 * old_simulant_xz.T[0] + torch.sum(B0 * old_simulant_h, axis=0) / n_neurons)
                old_simulant_h = new_simulant_h
                new_simulant_xz = g(old_simulant_xz) - 0.5 + torch.rand(size=(n_paths,2), device=device)
                old_simulant_xz = new_simulant_xz

                if k % sample == 0:
                    simulate_h[(k // sample)] = old_simulant_h

                    if store_xz == True: 
                        simulate_xz[(k // sample)] = old_simulant_xz

            # del new_simulant_h, old_simulant_h, simulate_sum_h, simulate_sum_h2, new_simulant_xz, old_simulant_xz

            if store_xz == True: 
                return simulate_h, time_arr, simulate_xz
            else:
                return simulate_h, time_arr