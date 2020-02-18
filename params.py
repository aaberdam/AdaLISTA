import argparse
import numpy as np

""" Scenarios
0 - column perturbations.
1 - noised dictionary.
2 - random dictionary.
3 - inpainting application.
"""
SCENARIO_FLAG = 0

SCENARIO_DICT = {
    0: "Column Perturbations",
    1: "Noisy Dictionary",
    2: "Random Dictionary",
    3: "Image Inpainting",
}

# Parameters Initialization
n_dict = {0: 50, 1: 50, 2: 50, 3: 8 ** 2}
m_dict = {0: 70, 1: 70, 2: 70, 3: (8 ** 2) * 4}
s = 4
batch_size_dict = {0: 512, 1: 512, 2: 512, 3: 512}
lambd_dict = {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.1}
N_TRAIN_dict = {0: 20000, 1: 20000, 2: 20000, 3: 50000}
N_TEST = 1000
BATCH_SIZE = 512  # Batch size
cudaopt = True

snr_dict = {0: 0, 1: 20, 2: 0, 3: 0}
ORACLE_LISTA_DATA_dict = {0: True, 1: True, 2: True, 3: False}
LISTA_RANDOM_DATA_dict = {0: False, 1: True, 2: False, 3: False}
INPAINTING_RATIO_dict = {0: 0, 1: 0, 2: 0, 3: 0.5}
SAVE_MODEL_dict = {0: False, 1: False, 2: False, 3: True}
snr_signals_dict = {0: np.inf, 1: np.inf, 2: np.inf, 3: np.inf}

# Training parameters
EPOCH_dict = {0: 200, 1: 200, 2: 200, 3: 100}
lr_dict = {0: 0.00005, 1: 0.00005, 2: 0.00005, 3: 0.0000000005}
momentum = 0.9
weight_decay = 0
step_size_dict = {0: 50, 1: 50, 2: 50, 3: 50}
gamma = 0.1

# Number of unfoldings
tstart, tend, tstep = 0, 11, 2

# Parser the input
parser = argparse.ArgumentParser()
parser.add_argument("-s", help="Sparsity", type=int)
parser.add_argument("-n", help="SNR", type=int)
parser.add_argument("-c", help="Scenario", type=int)
parser.add_argument("-tstart", help="T start (Number of unfoldings)", type=int)
parser.add_argument("-tend", help="T end (Number of unfoldings)", type=int)
parser.add_argument("-tstep", help="T step (Number of unfoldings)", type=int)
parser.add_argument("-ntrain", help="N Train", type=int)
parser.add_argument("-sigsnr", help="Signal SNR", type=int)
parser.add_argument("-epochs", help="Number of epochs", type=int)
args = parser.parse_args()

if args.s is not None:
    s = args.s
if args.n is not None:
    snr_dict[1] = args.n
if args.c is not None:
    SCENARIO_FLAG = args.c
if args.tstart is not None:
    tstart = args.tstart
if args.tend is not None:
    tend = args.tend
if args.tstep is not None:
    tstep = args.tstep
if args.ntrain is not None:
    N_TRAIN_dict[SCENARIO_FLAG] = args.ntrain
if args.sigsnr is not None:
    snr_signals_dict[SCENARIO_FLAG] = args.sigsnr
if args.epochs is not None:
    EPOCH_dict[SCENARIO_FLAG] = args.epochs

# Set parameters per scenario
data_prop_dict = {
    "n": n_dict[SCENARIO_FLAG],
    "m": m_dict[SCENARIO_FLAG],
    "s": s,
    "lambd": lambd_dict[SCENARIO_FLAG],
    "N_TRAIN": N_TRAIN_dict[SCENARIO_FLAG],
    "N_TEST": N_TEST,
    "BATCH_SIZE": BATCH_SIZE,
    "SCENARIO_FLAG": SCENARIO_FLAG,
    "ORACLE_LISTA_FLAG": ORACLE_LISTA_DATA_dict[SCENARIO_FLAG],
    "snr_dict": snr_dict[SCENARIO_FLAG],
    "INPAINTING_RATIO": INPAINTING_RATIO_dict[SCENARIO_FLAG],
    "snr_signals": snr_signals_dict[SCENARIO_FLAG],
}
# Training parameters
training_data_dict = {
    "EPOCH": EPOCH_dict[SCENARIO_FLAG],
    "cudaopt": cudaopt,
    "lr": lr_dict[SCENARIO_FLAG],
    "momentum": momentum,
    "weight_decay": weight_decay,
    "step_size": step_size_dict[SCENARIO_FLAG],
    "gamma": gamma,
    "file_name": "Adaptive-ISTA",
    "SAVE_FLAG": True,
    "LISTA_RANDOM_DATA": LISTA_RANDOM_DATA_dict[SCENARIO_FLAG],
    "SAVE_MODEL": SAVE_MODEL_dict[SCENARIO_FLAG],
}
SCENARIO_Name = SCENARIO_DICT[SCENARIO_FLAG]
snr = snr_dict[SCENARIO_FLAG]

# Plot design
design_dict = {"ISTA": "b^", "FISTA": "cs", "Oracle-LISTA": "rx", "LISTA": "m+", "Ada-LISTA": "go"}
