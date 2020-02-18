# Ada-LISTA: Learned Solvers Adaptive to Varying Models

## Getting Started

### Prerequisites

```
torch
numpy
scipy
matplotlib
PIL
```

## Simulated Experiments
We demonstrate the robustness of Ada-LISTA to three types of dictionary perturbations:
1. permuted columns,
2. additive Gaussian noise; 
3. and completely random dictionaries.
We demonstrate the ability of our model to handle complex and varying signal models while still providing an impressive advantage over both learned and non-learned solvers.

### 1. Permuted Columns
To run a small exmaple you may run the following:
```
python main.py -c0 -ntrain 1000 -epochs 10 -sigsnr 30
```
In this example the SNR of the signals is 30 [dB].

### 2. Noisy Dictionaries
To run a small exmaple you may run the following:
```
python main.py -c1 -ntrain 1000 -epochs 10 -sigsnr 30 -n 20
```
In this example the SNR of the signals is 30 [dB], while the SNR of the dictionaries is 20 [dB].

### 3. Random Dictionaries
To run a small exmaple you may run the following:
```
python main.py -c2 -ntrain 1000 -epochs 10 -sigsnr 30
```
In this example the SNR of the signals is 30 [dB].

## Image Inpainting
We demonstrate the use of Ada-LISTA on natural image inpainting, which cannot be directly used with hard-coded models as LISTA. We show a clear advantage of Ada-LISTA versus its non-learned counterparts.

In the `saved_models_inpainting` folder, there exists a trained Ada-LISTA model. However, to train a new model one can simply run the following:
```
python main.py -c3 -tstart 10 -tstep 1 -tend 11
```
Note that this script trains a model with 10 unfoldings.

To evaluate Ada-LISTA on set-11 and compare to ISTA and FISTA, you may run the following:
```
python eval.py
```

![Inpainting Result](figures/inpainting/Lena_ratio_0.5_T_20_lambd_0.1.png)
