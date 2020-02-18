import torch
import numpy as np
from os.path import dirname, abspath
import os
import matplotlib.pyplot as plt
from PIL import Image
import useful_utils
import generating
import torch.utils.data as Data

# load D that was calculated with train set
dict_path = "./data/data_8x8_N_100000_atoms_256.npz"
save_path = "./figures/inpainting/figs_for_paper/"
npzfile = np.load(dict_path, allow_pickle=True)
D, mean_avg, mean_std = (npzfile["D"].astype(np.float32), npzfile["avg_mean"], npzfile["avg_std"])
NUM_ATOMS = np.shape(D)[1]

T = 20
lambd = 0.1
RATIO = 0.5
data_path = "./data/set11"
image_list = sorted(os.listdir(data_path))
PATCH_SIZE = 8
n = PATCH_SIZE ** 2
OVERLAP = True
BATCH_SIZE = 512
cudaopt = True

# Load Ada-LISTA Model
model_adalista_path = (f"./saved_models_inpainting/adalista_T{T}")
model_adalista = torch.load(model_adalista_path)
model_adalista.eval()


def infer_adalista(corrupt_patches, mask_patches, D, model):
    NUM_PATCHES = np.shape(corrupt_patches)[1]
    data_for_loader = generating.SimulatedDataNoisedDict(
        y=torch.from_numpy(corrupt_patches),
        D_noised=torch.from_numpy(mask_patches),
        x=torch.zeros(NUM_ATOMS, NUM_PATCHES),
    )
    data_loader = Data.DataLoader(
        dataset=data_for_loader, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )
    count = 0
    recon_patches = np.zeros_like(corrupt_patches)

    for (b_y, b_M, b_x) in data_loader:
        if cudaopt:
            b_y, b_M, b_x = b_y.cuda(), b_M.cuda(), b_x.cuda()
        x_hat = model(b_y, b_M).T
        x_hat = (x_hat.data).cpu().numpy()
        if (count + 1) * BATCH_SIZE <= NUM_PATCHES:
            recon_patches[:, count * BATCH_SIZE : (count + 1) * BATCH_SIZE] = D @ x_hat
        else:
            recon_patches[:, count * BATCH_SIZE :] = D @ x_hat
        count += 1

    return recon_patches


def infer_ista(corrupt_patches, mask_patches, D, lambd, T, fista_flag=False, normalize_atoms=False):

    if fista_flag:
        # print("Computing Fista...")
        x_hat = generating.fista_inpainting(
            y=torch.from_numpy(corrupt_patches),
            D=torch.from_numpy(D),
            M=torch.from_numpy(mask_patches),
            lambd=lambd,
            L=None,
            max_itr=T,
            same_L=True,
        )
    else:
        # print("Computing Ista...")
        x_hat = generating.ista_inpainting(
            y=torch.from_numpy(corrupt_patches),
            D=torch.from_numpy(D),
            M=torch.from_numpy(mask_patches),
            lambd=lambd,
            L=None,
            max_itr=T,
            same_L=True,
        )

    x_hat = x_hat.data.cpu().numpy()
    recon_patches = D @ x_hat
    return recon_patches

def patches2image(recon_patches, name2print="ISTA"):
    # Un-normalization
    recon_patches *= np.expand_dims(mean_std, axis=-1)
    recon_patches += np.expand_dims(mean_avg, axis=-1)
    # Image from patches
    recon_image = useful_utils.patches_to_image(recon_patches, H, W, OVERLAP)
    recon_psnr = useful_utils.compute_psnr(image, recon_image)
    print("Corrupt PSNR = %.3f, %s PSNR: %.3f" % (corrupt_psnr, name2print, recon_psnr))
    return recon_image, recon_psnr


for i in range(len(image_list)):
    print(image_list[i])
    image = np.array(Image.open(data_path + "/" + image_list[i])) / 255.0
    H, W = image.shape[:2]
    mask = np.where(np.random.rand(H, W) < RATIO, 0, 1)
    # Extract corrupted patches
    corrupt_patches = useful_utils.image_to_patches(image * mask, PATCH_SIZE, OVERLAP).astype(
        np.float32
    )
    # Compute PSNR
    corrupt_psnr = useful_utils.compute_psnr(image, image * mask)
    N = np.shape(corrupt_patches)[1]
    # Mask for every patch
    mask_patches = useful_utils.image_to_patches(mask, PATCH_SIZE, OVERLAP).astype(np.float32)
    # Normalize patches
    patch_sum = np.sum(corrupt_patches, axis=-1)
    mask_sum = np.sum(mask_patches, axis=-1)
    mean_patch = np.expand_dims(patch_sum / mask_sum, axis=-1)
    mask_patches = (
        (np.expand_dims(mask_patches.T, axis=-1) * np.eye(n)).transpose(1, 2, 0).astype(np.float32)
    )
    corrupt_patches -= np.expand_dims(mean_avg, axis=-1)
    corrupt_patches /= np.expand_dims(mean_std, axis=-1)

    # Ada-LISTA Inference
    adalista_recon_patches = infer_adalista(corrupt_patches, mask_patches, D, model=model_adalista)
    adalista_image, adalista_psnr = patches2image(adalista_recon_patches, name2print="Ada-LISTA")
    
    # ISTA Inference
    ista_recon_patches = infer_ista(
        corrupt_patches, mask_patches, D, lambd, T + 1, fista_flag=False
    )
    ista_image, ista_psnr = patches2image(ista_recon_patches, name2print="ISTA")
    
    # FISTA Inference
    fista_recon_patches = infer_ista(
        corrupt_patches, mask_patches, D, lambd, T + 1, fista_flag=True
    )
    fista_image, fista_psnr = patches2image(fista_recon_patches, name2print="FISTA")
    
    # Plot figures
    fig = plt.figure(figsize=[30, 10])
    plt.subplot(1, 3, 1)
    plt.imshow(ista_image, "gray")
    plt.title("Ista Reconstruction\n PSNR = %.3f" % ista_psnr, fontsize=18)
    plt.subplot(1, 3, 2)
    plt.imshow(fista_image, "gray")
    plt.title("Fista Reconstruction\n PSNR = %.3f" % fista_psnr, fontsize=18)
    plt.subplot(1, 3, 3)
    plt.imshow(adalista_image, "gray")
    plt.title("Adalista Reconstruction\n PSNR = %.3f" % adalista_psnr, fontsize=18)
    fig.savefig(
        "./figures/inpainting/"
        + image_list[i].split('.')[0]
        + "_ratio_"
        + str(RATIO)
        + "_T_"
        + str(T)
        + "_lambd_"
        + str(lambd)
        + ".png"
    )
    plt.show()
