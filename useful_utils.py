import numpy as np
from os.path import dirname, abspath
import os
from PIL import Image
from sklearn.decomposition import MiniBatchDictionaryLearning
import math


def print_section_seperator(sec_name="", subsec=False):
    """Print section seperator"""
    line_len = 80
    if not subsec:
        print("-" * line_len)
    n1 = round((line_len - len(sec_name)) / 2)
    n2 = line_len - n1 - len(sec_name)
    print("-" * n1 + sec_name + "-" * n2)
    if not subsec:
        print("-" * line_len)
    return


def compute_psnr(orig, estimate):
    """Compute the PSNR."""
    orig = np.reshape(orig, (-1))
    estimate = np.reshape(estimate, (-1))
    dynamic_range = 1.0
    mse_val = (1 / len(orig)) * np.sum((orig - estimate) ** 2)
    psnr_val = 10 * math.log10(dynamic_range ** 2 / mse_val)
    return psnr_val


def image_to_patches(image, patch_size=8, overlap=False, is_mask=False):
    """Extract patches from images."""
    H, W = np.shape(image)
    num_patches = (
        (H - patch_size + 1) * (W - patch_size + 1)
        if overlap
        else int(H / patch_size) * int(W / patch_size)
    )
    patches = (
        np.zeros((patch_size ** 2, patch_size ** 2, num_patches))
        if is_mask
        else np.zeros((patch_size ** 2, num_patches))
    )
    overlap_step = 1 if overlap else patch_size
    count = 0
    for i in np.arange(H - patch_size + 1, step=overlap_step):
        for j in np.arange(W - patch_size + 1, step=overlap_step):
            if is_mask:
                patches[:, :, count] = np.diag(
                    np.reshape(image[i : i + patch_size, j : j + patch_size], (-1))
                )
            else:
                patches[:, count] = np.reshape(image[i : i + patch_size, j : j + patch_size], (-1))
            count += 1
    return patches


def patches_to_image(patches, H, W, overlap=False):
    """Create image from patches."""
    image = np.zeros((H, W))
    patch_size = int(np.sqrt(np.shape(patches)[0]))
    overlap_step = 1 if overlap else patch_size
    count = 0
    dev_mask = np.zeros_like(image)
    for i in np.arange(H - patch_size + 1, step=overlap_step):
        for j in np.arange(W - patch_size + 1, step=overlap_step):
            image[i : i + patch_size, j : j + patch_size] += np.reshape(
                patches[:, count], (patch_size, patch_size)
            )
            dev_mask[i : i + patch_size, j : j + patch_size] += 1
            count += 1
    if overlap:
        image = image / dev_mask
    return image


def collect_patches_and_dict(
    data_path=None,
    patch_size=8,
    num_atoms=128,
    num_patches_train=10000,
    train_val_test_split=[0.8, 0.1, 0.1],
    out_path=None,
    remove_mean=True,
):
    """Create signals and dictionary for image inpainting experiment."""
    parent_dir = dirname(dirname(abspath(__file__)))
    if out_path is None:
        out_path = parent_dir + "/adaptive_ista/data"
    out_file_name = (
        out_path
        + "/data_"
        + str(patch_size)
        + "x"
        + str(patch_size)
        + "_N_100000"
        + "_atoms_"
        + str(num_atoms)
    )
    # Load the data
    npzfile = np.load(out_file_name + ".npz", allow_pickle=True)
    y = npzfile["y"].item()
    D = npzfile["D"]
    avg_mean = npzfile["avg_mean"]
    avg_std = npzfile["avg_std"]
    return y, D, avg_mean, avg_std
