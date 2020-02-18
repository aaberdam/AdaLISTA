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
    if not os.path.isfile(out_file_name + ".npz"):
        if data_path is None:
            data_path = parent_dir + "/BSDS500"
        image_list = sorted(os.listdir(data_path))
        num_images = len(image_list)
        patches_per_image = int(num_patches_train / (num_images * train_val_test_split[0]))
        num_train, num_val, num_test = (np.array(train_val_test_split) * num_images).astype(int)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        y = {
            "train": np.zeros((patch_size ** 2, patches_per_image * num_train)),
            "val": np.zeros((patch_size ** 2, patches_per_image * num_val)),
            "test": np.zeros((patch_size ** 2, patches_per_image * num_test)),
        }
        train_ind, val_ind, test_ind = (0, 0, 0)
        print("Collecting Patches...")
        for i in range(num_images):
            image = np.array(Image.open(data_path + "/" + image_list[i])) / 255.0
            for j in range(patches_per_image):
                h = np.random.choice(np.shape(image)[0] - patch_size)
                w = np.random.choice(np.shape(image)[1] - patch_size)
                curr_patch = np.mean(image[h : h + patch_size, w : w + patch_size, :], axis=-1)
                curr_patch = np.reshape(curr_patch, (-1))
                if i < num_train:
                    y["train"][:, train_ind] = curr_patch
                    train_ind += 1
                elif (i >= num_train) & (i < num_train + num_val):
                    y["val"][:, val_ind] = curr_patch
                    val_ind += 1
                else:
                    y["test"][:, test_ind] = curr_patch
                    test_ind += 1
        avg_mean = np.mean(y["train"], axis=1)
        avg_std = np.std(y["train"], axis=1)
        y["train"] -= np.expand_dims(avg_mean, axis=-1)
        y["train"] /= np.expand_dims(avg_std, axis=-1)
        y["val"] -= np.expand_dims(avg_mean, axis=-1)
        y["val"] /= np.expand_dims(avg_std, axis=-1)
        print("Patch Collection is complete.")
        print("Training Dictionary...")
        dl = MiniBatchDictionaryLearning(n_components=num_atoms, alpha=0.1, n_iter=35 * 20000)
        dl.fit(y["train"].T)
        D = dl.components_.T
        print("Dictionary Learning is complete.")
        np.savez(out_file_name, y=y, D=D, avg_mean=avg_mean, avg_std=avg_std)
    else:
        # Load the data
        npzfile = np.load(out_file_name + ".npz", allow_pickle=True)
        y = npzfile["y"].item()
        D = npzfile["D"]
        avg_mean = npzfile["avg_mean"]
        avg_std = npzfile["avg_std"]
    return y, D, avg_mean, avg_std


y, D, avg_mean, avg_std = collect_patches_and_dict(
    patch_size=8, num_atoms=256, num_patches_train=100000
)
