import matplotlib.pyplot as plt
import numpy as np
import os
from .fmix import fmix_creater
from .mixup import mixup_creater


def plot_sample_images(dataset, save_path, name=None, normalize=None, n_img=16):
    plt.figure(figsize=(16, 16))
    for i in range(n_img):
        data = dataset[i]
        if normalize == "imagenet":
            img = data["image"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) \
                  + np.array([0.485, 0.456, 0.406])
        else:
            img = data["image"].numpy().transpose(1, 2, 0)
        label = data["target"]
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(f"label: {label}")
    plt.tight_layout()
    if name:
        if ".png" in name:
            plt.savefig(os.path.join(save_path, f"{str(name)}.png"))
        else:
            plt.savefig(os.path.join(save_path, str(name)))
    else:
        plt.savefig(os.path.join(save_path, "sample_image.png"))
    plt.close()


def plot_sample_images_fmix(dataloader, save_path, name=None, normalize=None, n_img=16):
    plt.figure(figsize=(16, 16))
    for i, data in enumerate(dataloader):
        if i == n_img:
            break
        data, target1, target2, lam = fmix_creater(data["image"], data["target"])
        data, target1, target2, lam = data[0], target1[0], target2[0], lam
        if normalize == "imagenet":
            img = data.numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) \
                  + np.array([0.485, 0.456, 0.406])
        else:
            img = data.numpy().transpose(1, 2, 0)
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(f"target1: {target1}, target2: {target2}, lam: {lam}")
    plt.tight_layout()
    if name:
        if ".png" in name:
            plt.savefig(os.path.join(save_path, f"{str(name)}.png"))
        else:
            plt.savefig(os.path.join(save_path, str(name)))
    else:
        plt.savefig(os.path.join(save_path, "sample_image.png"))
    plt.close()


def plot_sample_images_mixup(dataloader, save_path, name=None, normalize=None, n_img=16):
    plt.figure(figsize=(16, 16))
    for i, data in enumerate(dataloader):
        if i == n_img:
            break
        data, target1, target2, lam = mixup_creater(data["image"], data["target"])
        data, target1, target2, lam = data[0], target1[0], target2[0], lam
        if normalize == "imagenet":
            img = data.numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) \
                  + np.array([0.485, 0.456, 0.406])
        else:
            img = data.numpy().transpose(1, 2, 0)
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(f"target1: {target1}, target2: {target2}, lam: {lam}")
    plt.tight_layout()
    if name:
        if ".png" in name:
            plt.savefig(os.path.join(save_path, f"{str(name)}.png"))
        else:
            plt.savefig(os.path.join(save_path, str(name)))
    else:
        plt.savefig(os.path.join(save_path, "sample_image.png"))
    plt.close()
