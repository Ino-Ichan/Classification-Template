import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import albumentations
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import confusion_matrix, roc_auc_score

from tqdm import tqdm
import copy
import argparse
import os, sys, yaml

sys.path.append('./')
from src.logger import setup_logger, LOGGER
from src.models import Net
from src.losses import LabelSmoothingCrossEntropy
from src.dataset import CustomDataset
from src.utils import plot_sample_images
from src.augmix import RandomAugMix
import warnings

import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


from warmup_scheduler import GradualWarmupScheduler

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


def seed_torch(seed=516):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def ousm_loss(error, k=2):
    # ousm, drop large k sample
    bs = error.shape[0]
    if len(error.shape) == 2:
        error = error.mean(1)
    _, idxs = error.topk(bs - k, largest=False)
    error = error.index_select(0, idxs)
    return error


# Freeze batchnorm 2d
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()


# change targets, https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/186492
def forward(data, model, device, criterion, mode="train"):
    inputs = data["image"].to(device)
    targets = data["target"].to(device)
    pred = model(inputs)
    pred_labels = pred.sigmoid()

    loss = criterion(pred, targets).mean()
    # loss = ousm_loss(loss, 3).mean()

    return loss, pred.detach().cpu().numpy().tolist(),\
           targets.cpu().numpy().tolist(), pred_labels.detach().cpu().numpy().tolist()


# 2x TTA, horizontal flip
def forward_test(data, model, device, criterion, mode="train"):
    inputs = data["image"].to(device)
    # print(inputs.shape)
    inputs = torch.cat([inputs, inputs.flip(-1)], 0)  # hflip
    targets = data["target"].to(device)
    pred = model(inputs)

    pred_pre = copy.deepcopy(pred)
    pred = pred.view(1, 2, -1).mean(1)

    pred_labels = pred_pre.view(1, 2, -1).mean(1).sigmoid()
    # pred_labels = pred_pre.view(1, 2, -1).sigmoid().mean(1)

    loss = criterion(pred, targets).mean()
    # loss = ousm_loss(loss, 3).mean()

    return loss, pred.detach().cpu().numpy().tolist(),\
           targets.cpu().numpy().tolist(), pred_labels.detach().cpu().numpy().tolist()

def get_train_transforms(image_size):
    return albumentations.Compose([
           albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0., rotate_limit=30, p=0.8),
           albumentations.RandomResizedCrop(image_size, image_size, scale=(0.7, 1), p=0.5),
           albumentations.HorizontalFlip(p=0.5),
           albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
           albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
           albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
           # albumentations.OneOf([
           #     albumentations.OpticalDistortion(distort_limit=1.0),
           #     albumentations.GridDistortion(num_steps=5, distort_limit=1.),
           #     albumentations.ElasticTransform(alpha=3),
           # ], p=0.2),
           albumentations.OneOf([
               albumentations.GaussNoise(var_limit=[10, 50]),
               albumentations.GaussianBlur(),
               albumentations.MotionBlur(),
               albumentations.MedianBlur(),
           ], p=0.2),
          albumentations.Resize(image_size, image_size),
          # albumentations.OneOf([
          #     albumentations.augmentations.transforms.JpegCompression(),
          #     albumentations.augmentations.transforms.Downscale(scale_min=0.1, scale_max=0.15),
          # ], p=0.2),
          # albumentations.imgaug.transforms.IAAPiecewiseAffine(p=0.2),
          # albumentations.imgaug.transforms.IAASharpen(p=0.2),
          albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=5, p=0.5),
          albumentations.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225],
          ),
          ToTensorV2(p=1)
])



def get_val_transforms(image_size):
    return albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        # albumentations.CenterCrop(image_size, image_size, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(p=1)
])


if __name__ == "__main__":
    print('Start!!!')
    warnings.simplefilter('ignore')

    parser = argparse.ArgumentParser(description="training")
    parser.add_argument('-y', '--yaml_path', type=str,
                        help='configを書いたyamlのPath。例）-y ../config/exp0001.yaml')

    args = parser.parse_args()

    yaml_path = args.yaml_path
    yaml_path = args.yaml_path
    if os.path.isfile(yaml_path):
        with open(yaml_path) as file:
            cfg = yaml.safe_load(file.read())
    else:
        print('Error: No such yaml file')
        sys.exit()
    # seed_everything
    seed_torch()

    # output
    exp_name = cfg["exp_name"]  # os.path.splitext(os.path.basename(__file__))[0]
    output_path = os.path.join("/workspace/output", exp_name)
    # path
    model_path = output_path + "/model"
    plot_path = output_path + "/plot"
    oof_path = output_path + "/oof"
    sample_img_path = output_path + "/sample_img"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path + "/log", exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(oof_path, exist_ok=True)
    os.makedirs(sample_img_path, exist_ok=True)

    # logger
    log_path = os.path.join(output_path, "log/log.txt")
    setup_logger(out_file=log_path)
    LOGGER.info("config")
    LOGGER.info(cfg)
    LOGGER.info('')

    debug = cfg["debug"]
    if debug:
        LOGGER.info("Debug!!!!!")

    # params
    device_id = cfg["device_id"]
    try:
        device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"
    except Exception as e:
        LOGGER.info('GPU is not available, {}'.format(e))
        sys.exit()

    print(device)

    #######################################
    ## params
    #######################################
    model_name = cfg["model_name"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]
    n_workers = cfg["n_workers"]
    n_epochs = cfg["n_epochs"]
    start_epoch = cfg["start_epoch"]
    transform = cfg["transform"]
    hold_out = cfg["hold_out"]
    accumulation_steps = cfg["accumulation_steps"]
    early_stopping_steps = cfg["early_stopping_steps"]
    freeze_bn = cfg["freeze_bn"]
    img_dir = cfg["img_dir"]
    target_columns = cfg["target_columns"]
    mixe_precision = cfg["mixe_precision"]

    #######################################
    ## CV
    #######################################
    df = pd.read_csv(cfg["df_train_path"])

    cv_list = hold_out if hold_out else [0, 1, 2, 3, 4]
    oof = np.zeros((len(df), len(target_columns)))
    best_eval_score_list = []

    for cv in cv_list:

        LOGGER.info('# ===============================================================================')
        LOGGER.info(f'# Start CV: {cv}')
        LOGGER.info('# ===============================================================================')

        # tensorboard
        writer = SummaryWriter(log_dir=output_path)

        df_train = df[df.cv != cv].reset_index(drop=True)
        df_val = df[df.cv == cv].reset_index(drop=True)
        val_index = df[df.cv == cv].index

        #######################################
        ## Dataset
        #######################################
        # transform
        train_transform = get_train_transforms(img_size)
        val_transform = get_val_transforms(img_size)

        train_dataset = CustomDataset(df=df_train, image_size=img_size, cols=len(target_columns),
                                      image_folder=img_dir, from_image_folder=True,
                                      transform=train_transform, mode="train")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      pin_memory=False, num_workers=n_workers, drop_last=True)
        # plot sample image
        plot_sample_images(train_dataset, sample_img_path, "train", normalize="imagenet")

        val_dataset = CustomDataset(df=df_val, image_size=img_size, cols=len(target_columns),
                                    image_folder=img_dir, from_image_folder=True,
                                    transform=val_transform, mode="train")
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    pin_memory=False, num_workers=n_workers, drop_last=False)

        plot_sample_images(val_dataset, sample_img_path, "val",  normalize="imagenet")

        # ==== INIT MODEL
        device = torch.device(device)
        model = Net(name=model_name, n_classes=len(target_columns)).to(device)
        load_checkpoint = cfg["load_checkpoint"][cv]
        LOGGER.info("-" * 10)
        if os.path.exists(load_checkpoint):
            weight = torch.load(load_checkpoint, map_location=device)
            if "exp" in load_checkpoint:
                model.load_state_dict(weight["state_dict"])
            else:
                model.load_state_dict(weight)
            LOGGER.info(f"Successfully loaded model, model path: {load_checkpoint}")
        else:
            LOGGER.info(f"Training from scratch..")
        LOGGER.info("-" * 10)

        optimizer = optim.Adam(model.parameters(), lr=1e-5, eps=1e-7)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)



        # optimizer = optim.Adam(model.parameters(), lr=(1e-4/3) / 10)
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-7)
        # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
        #                                            after_scheduler=scheduler_cosine)

        # criterion = LabelSmoothingCrossEntropy()
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        if mixe_precision:
            scaler = GradScaler()

        # ==== TRAIN LOOP

        best = -1
        best_epoch = 0
        early_stopping_cnt = 0

        for e in range(start_epoch , start_epoch + n_epochs):

            if e > 0:

                # warmup
                # scheduler_warmup.step(e - 1)
                writer.add_scalar(f"Learning Rate", optimizer.param_groups[0]["lr"], global_step=e)

                losses_train = []
                targets_list = []
                pred_list = []
                train_correct = 0

                train_time = time.time()
                LOGGER.info("")
                LOGGER.info("+" * 30)
                LOGGER.info(f"+++++  Epoch {e}")
                LOGGER.info("+" * 30)
                LOGGER.info("")
                progress_bar = tqdm(train_dataloader)

                model.train()
                torch.set_grad_enabled(True)

                # freeze batchnorm2d
                if freeze_bn:
                    model = model.apply(set_bn_eval)

                for step_train, data in enumerate(progress_bar):
                    if debug:
                        if step_train == 2:
                            break

                    if mixe_precision:
                        with autocast():
                            loss, pred, targets, pred_labels = forward(data, model, device, criterion)
                    else:
                        loss, pred, targets, pred_labels = forward(data, model, device, criterion)

                    if mixe_precision:
                        # Backward pass
                        if accumulation_steps > 1:
                            loss_bw = loss / accumulation_steps
                            scaler.scale(loss_bw).backward()
                            if (step_train + 1) % accumulation_steps == 0:
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                        else:
                            optimizer.zero_grad()
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        # Backward pass
                        if accumulation_steps > 1:
                            loss_bw = loss / accumulation_steps
                            loss.backward()
                            if (step_train + 1) % accumulation_steps == 0:
                                optimizer.step()
                                optimizer.zero_grad()
                        else:
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()

                    losses_train.append(loss.item())
                    targets_list.extend(targets)
                    pred_list.extend(pred_labels)
                    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

                mean_roc = []
                pred_list = np.array(pred_list)
                targets_list = np.array(targets_list)
                if debug:
                    targets_list = np.random.random(pred_list.shape)
                    targets_list = np.where(targets_list>0.5, 1, 0)
                for i in range(len(target_columns)):
                    mean_roc.append(roc_auc_score(targets_list[:, i], pred_list[:, i]))
                mean_roc = np.mean(mean_roc)

                LOGGER.info(f"Train loss: {np.mean(losses_train)}")
                LOGGER.info(f"Train AUC: {mean_roc}")
                LOGGER.info(f"Train time: {(time.time() - train_time) / 60:.3f} min")

                writer.add_scalar(f"Loss/train_cv{cv}", np.mean(losses_train), global_step=e)
                writer.add_scalar(f"AUC/train_cv{cv}", mean_roc, global_step=e)

            # ==== EVAL LOOP
            eval_time = time.time()
            model.eval()
            torch.set_grad_enabled(False)
            losses_eval = []
            eval_correct = 0
            pred_list = []
            targets_list = []

            oof_list = []

            progress_bar_eval = tqdm(val_dataloader)
            for step_eval, data in enumerate(progress_bar_eval):
                if debug:
                    if step_eval == 2:
                        break
                loss, pred, targets, pred_labels = forward_test(data, model, device, criterion)
                losses_eval.append(loss.item())
                pred_list.extend(pred_labels)
                targets_list.extend(targets)
                oof_list.extend(pred)
                progress_bar_eval_text = f"Running EVAL, loss: {loss.item()} loss(avg): {np.mean(losses_eval)}"
                progress_bar_eval.set_description(progress_bar_eval_text)

            # scheduler
            scheduler.step()

            each_roc = []
            targets_list = np.array(targets_list)
            pred_list = np.array(pred_list)
            if debug:
                targets_list = np.random.random(pred_list.shape)
                targets_list = np.where(targets_list>0.5, 1, 0)
            for i in range(len(target_columns)):
                each_roc.append(roc_auc_score(targets_list[:, i], pred_list[:, i]))
            mean_roc = np.mean(each_roc)

            LOGGER.info(f"Val loss: {np.mean(losses_eval)}")
            LOGGER.info(f"Val AUC: {mean_roc}")
            LOGGER.info(f"Val time: {(time.time() - eval_time) / 60:.3f} min")

            writer.add_scalar(f"Loss/eval_cv{cv}", np.mean(losses_eval), global_step=e)
            writer.add_scalar(f"AUC/eval_cv{cv}", mean_roc, global_step=e)



            LOGGER.info('Saving model ...')
            model_save_path = os.path.join(model_path, f"cv{cv}_weight_checkpoint{e}.pth")

            torch.save({
                "state_dict": model.state_dict(),
            }, model_save_path)

            if best < mean_roc:
                LOGGER.info(f'Best score update: {best:.5f} --> {mean_roc:.5f}')
                best = mean_roc
                best_epoch = e

                try:
                    if debug:
                        oof[batch_size * 2] = np.array(oof_list)
                    else:
                        oof[val_index] = np.array(oof_list)

                except Exception as error:
                    LOGGER.info(error)

                early_stopping_cnt = 0
            else:
                # early stopping
                early_stopping_cnt += 1
                if early_stopping_cnt >= early_stopping_steps:
                    LOGGER.info(f"Early stopping at Epoch {e}")
                    break

            LOGGER.info('-' * 20)
            LOGGER.info(f'Best val score: {best}, at epoch {best_epoch} cv{cv}')
            LOGGER.info('-' * 20)

        best_eval_score_list.append(best)
        writer.close()

    #######################################
    ## Save oof
    #######################################
    np.save(os.path.join(oof_path, "oof"), oof)
    LOGGER.info(f'Mean of best auc score: {np.mean(best_eval_score_list)}')

    pred_list = oof
    targets_list = df.loc[:, cols].values
    each_roc = []
    if debug:
        targets_list = np.random.random(pred_list.shape)
        targets_list = np.where(targets_list > 0.5, 1, 0)
    for i in range(len(target_columns)):
        each_roc.append(roc_auc_score(targets_list[:, i], pred_list[:, i]))
    mean_roc = np.mean(each_roc)

    LOGGER.info('-' * 20)
    LOGGER.info(f'Oof score: {mean_roc}')
    LOGGER.info('-' * 20)

