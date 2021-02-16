import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self,
                 df,
                 image_size,
                 image_folder,
                 from_image_folder=False,
                 transform=None,
                 mode="train",
                 cols=None
                 ):
        """
        df: Dataframe
        image_size: image_size
        image_folder: path to image folder
        from_image_folder:if alse, DataFrame should have 'img_path' columns
        transform: Transform
        mode: if "train", return labels
        cols: Columns name of Target in the Dataframe
        """

        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.image_folder = image_folder
        self.from_image_folder = from_image_folder
        self.transform = transform

        self.mode = mode

        self.cols = []

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        if self.from_image_folder:
            img_path = os.path.join(self.image_folder, row["img_name"] + ".jpg")
        else:
            img_path = row["img_path"]
        images = cv2.imread(img_path).astype(np.float32)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)


        if self.transform is not None:
            images = self.transform(image=images.astype(np.uint8))['image']
        else:
            images = images.transpose(2, 0, 1)

        if self.mode == "train":
            label = row[self.cols].values.astype(np.float16)
            return {
                # "image": torch.tensor(images, dtype=torch.float),
                "image": images,
                "target": torch.tensor(label, dtype=torch.float)
            }
        else:
            return {
                "image": torch.tensor(images, dtype=torch.float)
            }

