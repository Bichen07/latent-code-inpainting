"""
A minimal text + image (+ optional depth) dataset that
reads samples from a CSV file.

Required columns in the CSV (can be renamed via args):
    - image_path : 相對於 root 的影像檔路徑
Optional columns:
    - depth_path : 相對於 root 的灰階深度圖路徑
    - text       : 任意文字描述

作者：你
"""

import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


class CsvTextImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_file: str,
                 root: str                = "",
                 img_col: str             = "image_path",
                 depth_col: str | None    = "depth_path",
                 text_col: str | None     = "text",
                 img_size: int            = 256):
        """
        Args
        ----
        csv_file :  CSV 檔完整路徑
        root     :  影像 / 深度檔所在的資料夾 (prefix)
        img_col  :  csv 欄名──影像路徑
        depth_col:  csv 欄名──深度圖路徑 (None → 不讀)
        text_col :  csv 欄名──文字描述    (None → 不讀)
        img_size :  影像最終邊長 (Resize ➜ CenterCrop)
        """
        self.df        = pd.read_csv(csv_file)
        self.root      = root
        self.img_col   = img_col
        self.depth_col = depth_col
        self.text_col  = text_col

        # transforms：RGB 與 Gray 分開各自 Normalize
        self.tf_rgb = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),                    # → (0‥1)
            transforms.Normalize([0.5]*3, [0.5]*3)    # → (-1‥1)
        ])
        self.tf_gray = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5],  [0.5])
        ])

    # --------------------------------------------------------------------- #
    #                               utilities                               #
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.df)

    def _load_img(self, path: str, grayscale: bool = False) -> torch.Tensor:
        img = Image.open(path).convert("L" if grayscale else "RGB")
        tf  = self.tf_gray if grayscale else self.tf_rgb
        return tf(img)

    # --------------------------------------------------------------------- #
    #                                main                                   #
    # --------------------------------------------------------------------- #
    def __getitem__(self, idx: int) -> dict:
        row        = self.df.iloc[idx]
        img_path   = os.path.join(self.root, str(row[self.img_col]))
        image      = self._load_img(img_path, grayscale=False)

        sample = {"image": image}

        # optional depth
        if self.depth_col is not None and pd.notna(row[self.depth_col]):
            depth_path = os.path.join(self.root, str(row[self.depth_col]))
            depth      = self._load_img(depth_path, grayscale=True)
            sample["depth"] = depth

        # optional text
        if self.text_col is not None:
            sample["text"] = str(row[self.text_col])

        return sample
