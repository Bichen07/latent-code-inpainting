import os
import numpy as np
import albumentations
import cv2
import glob
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from core.modules.util import write_images, box_mask, RandomMask
from core.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from PIL import Image

DEBUG_MODE = False

def readmask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return mask[..., None]


class Places(Dataset):
    def __init__(self, dataroot="", maskroot=None, crop_size=256, rescale=True, rescale_size=256, extension='jpg', split='train'):
        self.name = 'places365'
        self.ext = extension
        self.split = split 
        self.rescale = rescale
        self.rescale_size = rescale_size
        self.crop_size = crop_size
        self.dataroot = dataroot
        self.maskroot = maskroot
        print(f"[Dataloader] Gathering data from {dataroot}...")
        self.initialize_paths()
        self.initialize_processor()

    def initialize_paths(self):
        data_list = []
        for root, dirs, files in os.walk(self.dataroot):
            image_files = glob.glob(os.path.join(root, "*.jpg"))
            data_list += [(path, '_'.join(path.split('/')[-3:])) for path in image_files]
        self.data_list = data_list            

        if self.maskroot is not None:
            data_list = []
            for root, dirs, files in os.walk(self.maskroot):
                image_files = glob.glob(os.path.join(root, "*.png"))
                data_list += image_files
            self.mask_list = data_list            
        else:
            self.mask_list = None

    def initialize_processor(self):
        if self.split != "train":
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            self.hflipper = albumentations.HorizontalFlip(p=0.0)
        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.hflipper = albumentations.HorizontalFlip(p=0.5)
        self.safety_rescaler = albumentations.SmallestMaxSize(max_size=self.crop_size)
        
        if self.rescale:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.rescale_size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper, self.hflipper])
        else:
            self.preprocessor = albumentations.Compose([self.cropper, self.hflipper])

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # ---------------------------------------------------
        # handle cases where image has smaller size than the crop size
        h,w = image.shape[:2]
        if min(h,w) < self.crop_size:
            image = self.safety_rescaler(image=image)
            image = image['image']
        # ---------------------------------------------------

        processed = self.preprocessor(image=image)
        image = processed["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def preprocess_mask(self, mask):
        image = mask
        # ---------------------------------------------------
        # handle cases where image has smaller size than the crop size
        h,w = image.shape[:2]
        if min(h,w) < self.crop_size:
            image = self.safety_rescaler(image=image)
            image = image['image']
        # ---------------------------------------------------

        processed = self.preprocessor(image=image)
        image = processed["image"]
        return np.round(image)

    def get(self, i):
        img_path, name = self.data_list[i]

        mask = None
        if self.maskroot is not None:
            img_name = os.path.basename(img_path)
            img_id = img_name[-10:-4]
            #################
            img_id = int(img_id) - 1 
            #################
            mask_path = os.path.join(self.maskroot, f"{img_id:06d}.png")
            
            if not os.path.exists(mask_path):
                mask_path = self.mask_list[i]

            mask = self.preprocess_mask(readmask(mask_path))

        image = self.preprocess_image(img_path)
        segmentation = image
        seg_path = img_path

        # ==== 新增：獲取文字提示 ====
        text_prompt = "an image" # 預設提示，以防提取失敗
        try:
            # 從 img_path 提取類別名
            # 假設您的 dataroot 是 /home/carlos11/.../dataset/
            # 且 img_path 是 /home/carlos11/.../dataset/football_field/xxxx.jpg
            # 那麼 relative_path 將是 football_field/xxxx.jpg
            relative_path = os.path.relpath(img_path, self.dataroot)
            # category_name 將是 football_field
            category_name = relative_path.split(os.sep)[0] 
            text_prompt = category_name.replace("_", " ") # 將 "football_field" 轉換為 "football field"
        except Exception as e:
            # 如果您的 dataroot 直接就是 football_field 文件夾，上面的邏輯會出錯
            # 因為 relative_path 就直接是 xxxx.jpg, split後取第一個就是文件名
            # 這種情況下，您可以直接硬編碼，或者從 name 變數想辦法
            # 例如，如果您的 dataroot 直接指向 /home/carlos11/.../dataset/football_field/
            # 那麼可以簡單地：
            if "football_field" in self.dataroot: # 簡單判斷
                 text_prompt = "football field"
            else:
                 print(f"Warning: Could not reliably extract category name for {img_path} from dataroot '{self.dataroot}'. Using default prompt. Error: {e}")
                 # 或者，如果 name 變數包含類別信息，嘗試從 name 提取
                 # if 'football_field' in name.lower():
                 #    text_prompt = "football field"
                 
        example = {"image": image,
                   "segmentation": segmentation,
                   "img_path": img_path,
                   "seg_path": seg_path,
                   "filename_": name
                    }

        if mask is not None:
            example["mask"] = mask
            example["mask_path"] = mask_path

        return example

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.get(i)