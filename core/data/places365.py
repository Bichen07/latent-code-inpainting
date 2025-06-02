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
import csv # <--- 新增導入 csv 模組

DEBUG_MODE = False

def readmask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return mask[..., None]


class Places(Dataset):
    def __init__(self, dataroot="", maskroot=None, crop_size=256, rescale=True, 
                 rescale_size=256, extension='jpg', split='train', 
                 annotations_filepath=None): # <--- 這裡已經在您的程式碼中加入了，很好！
        self.name = 'places365'
        self.ext = extension
        self.split = split 
        self.rescale = rescale
        self.rescale_size = rescale_size
        self.crop_size = crop_size
        self.dataroot = dataroot
        self.maskroot = maskroot

        # ==== 新增：加載標註文件 ====
        self.annotations = {}
        if annotations_filepath and os.path.exists(annotations_filepath):
            print(f"[Dataloader] Loading annotations from {annotations_filepath}...")
            try:
                with open(annotations_filepath, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.annotations[os.path.normpath(row['image_path'])] = row['text_prompt']
                print(f"[Dataloader] Loaded {len(self.annotations)} annotations.")
            except Exception as e:
                print(f"Error loading annotations file {annotations_filepath}: {e}")
        else:
            if annotations_filepath: # 如果提供了路徑但文件不存在
                 print(f"Warning: Annotation file '{annotations_filepath}' not found.")
            # else: # 如果根本沒提供 annotations_filepath，則不打印警告 (表示我們不打算用它)
            #    pass

        print(f"[Dataloader] Gathering data from {dataroot}...")
        self.initialize_paths()
        self.initialize_processor()

    # def initialize_paths(self):
    #     data_list = []
    #     for root, dirs, files in os.walk(self.dataroot):
    #         image_files = glob.glob(os.path.join(root, "*.jpg"))
    #         data_list += [(path, '_'.join(path.split('/')[-3:])) for path in image_files]
    #     self.data_list = data_list            

    #     if self.maskroot is not None:
    #         data_list = []
    #         for root, dirs, files in os.walk(self.maskroot):
    #             image_files = glob.glob(os.path.join(root, "*.png"))
    #             data_list += image_files
    #         self.mask_list = data_list            
    #     else:
    #         self.mask_list = None

    def initialize_paths(self): # <--- [維持] 原始的 initialize_paths 邏輯
        data_list = []
        for root, dirs, files in os.walk(self.dataroot): #
            # 原始程式碼只查找 .jpg，如果您的圖片有其他格式 (png, jpeg)，需要修改這裡
            image_files_in_current_root = glob.glob(os.path.join(root, f"*.{self.ext}")) # 使用 self.ext
            # 標準化路徑，並保持原始的 name 生成方式
            data_list += [(os.path.normpath(path), '_'.join(os.path.normpath(path).split(os.sep)[-3:])) for path in image_files_in_current_root]
        
        # 為了確保 data_list 中的路徑與 annotations 中的鍵可以匹配，建議也對這裡的路徑排序
        self.data_list = sorted(data_list, key=lambda x: x[0]) # 按路徑排序
        print(f"[Dataloader] Found {len(self.data_list)} image paths in {self.dataroot}.")


        if self.maskroot is not None: #
            mask_list_temp = [] # 使用臨時列表以進行排序
            for root, dirs, files in os.walk(self.maskroot): #
                # 原始程式碼查找 .png
                mask_files_in_current_root = glob.glob(os.path.join(root, "*.png")) #
                mask_list_temp.extend([os.path.normpath(p) for p in mask_files_in_current_root])
            self.mask_list = sorted(mask_list_temp) # 對 mask_list 也進行排序
            print(f"[Dataloader] Found {len(self.mask_list)} mask paths in {self.maskroot}.")
        else:
            self.mask_list = None #

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

    # def get(self, i):
    #     img_path, name = self.data_list[i]

    #     mask = None
    #     if self.maskroot is not None:
    #         img_name = os.path.basename(img_path)
    #         img_id = img_name[-10:-4]
    #         #################
    #         img_id = int(img_id) - 1 
    #         #################
    #         mask_path = os.path.join(self.maskroot, f"{img_id:06d}.png")
            
    #         if not os.path.exists(mask_path):
    #             mask_path = self.mask_list[i]

    #         mask = self.preprocess_mask(readmask(mask_path))

    #     image = self.preprocess_image(img_path)
    #     segmentation = image
    #     seg_path = img_path
    #     project_root = os.path.abspath(os.path.join(self.dataroot, os.pardir)) 
    #     relative_img_path_for_csv_lookup = "./" + os.path.relpath(img_path, project_root).replace(os.sep, '/')
        
    #     text_prompt = self.annotations.get(relative_img_path_for_csv_lookup, "football field") # 如果找不到，預設為 "football field"
    #     # ===================================================
                  
    #     example = {"image": image,
    #                "segmentation": segmentation,
    #                "img_path": img_path,
    #                "seg_path": seg_path,
    #                "filename_": name
    #                 }

    #     if mask is not None:
    #         example["mask"] = mask
    #         example["mask_path"] = mask_path

    #     return example

    def get(self, i):
        img_path, name = self.data_list[i] # img_path 來自 self.data_list，已經是 normpath

        mask = None
        mask_path_to_return = None 
        if self.maskroot is not None and self.mask_list: #
            img_name_base = os.path.basename(img_path) #
            mask_path_from_id = None
            try:
                img_id_str = img_name_base[-10:-4] #
                img_id_num = int(img_id_str) - 1  #
                constructed_mask_path = os.path.join(self.maskroot, f"{img_id_num:06d}.png") #
                if os.path.exists(constructed_mask_path):
                    mask_path_from_id = constructed_mask_path
            except ValueError:
                pass 

            if mask_path_from_id:
                mask_path_to_return = mask_path_from_id
            elif i < len(self.mask_list): 
                # 原始的 fallback 邏輯：mask_path = self.mask_list[i]
                # 我們保持這個 fallback，但確保路徑有效
                sequential_mask_path = self.mask_list[i]
                if os.path.exists(sequential_mask_path):
                     mask_path_to_return = sequential_mask_path
                # else:
                    # print(f"Warning: Fallback mask file not found at {sequential_mask_path}")
            
            if mask_path_to_return and os.path.exists(mask_path_to_return):
                mask = self.preprocess_mask(readmask(mask_path_to_return)) #


        image = self.preprocess_image(img_path) #
        segmentation = image 
        seg_path = img_path

        # ==== [修改] 從 self.annotations 獲取 text_prompt ====
        # 假設 CSV 中的 image_path 是 "./dataset/filename.jpg" 格式
        # 並且 self.dataroot 是 ".../latent-code-inpainting/dataset/"
        # 我們需要將 img_path (例如 ".../latent-code-inpainting/dataset/001.jpg")
        # 轉換為 CSV 中的 key ("./dataset/001.jpg")
        
        # 獲取專案根目錄 (假設是 self.dataroot 的上一級)
        project_root = os.path.abspath(os.path.join(self.dataroot, os.pardir)) 
        relative_img_path_for_csv_lookup = "./" + os.path.relpath(img_path, project_root).replace(os.sep, '/')
        
        text_prompt = self.annotations.get(relative_img_path_for_csv_lookup, "football field") # 如果找不到，預設為 "football field"
        # ===================================================
        
        example = {"image": image,
                   "segmentation": segmentation, 
                   "img_path": img_path,
                   "seg_path": seg_path,
                   "filename_": name, # filename_ 來自原始的 initialize_paths 邏輯
                   "text_prompt": text_prompt # <--- [新增] text_prompt
                    }

        if mask is not None: #
            example["mask"] = mask #
            if mask_path_to_return: 
                example["mask_path"] = mask_path_to_return # (原始碼是 mask_path)
            
        return example

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self.get(i)