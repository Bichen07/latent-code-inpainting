# test_dataset.py (放在專案根目錄)
from core.data.places365 import Places
from torch.utils.data import DataLoader
import yaml
from omegaconf import OmegaConf

if __name__ == '__main__':
    # 載入你的 Transformer 訓練配置文件，以獲取 data 部分的配置
    # 確保這個 YAML 文件中的 dataroot 指向你的 football_field 數據集
    # 並且 batch_size 設置為 1
    config_path = "training_configs/places_transformer_semantic_football.yaml" # 替換為你的文件名 # 替換為你的文件名
    cfg = OmegaConf.load(config_path)
    
    # 實例化 dataset
    # 我們只需要 train 部分的配置
    train_data_cfg = cfg.data.params.train 
    dataset = Places(**train_data_cfg.params)

    if len(dataset) > 0:
        print(f"Dataset loaded successfully. Number of samples: {len(dataset)}")
        # 取第一個樣本看看
        sample = dataset[0]
        print("\nFirst sample content:")
        for key, value in sample.items():
            if isinstance(value, str):
                print(f"  {key}: {value}")
            else: # 通常是 numpy array
                print(f"  {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
        
        # 檢查 text_prompt 是否存在且內容合理
        if "text_prompt" in sample:
            print(f"\nText prompt for first sample: '{sample['text_prompt']}'")
        else:
            print("\nError: 'text_prompt' not found in the sample.")
            
        # 也可以嘗試用 DataLoader 加載
        # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        # first_batch = next(iter(data_loader))
        # print("\nFirst batch from DataLoader:")
        # for key, value in first_batch.items():
        #     if key == "text_prompt":
        #         print(f"  {key}: {value}")
        #     else:
        #         print(f"  {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")

    else:
        print("Dataset is empty. Check dataroot and data structure.")