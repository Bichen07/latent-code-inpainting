import os
import csv
import glob

def generate_base_annotations_sorted(dataroot, output_csv_filepath, default_prompt="football field", relative_to_project_root=False):
    """
    掃描指定的 dataroot 文件夾，為其中的圖片創建一個 CSV 標註文件。
    圖片路徑會先排序，然後可以選擇輸出相對於 dataroot 的路徑，
    或者相對於 dataroot 上一級目錄（假設為專案根目錄）並以 "./dataset/" 開頭的格式。

    參數:
        dataroot (str): 包含圖像文件的根文件夾路徑。
        output_csv_filepath (str): 要保存的 CSV 標註文件的完整路徑名。
        default_prompt (str): 為每張圖片指定的預設文字提示。
        relative_to_project_root (bool): 如果為 True，則 image_path 會是相對於 dataroot 上一級目錄的
                                         類似 "./dataset/filename.jpg" 的格式。
                                         如果為 False，則是相對於 dataroot 的路徑 (通常是文件名)。
    """
    image_files_temp = []
    supported_extensions = ["*.jpg", "*.jpeg", "*.png"]

    print(f"正在掃描文件夾: {dataroot} 中的圖片...")
    for ext in supported_extensions:
        image_files_temp.extend(glob.glob(os.path.join(dataroot, ext)))
    
    if not image_files_temp:
        print(f"警告：在 {dataroot} 中沒有找到任何支持的圖片文件。")
        return

    # 對找到的圖片文件列表按完整路徑進行排序
    # 這通常能確保基於文件名的順序，例如 00000001.jpg, 00000002.jpg ...
    image_files_sorted = sorted(image_files_temp)

    print(f"找到了 {len(image_files_sorted)} 張圖片，已排序。")
    print(f"正在將標註寫入到: {output_csv_filepath} ...")

    try:
        with open(output_csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_path', 'text_prompt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            project_root_if_needed = None
            if relative_to_project_root:
                # 假設 dataroot 是 ".../project_name/dataset/"
                # 那麼 project_root_if_needed 就是 ".../project_name/"
                project_root_if_needed = os.path.abspath(os.path.join(dataroot, os.pardir))

            for img_full_path in image_files_sorted:
                if relative_to_project_root and project_root_if_needed:
                    # 計算相對於專案根目錄的相對路徑，例如 "dataset/filename.jpg"
                    path_in_csv = os.path.relpath(img_full_path, project_root_if_needed)
                    # 添加 "./" 並確保是 Unix 風格路徑分隔符
                    path_in_csv = "./" + path_in_csv.replace(os.sep, '/')
                else:
                    # 僅保存相對於 dataroot 的路徑 (通常是文件名)
                    path_in_csv = os.path.basename(img_full_path)
                
                writer.writerow({'image_path': path_in_csv,
                                 'text_prompt': default_prompt})
        
        print(f"基礎標註文件 '{output_csv_filepath}' 已成功創建。")

    except IOError:
        print(f"錯誤：無法寫入文件 {output_csv_filepath}。請檢查文件權限或路徑。")
    except Exception as e:
        print(f"創建標註文件時發生了未預期的錯誤: {e}")

if __name__ == '__main__':
    # 數據集圖片所在的【直接】文件夾路徑
    image_dataroot_actual = "/home/carlos11/Downloads/code/SecondSemester/DLP/final/latent-code-inpainting/dataset/"
    
    # CSV 標註文件的保存名稱和位置
    # output_csv_file = "football_annotations_sorted_relative_v1.csv" 
    output_csv_file = "annotations/football_annotations_v1.csv" 

    default_text_prompt = "football field"

    # 控制輸出路徑的格式：
    # 如果為 True，輸出類似 "./dataset/00000634.jpg" (假設您的 Places Dataset 的 dataroot 是專案根目錄)
    # 如果為 False，輸出類似 "00000634.jpg" (假設您的 Places Dataset 的 dataroot 就是 image_dataroot_actual)
    # 根據您之前的要求，您希望是 "./dataset/00000634.jpg" 格式
    output_relative_to_project_root = True 

    # --- 以下代碼不需要修改 ---
    output_dir = os.path.dirname(output_csv_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已創建輸出目錄: {output_dir}")

    generate_base_annotations_sorted(
        image_dataroot_actual, 
        output_csv_file, 
        default_text_prompt,
        relative_to_project_root=output_relative_to_project_root
    )