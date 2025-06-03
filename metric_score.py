import torch
from pytorch_fid import fid_score
import lpips
from PIL import Image
from torchvision import transforms
import os
import itertools
import argparse

def fid(real_dir, gen_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir],
        batch_size=50,
        device=device,
        dims=2048,
        num_workers=0  # 避免 Windows multiprocessing 問題
    )

    print(f'FID Score = {fid_value:.4f}')
    return fid_value

def diversity(gen_dir):
    loss_fn = lpips.LPIPS(net='alex')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    images = []
    for filename in os.listdir(gen_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(gen_dir, filename)).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            images.append(img_tensor)

    if len(images) < 2:
        print("Not enough images to calculate diversity.")
        return None

    distances = []
    for img1, img2 in itertools.combinations(images, 2):
        d = loss_fn(img1, img2)
        distances.append(d.item())

    diversity_score = sum(distances) / len(distances)
    print(f'Diversity (LPIPS): {diversity_score:.4f}')
    return diversity_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FID and Diversity (LPIPS)')
    parser.add_argument('--real', type=str, required=True, help='Path to ground truth images folder')
    parser.add_argument('--gen', type=str, required=True, help='Path to generated images folder')

    args = parser.parse_args()

    fid(args.real, args.gen)
    diversity(args.gen)