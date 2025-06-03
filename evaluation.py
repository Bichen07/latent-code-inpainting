# evaluation_clean.py
import numpy as np
import glob
import os
import importlib
import yaml
import warnings
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.transform import rescale
from omegaconf import OmegaConf
from skimage.io import imread
from PIL import Image
import argparse # Import argparse for command-line arguments
from core.modules.util import box_mask, BatchRandomMask


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. Utility Functions
# ==============================================================================

# Global variables for device and target resolution (set in main)
_GLOBAL_DEVICE = None
_GLOBAL_TARGET_RES = 256

def show_and_save(original_x, mask_up, rec, filename_prefix, output_base_dir):
    """
    Saves original image with mask overlay and inpainted result to specified directory.
    """
    output_subdir = os.path.join(output_base_dir, filename_prefix)
    os.makedirs(output_subdir, exist_ok=True)

    orig_rgb_uint8 = to_img(original_x)
    mask_np = mask_up[0, 0].detach().cpu().numpy()
    hole_locations = mask_np < 0.5
    input_overlay = orig_rgb_uint8.copy()
    input_overlay[hole_locations, :] = 127

    pred_rgb_uint8 = to_img(rec)

    input_bgr = cv2.cvtColor(input_overlay, cv2.COLOR_RGB2BGR)
    pred_bgr  = cv2.cvtColor(pred_rgb_uint8,  cv2.COLOR_RGB2BGR)

    in_name  = filename_prefix + "_input.png"
    pr_name  = filename_prefix + "_pred.png"

    cv2.imwrite(os.path.join(output_subdir, in_name),  input_bgr)
    cv2.imwrite(os.path.join(output_subdir, pr_name),  pred_bgr)
    print(f"Saved {in_name} and {pr_name} to {output_subdir}")

def imshow(images, titles=None, save_path=None):
    """Displays images using matplotlib."""
    n_img = len(images)
    fig, ax = plt.subplots(1, n_img, figsize=(4*n_img, 4*n_img))

    if n_img > 1:
        for i in range(n_img):
            if titles is not None and i < len(titles):
                ax[i].set_title(titles[i])
            ax[i].axis('off')
            ax[i].imshow(images[i])
    else:
        if titles is not None:
            ax.set_title(titles[0])
        ax.axis('off')
        ax.imshow(images[0])

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def to_img(x):
    """Converts a normalized torch tensor to a uint8 numpy image."""
    x = (x.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    return x[0].detach().cpu().numpy()

def read_image(path):
    """
    Reads an RGB image, resizes to _GLOBAL_TARGET_RES, and normalizes to [-1, 1].
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((_GLOBAL_TARGET_RES, _GLOBAL_TARGET_RES), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(_GLOBAL_DEVICE)
    img_tensor = img_tensor.div(127.5).sub(1.0)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def instantiate_from_config(config):
    """Instantiates a Python object from a configuration dictionary."""
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    """Retrieves a Python object from a string path."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# ==============================================================================
# 2. Model-Specific Functions (assuming these parameters are set globally or passed)
# ==============================================================================

def forward_to_indices(model, batch, z_indices, mask, sampling_params):
    """Performs forward pass to get completed latent indices."""
    xc_output = model.get_xc(batch)
    if len(xc_output) == 3: # If text_prompts are returned
        x, c, text_prompts = xc_output
    else: # Backward compatibility
        x, c = xc_output
        text_prompts = None # No text prompts

    x = x.to(_GLOBAL_DEVICE).float()
    c = c.to(_GLOBAL_DEVICE).float()
    
    quant_c, c_indices = model.encode_to_c(c)
    mask = model.preprocess_mask(mask, z_indices)
    r_indices = torch.full_like(z_indices, model.mask_token)
    z_start_indices = mask * z_indices + (1 - mask) * r_indices      
    
    index_sample, probs, candidates = model.sample(
        z_start_indices.to(_GLOBAL_DEVICE),
        c_indices.to(_GLOBAL_DEVICE),
        text_prompts_for_sampling=text_prompts,
        sampling_ratio=sampling_params['sampling_ratio'],
        temperature=sampling_params['temperature'],
        sample=not sampling_params['deterministic'],
        temperature_degradation=sampling_params['temperature_annealing'],
        top_k=None,
        return_probs=True,
        scheduler='cosine',
    )
    return index_sample, probs, candidates

def load_main_model(config_path, ckpt_paths):
    """Loads the main InpaintingMaster model and its internal checkpoints."""
    print(f"Loading configuration from: {config_path}")
    config = OmegaConf.load(config_path)
    config['data']['params']['batch_size'] = 1 # Force batch size to 1 for evaluation

    # Instantiate the main model (e.g., InpaintingMaster)
    model = instantiate_from_config(config.model).to(_GLOBAL_DEVICE)

    # InpaintingMaster is designed to load its sub-module checkpoints via its config.
    # We should ensure the YAML points to correct ckpt_path for each sub-module.
    # The `places_inpainting.yaml` already has `ckpt_path` for VQModel, Encoder, Decoder, UNet, Transformer.
    # So, typically no *manual* loading is needed here beyond what InpaintingMaster does.
    # The following block from original `evaluation_text.py` is likely for loading the *main* Transformer checkpoint if `model` itself *is* Transformer,
    # which is not the case if `model` is `InpaintingMaster`.
    # Keeping it commented out as per the last successful run's logic.
    # if ckpt_paths["transformer"] and os.path.exists(ckpt_paths["transformer"]):
    #     print(f"Loading Transformer checkpoint from: {ckpt_paths['transformer']}")
    #     try:
    #         ckpt = torch.load(ckpt_paths["transformer"], map_location=_GLOBAL_DEVICE)
    #         if "state_dict" in ckpt:
    #             model.load_state_dict(ckpt["state_dict"], strict=False)
    #         else:
    #             model.load_state_dict(ckpt, strict=False)
    #         print("Transformer checkpoint loaded successfully.")
    #     except Exception as e:
    #         print(f"Error loading Transformer checkpoint: {e}")

    model.eval() # Set model to evaluation mode
    return model

# ==============================================================================
# 3. Main Execution Function
# ==============================================================================

def main(args):
    """Main function to run the image inpainting evaluation."""
    global _GLOBAL_DEVICE, _GLOBAL_TARGET_RES
    
    # Set global variables
    _GLOBAL_DEVICE = torch.device(f'cuda:{args.gpu_idx}')
    _GLOBAL_TARGET_RES = args.target_res

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Model Checkpoint Paths (defined here for clarity, could be passed from config if needed)
    # These are used by the YAML config, or if a model has `ckpt_path` param in its __init__.
    # In InpaintingMaster, these are primarily used within its sub-configs in the YAML.
    ckpt_paths = {
        "transformer": "./ckpts/transformer.ckpt",
        "vqgan_encoder": "./ckpts/encoder.ckpt",
        "vqgan_decoder": "./ckpts/places256_decoder.ckpt",
        "unet": "./ckpts/unet_256.ckpt",
        "vqgan_base": "./ckpts/places256_vqgan1024_BASE.ckpt"
    }

    # Inference Parameters (packed into a dictionary)
    sampling_params = {
        "clamp_ratio": args.clamp_ratio,
        "sampling_ratio": args.sampling_ratio,
        "temperature": args.temperature,
        "temperature_annealing": args.temperature_annealing,
        "deterministic": args.deterministic,
    }

    # Load the main model
    model = load_main_model(args.config_path, ckpt_paths)

    # Load input images
    input_image_list = glob.glob(args.input_image_glob)
    if not input_image_list:
        print(f"Error: No images found matching '{args.input_image_glob}'. Please check the path.")
        exit()
    print(f"Found {len(input_image_list)} images to process.")

    # --- Main Loop for Inference ---
    for idx, img_path in enumerate(input_image_list):
        print(f'Processing image {idx+1}/{len(input_image_list)}: {img_path}')
        x = read_image(img_path) # Uses _GLOBAL_DEVICE and _GLOBAL_TARGET_RES

        # Determine x_down based on resolution
        x_down = torch.nn.functional.interpolate(x, (256, 256)) if x.shape[-1] != 256 else x
        
        # Mask generation (box mask example)
        mask = box_mask(x_down.shape, _GLOBAL_DEVICE, 0.3, det=True).float()

        # Resize mask to target_res if necessary
        mask_up = torch.nn.functional.interpolate(mask, (_GLOBAL_TARGET_RES, _GLOBAL_TARGET_RES)) if _GLOBAL_TARGET_RES != mask.shape[-1] else mask

        ##### Step 5: Input execution #####
        with torch.no_grad():
            mask = torch.round(mask)

            # Get sub-modules from InpaintingMaster's helper_model (if stage is 'final')
            # These are dynamically unpacked based on InpaintingMaster's internal structure.
            VQModel, Encoder, Transformer, Unet = model.helper_model
            
            # Ensure sub-modules are on the correct device (they should be if `model` was moved)
            VQModel.to(_GLOBAL_DEVICE)
            Encoder.to(_GLOBAL_DEVICE)
            Transformer.to(_GLOBAL_DEVICE)
            if Unet is not None:
                Unet.to(_GLOBAL_DEVICE)

            # Encode original image with mask
            quant_z, _, info, mask_out = Encoder.encode(x_down * mask, mask, clamp_ratio=sampling_params['clamp_ratio'])
            mask_out = mask_out.reshape(x.shape[0], -1)
            z_indices = info[2].reshape(x.shape[0], -1)
            new_batch = {'image': (x_down * mask).permute(0, 2, 3, 1)}

            # Perform iterative sampling with the Transformer
            z_indices_complete, probs, candidates = forward_to_indices(
                Transformer, new_batch, z_indices, mask_out, sampling_params
            )

            B, C, H, W = quant_z.shape
            quant_z_complete = VQModel.quantize.get_codebook_entry(
                z_indices_complete.reshape(-1).int(), shape=(B, H, W, C)
            )

            # Decode and refine using the main model's current_model (which is the Decoder)
            dec, _, mout, f0, f1 = model.current_model(
                new_batch,
                quant=quant_z_complete,
                mask_in=mask,
                mask_out=mask_out.reshape(B, 1, H, W),
                return_fstg=False,
                debug=True,
            )
            rec = x_down * mask + dec * (1 - mask)

            if Unet is not None:
                rec = Unet.refine(rec, None, recomp=False)
                rec = x * mask_up + rec * (1 - mask_up)

        # Display and save results
        imshow([to_img(x), to_img(rec)], titles=['Original', 'Prediction'])
        output_prefix = os.path.splitext(os.path.basename(img_path))[0]
        show_and_save(x, mask_up, rec, output_prefix, args.output_folder)

    print("Evaluation complete.")

# ==============================================================================
# 4. Command-Line Argument Parsing
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image inpainting evaluation.")

    # --- Model and System Parameters ---
    parser.add_argument("--config_path", type=str, default="configs/places_inpainting.yaml",
                        help="Path to the model configuration YAML file.")
    parser.add_argument("--target_res", type=int, default=256,
                        help="Target image resolution for processing.")
    parser.add_argument("--gpu_idx", type=str, default="0",
                        help="GPU index to use (e.g., '0', '1').")

    # --- Inference Parameters ---
    parser.add_argument("--clamp_ratio", type=float, default=0.5,
                        help="Clamp ratio for the restrictive encoder.")
    parser.add_argument("--sampling_ratio", type=float, default=0.2,
                        help="Sampling ratio for iterative token predictions.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature scaling before logits softmax.")
    parser.add_argument("--temperature_annealing", type=float, default=0.9,
                        help="Annealing factor applied to the temperature in each iteration.")
    parser.add_argument("--deterministic", action="store_true",
                        help="If set, generates deterministic results (no random sampling).")

    # --- Input/Output Paths ---
    parser.add_argument("--input_image_glob", type=str, default="dataset/val/*.jpg",
                        help="Glob pattern for input images (e.g., 'dataset/val/*.jpg').")
    parser.add_argument("--output_folder", type=str, default="my_inpainting_results/base",
                        help="Folder to save output images.")

    args = parser.parse_args()
    main(args)