#!/usr/bin/env python3
"""
Standalone HoVerNet inference runner for Apple Silicon (MPS) / CPU.

Designed to be invoked as a subprocess from the main server:
    conda run -n hovernet python server/hovernet_runner.py \
        --image_path /path/to/region.png \
        --output_path /path/to/results.json \
        --model_path hover_net/hovernet_fast_pannuke_type_tf2pytorch.tar \
        --type_info_path hover_net/type_info.json

Output: JSON file with structure:
{
    "status": "success",
    "nuclei_count": N,
    "type_counts": {"neoplastic": ..., ...},
    "nuclei": {
        "1": {"bbox": [...], "centroid": [...], "contour": [...], "type": 1, "type_name": "neopla"},
        ...
    }
}
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict


def get_device():
    """Select best available device: MPS > CPU (no CUDA on Mac)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path, nr_types=6, mode="fast", device=None):
    """Load HoVerNet model with Apple Silicon support."""
    if device is None:
        device = get_device()

    # Add hover_net to path so we can import its modules
    hover_net_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hover_net")
    if hover_net_dir not in sys.path:
        sys.path.insert(0, hover_net_dir)

    from models.hovernet.net_desc import create_model
    from run_utils.utils import convert_pytorch_checkpoint

    net = create_model(nr_types=nr_types, mode=mode)
    saved_state_dict = torch.load(model_path, map_location="cpu")["desc"]
    saved_state_dict = convert_pytorch_checkpoint(saved_state_dict)
    net.load_state_dict(saved_state_dict, strict=True)

    # Wrap in DataParallel for compatibility (HoVerNet accesses model.module)
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    net.eval()

    return net, device


def infer_step(batch_data, model, device):
    """Run inference on a batch of patches (Apple Silicon compatible)."""
    patch_imgs_gpu = batch_data.to(device).type(torch.float32)
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    with torch.no_grad():
        pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    return pred_output.cpu().numpy()


def prepare_patching(img, window_size, mask_size):
    """Prepare patch information for tile processing."""
    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()

    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
    return img, patch_info, [padt, padl]


def process_image(img, net, device, patch_input_shape=256, patch_output_shape=164,
                  nr_types=6, batch_size=8):
    """Process a single image through HoVerNet and return nuclei info.

    Args:
        img: RGB numpy array (H, W, 3)
        net: loaded HoVerNet model
        device: torch device
        patch_input_shape: input patch size
        patch_output_shape: output patch size
        nr_types: number of nuclei types
        batch_size: batch size for inference

    Returns:
        dict with pred_inst (instance map) and inst_info_dict (per-nucleus info)
    """
    hover_net_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hover_net")
    if hover_net_dir not in sys.path:
        sys.path.insert(0, hover_net_dir)

    from models.hovernet.post_proc import process as post_process

    src_shape = img.shape[:2]

    # Prepare patches
    padded_img, patch_info, top_corner = prepare_patching(
        img, patch_input_shape, patch_output_shape
    )

    # Extract patches and run inference
    accumulated_output = []
    patches = []
    infos = []

    for info in patch_info:
        y, x = info[0], info[1]
        patch = padded_img[y:y + patch_input_shape, x:x + patch_input_shape]
        patches.append(patch)
        infos.append(info)

    # Process in batches
    total = len(patches)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_patches = np.array(patches[batch_start:batch_end])
        batch_tensor = torch.from_numpy(batch_patches)

        batch_output = infer_step(batch_tensor, net, device)

        for i in range(batch_output.shape[0]):
            idx = batch_start + i
            accumulated_output.append((infos[idx], batch_output[i:i+1]))

        progress = min(100, int((batch_end / total) * 80))  # 0-80% for inference
        print(f"PROGRESS:{progress}", flush=True)

    # Reassemble prediction map
    accumulated_output = sorted(accumulated_output, key=lambda x: [x[0][0], x[0][1]])
    patch_infos, patch_data = zip(*accumulated_output)

    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_infos]) + 1
    nr_col = max([x[3] for x in patch_infos]) + 1
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(
        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
    )
    pred_map = np.squeeze(pred_map[:src_shape[0], :src_shape[1]])

    print("PROGRESS:85", flush=True)

    # Post-processing
    pred_inst, inst_info_dict = post_process(
        pred_map, nr_types=nr_types, return_centroids=True
    )

    print("PROGRESS:95", flush=True)

    return pred_inst, inst_info_dict


def main():
    parser = argparse.ArgumentParser(description="HoVerNet inference runner")
    parser.add_argument("--image_path", required=True, help="Path to input image (PNG/JPEG)")
    parser.add_argument("--output_path", required=True, help="Path to output JSON file")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint (.tar)")
    parser.add_argument("--type_info_path", default=None, help="Path to type_info.json")
    parser.add_argument("--nr_types", type=int, default=6, help="Number of nucleus types")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"],
                        help="Device to use")
    args = parser.parse_args()

    start_time = time.time()

    # Select device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    print(f"STATUS:Loading model on {device}...", flush=True)
    print("PROGRESS:5", flush=True)

    # Load model
    net, device = load_model(args.model_path, nr_types=args.nr_types, mode="fast", device=device)
    print("PROGRESS:15", flush=True)
    print("STATUS:Running inference...", flush=True)

    # Load image
    img = cv2.imread(args.image_path)
    if img is None:
        result = {"status": "error", "error": f"Failed to read image: {args.image_path}"}
        with open(args.output_path, "w") as f:
            json.dump(result, f)
        sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load type info
    type_info_dict = {}
    if args.type_info_path and os.path.exists(args.type_info_path):
        with open(args.type_info_path, "r") as f:
            type_info_dict = json.load(f)

    # Run inference
    pred_inst, inst_info_dict = process_image(
        img, net, device,
        nr_types=args.nr_types,
        batch_size=args.batch_size,
    )

    print("STATUS:Saving results...", flush=True)

    # Build output
    nuclei = {}
    type_counts = {}
    for inst_id, inst_info in inst_info_dict.items():
        nuc = {}
        for k, v in inst_info.items():
            if isinstance(v, np.ndarray):
                nuc[k] = v.tolist()
            else:
                nuc[k] = v

        # Map type ID to name
        type_id = nuc.get("type", 0)
        type_name = "unknown"
        if str(type_id) in type_info_dict:
            type_name = type_info_dict[str(type_id)][0]
        nuc["type_name"] = type_name

        nuclei[str(inst_id)] = nuc

        # Count types
        if type_name not in type_counts:
            type_counts[type_name] = 0
        type_counts[type_name] += 1

    elapsed = time.time() - start_time

    result = {
        "status": "success",
        "nuclei_count": len(nuclei),
        "type_counts": type_counts,
        "elapsed_seconds": round(elapsed, 2),
        "device": str(device),
        "image_shape": list(img.shape[:2]),
        "nuclei": nuclei,
    }

    with open(args.output_path, "w") as f:
        json.dump(result, f)

    print("PROGRESS:100", flush=True)
    print(f"STATUS:Complete! {len(nuclei)} nuclei detected in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
