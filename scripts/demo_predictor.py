"""Demo script for standalone Predictor inference and visualization."""
import os

import cv2
import yaml
import numpy as np
import torch
from PIL import Image
from transformers import (SegformerFeatureExtractor,
                          SegformerForSemanticSegmentation)

from pedgen.model.predictor_model import PredictorModel
from pedgen.utils.rot import depth_to_3d
from pedgen.utils.scene_tokens import build_scene_tokens_from_points


PREDICTOR_CKPT_PATH = "experiments/predictor/with_context/ckpts/last-v3.ckpt"
PREDICTOR_CFG_PATH = "predictor_with_context.yaml"
# INIT_POS = [0.5, 0.5, 5.0]
# IMAGE_PATH = "scripts/demo_input.png"
# OUTPUT_PATH = "/home/u202520081000126/Pedgen/output/pred_goal_vis.png"
# INIT_POS = [0.0, 0.5, 1.5]
# IMAGE_PATH = "/home/u202520081000126/Pedgen/myimage/road.png"
# OUTPUT_PATH = "/home/u202520081000126/Pedgen/myimage/road_vis.png"
# INIT_POS = [-0.1, 0.5, 1.8]
# IMAGE_PATH = "/home/u202520081000126/Pedgen/myimage/zebra.png"
# OUTPUT_PATH = "/home/u202520081000126/Pedgen/myimage/zebra_vis.png"
INIT_POS = [-0.2, 0.5, 2.0]
IMAGE_PATH = "/home/u202520081000126/Pedgen/myimage/umi.jpg"
OUTPUT_PATH = "/home/u202520081000126/Pedgen/myimage/umi_vis.jpg"
# INIT_POS = [0, 0.5, 2.0]
# IMAGE_PATH = "/home/u202520081000126/Pedgen/myimage/sidewalk.jpg"
# OUTPUT_PATH = "/home/u202520081000126/Pedgen/myimage/sidewalk_vis.png"
# INIT_POS = [-0.2, 0.5, 2.0]
# IMAGE_PATH = "/home/u202520081000126/Pedgen/myimage/japan.png"
# OUTPUT_PATH = "/home/u202520081000126/Pedgen/myimage/japan_vis.png"
# INIT_POS = [-0.2, 0.5, 4.0]
# IMAGE_PATH = "/home/u202520081000126/Pedgen/myimage/street.png"
# OUTPUT_PATH = "/home/u202520081000126/Pedgen/myimage/street_vis.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_MODEL_PATH = "/home/u202520081000126/Pedgen/ZoeDepth"
SEG_MODEL_PATH = "/home/u202520081000126/Pedgen/segformer-b5-finetuned-cityscapes-1024-1024"


def build_intrinsics(width: int, height: int) -> np.ndarray:
    f = (width**2 + height**2)**0.5
    cx = 0.5 * width
    cy = 0.5 * height
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = f
    intrinsics[1, 1] = f
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    return intrinsics


def project_points(points_xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(points_xyz[:, 2], 1e-4, None)
    u = intrinsics[0, 0] * points_xyz[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_xyz[:, 1] / z + intrinsics[1, 2]
    return np.stack([u, v], axis=-1)


def main():
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(PREDICTOR_CFG_PATH, "r") as f:
        predictor_cfg = yaml.safe_load(f)

    predictor_model_conf = predictor_cfg["model"]
    grid_size = predictor_cfg["data"]["grid_size"]
    scene_token_points = predictor_cfg["data"].get("scene_token_points", 2048)

    image = Image.open(IMAGE_PATH).convert("RGB")

    # repo = "isl-org/ZoeDepth"
    # model_zoe_nk = torch.hub.load(repo, "ZoeD_NK",
    #                               pretrained=True).to(DEVICE).eval()
    # depth = model_zoe_nk.infer_pil(image)
    # depth = depth.astype(np.float32)

    # image_processor = SegformerFeatureExtractor.from_pretrained(
    #     "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    # model_seg = SegformerForSemanticSegmentation.from_pretrained(
    #     "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(DEVICE)
    model_zoe_nk = torch.hub.load(
        DEPTH_MODEL_PATH,
        "ZoeD_NK",
        source="local",
        pretrained=True,
        ).to(DEVICE).eval()
    depth = model_zoe_nk.infer_pil(image)
    depth = depth.astype(np.float32)

    image_processor = SegformerFeatureExtractor.from_pretrained(
        SEG_MODEL_PATH,
        local_files_only=True,
    )
    model_seg = SegformerForSemanticSegmentation.from_pretrained(
        SEG_MODEL_PATH,
        local_files_only=True,
    ).to(DEVICE).eval()

    inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
    pred = model_seg(**inputs)
    logits = pred.logits
    logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False)
    segmentation = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    segmentation = segmentation.astype(np.float32)

    intrinsics = build_intrinsics(image.size[0], image.size[1])
    tt = lambda x: torch.from_numpy(x).float()
    depth_3d_predictor = depth_to_3d(depth, intrinsics)
    depth_3d_predictor = tt(depth_3d_predictor)
    semantic_raw = tt(segmentation).unsqueeze(-1)
    depth_3d_predictor = torch.cat([depth_3d_predictor, semantic_raw], dim=-1)
    scene_tokens = build_scene_tokens_from_points(
        depth_3d_predictor,
        grid_size=grid_size,
        scene_token_points=scene_token_points,
    )

    predictor_batch = {
        "scene_tokens": scene_tokens.unsqueeze(0).to(DEVICE),
        "gt_init_pos": torch.Tensor(INIT_POS).unsqueeze(0).to(DEVICE),
    }

    print(f"Using predictor checkpoint: {PREDICTOR_CKPT_PATH}")
    model = PredictorModel.load_from_checkpoint(
        PREDICTOR_CKPT_PATH,
        map_location=DEVICE,
        **predictor_model_conf,
    )
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        out_dict = model.eval_step(predictor_batch)

    pred_traj_150 = out_dict["pred_traj_150"][0].detach().cpu().numpy()

    bg = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if bg is None:
        raise FileNotFoundError(f"Cannot read background image: {IMAGE_PATH}")
    h, w = bg.shape[:2]
    intrinsics = build_intrinsics(w, h)
    points_xy = project_points(pred_traj_150, intrinsics)
    visible = (
        (points_xy[:, 0] >= 0) & (points_xy[:, 0] < w) &
        (points_xy[:, 1] >= 0) & (points_xy[:, 1] < h)
    )

    vis = bg.copy()
    traj_color = (255, 200, 80)  # 浅蓝色(BGR)
    for idx in range(1, points_xy.shape[0]):
        if not (visible[idx - 1] and visible[idx]):
            continue
        p0 = (int(points_xy[idx - 1][0]), int(points_xy[idx - 1][1]))
        p1 = (int(points_xy[idx][0]), int(points_xy[idx][1]))
        cv2.line(vis, p0, p1, traj_color, 2)

    if visible[0]:
        p0 = (int(points_xy[0][0]), int(points_xy[0][1]))
        cv2.circle(vis, p0, 7, (0, 0, 255), -1)
        cv2.putText(vis, "start", (p0[0] + 8, p0[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for frame_idx in [60, 120]:
        if frame_idx < points_xy.shape[0] and visible[frame_idx]:
            pf = (int(points_xy[frame_idx][0]), int(points_xy[frame_idx][1]))
            cv2.circle(vis, pf, 6, (0, 255, 255), -1)
            cv2.putText(vis, f"f{frame_idx}", (pf[0] + 8, pf[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imwrite(OUTPUT_PATH, vis)

    print("pred_traj_150:")
    print(pred_traj_150)
    print(f"Saved visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()