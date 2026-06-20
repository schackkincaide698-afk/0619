'''python build_points_npy.py \
  --data_root /hdd/u202520081000126/WHAM/preprocess/yt_videos \
  --label_file /hdd/u202520081000126/WHAM/preprocess/yt_videos/b.pkl \
  --depth_root depth_b \
  --semantic_root semantic_b \
  --points_root points/b \
  --min_timestamp 10 \
  --use_partial \
  --num_timestamp 60 \
  --sample_interval 30 \
  --sample_start_idx 0
'''
import argparse
import copy
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pedgen.utils.rot import depth_to_3d


def build_intrinsics(width: int, height: int) -> np.ndarray:
    f = (width**2 + height**2) ** 0.5
    cx = 0.5 * width
    cy = 0.5 * height
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = f
    intrinsics[1, 1] = f
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    return intrinsics

def parse_clip_and_frame(image_field: str):
    """
    只从 val["image"] 字符串中解析 clip 名和帧号。
    不读取 image 文件。

    输入示例:
        AyIRug0Jha8_97/000001.jpg

    输出:
        clip_name = AyIRug0Jha8_97
        frame_id = 1
    """
    p = Path(str(image_field).replace("\\", "/"))
    clip_name = p.parent.name
    frame_id = int(p.stem)
    return clip_name, frame_id

def build_train_label_list(
    labels,
    data_root: str,
    # img_root: str,
    depth_root: str,
    semantic_root: str,
    points_root: str,
    min_timestamp: int,
    use_partial: bool,
    num_timestamp: int,
    sample_interval: int,
    sample_start_idx: int,
):
    label_list = []
    for idx, val in enumerate(labels):
        if np.isnan(val["global_trans"]).any() or np.isnan(val["local_trans"]).any():
            continue

        # image_path = os.path.join(data_root, img_root, val["image"])
        # 注意：这里不是读取 image 文件，只是使用 a.pkl 里的 image 字符串
        clip_name, start_frame_id = parse_clip_and_frame(val["image"])

        i = sample_start_idx
        if use_partial:
            max_i = val["global_trans"].shape[0] - min_timestamp + 1
        else:
            max_i = val["global_trans"].shape[0] - num_timestamp + 1

        while i < max_i:
            frame_id = start_frame_id + i
            frame_name = str(frame_id).zfill(6)

            depth_path = os.path.join(data_root, depth_root, clip_name, frame_name + ".png")
            semantic_path = os.path.join(data_root, semantic_root, clip_name, frame_name + ".png")
            points_path = os.path.join(data_root, points_root, clip_name, frame_name + ".npy")

            new_val = copy.deepcopy(val)
            new_val["start_t"] = i
            new_val["depth"] = depth_path
            new_val["semantic"] = semantic_path
            new_val["points"] = points_path
            new_val["index"] = idx

            label_list.append(new_val)
            i += sample_interval

    return label_list


def generate_points_from_depth_semantic(depth_path: str, semantic_path: str) -> np.ndarray:
    depth = cv2.imread(depth_path, -1)
    if depth is None:
        raise FileNotFoundError(f"Missing depth file: {depth_path}")

    semantic = cv2.imread(semantic_path, -1)
    if semantic is None:
        raise FileNotFoundError(f"Missing semantic file: {semantic_path}")

    depth = np.array(depth, dtype=np.float32) / 256.0
    intrinsics = build_intrinsics(depth.shape[1], depth.shape[0])
    depth_3d = depth_to_3d(depth, intrinsics).astype(np.float32)
    semantic_raw = np.array(semantic, dtype=np.float32)[..., None]
    points = np.concatenate([depth_3d, semantic_raw], axis=-1)
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--label_file", type=str, required=True)
    # parser.add_argument("--img_root", type=str, default="images")
    parser.add_argument("--depth_root", type=str, default="depth")
    parser.add_argument("--semantic_root", type=str, default="semantic")
    parser.add_argument("--points_root", type=str, default="points")

    parser.add_argument("--min_timestamp", type=int, required=True)
    parser.add_argument("--use_partial", action="store_true")
    parser.add_argument("--num_timestamp", type=int, required=True)
    parser.add_argument("--sample_interval", type=int, required=True)
    parser.add_argument("--sample_start_idx", type=int, required=True)

    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.data_root, args.label_file), "rb") as f:
        labels = pickle.load(f)

    label_list = build_train_label_list(
        labels=labels,
        data_root=args.data_root,
        depth_root=args.depth_root,
        semantic_root=args.semantic_root,
        points_root=args.points_root,
        min_timestamp=args.min_timestamp,
        use_partial=args.use_partial,
        num_timestamp=args.num_timestamp,
        sample_interval=args.sample_interval,
        sample_start_idx=args.sample_start_idx,
    )

    generated = 0
    skipped_exists = 0
    skipped_missing_depth = 0
    skipped_missing_semantic = 0

    for label in tqdm(label_list, desc="Generate train .npy points"):
        points_path = Path(label["points"])

        if points_path.exists() and not args.overwrite:
            skipped_exists += 1
            continue

        if not Path(label["depth"]).exists():
            skipped_missing_depth += 1
            continue

        if not Path(label["semantic"]).exists():
            skipped_missing_semantic += 1
            continue

        points = generate_points_from_depth_semantic(
            label["depth"],
            label["semantic"],
        )

        points_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(points_path), points)

        generated += 1

    print(
        f"Done. total_labels={len(label_list)} "
        f"generated={generated} "
        f"skipped_exists={skipped_exists} "
        f"skipped_missing_depth={skipped_missing_depth} "
        f"skipped_missing_semantic={skipped_missing_semantic}"
    )


if __name__ == "__main__":
    main()