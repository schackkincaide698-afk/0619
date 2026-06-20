"""Our custom pedestrian dataset.(读取.h5版本)"""
import copy
import json
import os
import pickle
from functools import reduce
from pathlib import Path
from typing import Dict, List, Sequence, Union

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from pedgen.utils.colors import IMG_MEAN, IMG_STD
from pedgen.utils.rot import (axis_angle_to_matrix, create_2d_grid,
                              create_occupancy_grid, depth_to_3d,
                              matrix_to_rotation_6d)
from pedgen.utils.scene_tokens import build_scene_tokens_from_points

class CityWalkersDataset(Dataset):
    """Lightning dataset for pedestrian generation."""

    def __init__(
        self,
        label_file: Union[str, Sequence[str]],
        mode: str,
        data_root: str,
        img_root: str,
        img_dim: List,
        min_timestamp: int,
        use_partial: bool,
        num_timestamp: int,
        depth_root: str,
        semantic_root: str,
        sample_interval: int,
        sample_start_idx: int,
        grid_size: List,
        grid_points: List,
        scene_token_points: int,
        mode_train_target: str,
        use_image: bool,
        use_data_augmentation: bool,
        train_percent: float,
        points_h5_root: str = "h5",
        points_h5_index_name: str = "points_h5_index.json",
        scene_token_mode: str = "raw_distance_stratified",
    ) -> None:
        labels = self.load_labels(data_root, label_file, mode)
        self.label_list = []
        self.img_mean = np.array(IMG_MEAN)
        self.img_std = np.array(IMG_STD)
        self.img_w = img_dim[1]
        self.img_h = img_dim[0]
        self.mode = mode
        self.num_timestamp = num_timestamp
        self.min_timestamp = min_timestamp  # hardcoded
        self.use_partial = use_partial
        self.grid_size = grid_size
        self.use_image = use_image
        self.grid_points = grid_points
        self.scene_token_points = scene_token_points
        self.scene_token_mode = scene_token_mode
        self.mode_train_target = mode_train_target
        self.use_data_augmentation = use_data_augmentation
        # HDF5 shard index: points_key -> shard_path
        self.points_h5_root = os.path.join(data_root, points_h5_root)
        self.points_h5_index = self.load_points_h5_index(
            data_root=data_root,
            points_h5_root=points_h5_root,
            points_h5_index_name=points_h5_index_name,
        )
        self.points_h5_files = {}

        for idx, val in enumerate(labels):
            if np.isnan(val["global_trans"]).any() or np.isnan(val["local_trans"]).any():
                continue

            image_path = os.path.join(data_root, img_root, val["image"])# 从pkl的val["image"]得到原始图像路径

            if self.mode != "pred":  # 训练/验证/测试模式
                i = sample_start_idx
                max_i = (
                    val["global_trans"].shape[0] - self.min_timestamp + 1
                    if self.use_partial and self.mode == "train"
                    else val["global_trans"].shape[0] - self.num_timestamp + 1
                )

                while i < max_i:
                    img_id = int(image_path.split("/")[-1].split(".")[0]) + i
                    new_val = copy.deepcopy(val)
                    new_val["start_t"] = i

                    new_image_path = image_path[:-10] + str(img_id).zfill(6) + ".jpg"
                    new_val["image"] = new_image_path
                    new_val["depth"] = new_image_path.replace(img_root, depth_root).replace("jpg", "png")
                    new_val["semantic"] = new_image_path.replace(img_root, semantic_root).replace("jpg", "png")

                    # points_key 规则统一：clip/frame（frame不带后缀）
                    clip_name = new_image_path.split("/")[-2]
                    frame_name = new_image_path.split("/")[-1].split(".")[0]
                    new_val["points_key"] = f"{clip_name}/{frame_name}"

                    # 只有points_key同时存在于pkl和h5 index中，样本才会进入self.label_list
                    if new_val["points_key"] not in self.points_h5_index:
                        i += sample_interval
                        continue

                    new_val["index"] = idx
                    self.label_list.append(new_val)
                    i += sample_interval

            else:  # 推理模式
                val["start_t"] = 0
                val["image"] = image_path
                val["depth"] = image_path.replace(img_root, depth_root).replace("jpg", "png")
                val["semantic"] = image_path.replace(img_root, semantic_root).replace("jpg", "png")

                # points_key 规则统一：clip/frame（frame不带后缀）
                clip_name = image_path.split("/")[-2]
                frame_name = image_path.split("/")[-1].split(".")[0]
                val["points_key"] = f"{clip_name}/{frame_name}"

                # 只按 points_key 是否在 h5 索引里筛选
                if val["points_key"] not in self.points_h5_index:
                    continue

                val["index"] = idx
                self.label_list.append(val)

        if self.mode == "train" and train_percent < 1.0:
            self.label_list = self.label_list[:int(len(self.label_list) * train_percent)]

    @staticmethod
    def _as_sequence(value: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(value, str):
            return [value]
        return list(value)

    @classmethod
    def load_labels(cls, data_root: str, label_file: Union[str, Sequence[str]],
                    mode: str) -> List[Dict]:
        label_paths = []
        root = Path(data_root)
        split_name = "train.pkl" if mode == "train" else "val.pkl"

        for label_item in cls._as_sequence(label_file):
            path = root / label_item
            if not path.is_dir():
                raise FileNotFoundError(f"label_file directory not found: {path}")

            group_split = path / split_name
            if group_split.is_file():
                label_paths.append(group_split)
            label_paths.extend(sorted(path.glob(f"*/{split_name}")))

        label_paths = sorted(dict.fromkeys(label_paths))
        if not label_paths:
            raise RuntimeError(
                f"No {split_name} files found under label directories: {label_file}"
            )

        labels = []
        skipped_empty = []
        for label_path in label_paths:
            if label_path.stat().st_size == 0:
                skipped_empty.append(label_path)
                continue

            with open(label_path, "rb") as f:
                try:
                    group_labels = pickle.load(f)
                except EOFError as exc:
                    raise RuntimeError(
                        f"Failed to read label pkl because it is empty or truncated: "
                        f"{label_path}"
                    ) from exc

            if group_labels is None:
                skipped_empty.append(label_path)
                continue

            labels.extend(group_labels)

        for label_path in skipped_empty:
            print(f"[WARN] skip empty label pkl: {label_path}")

        if not labels:
            raise RuntimeError(
                f"No labels loaded for mode={mode} from: "
                f"{[str(path) for path in label_paths]}"
            )

        return labels

    @staticmethod
    def load_points_h5_index(data_root: str, points_h5_root: str,
                             points_h5_index_name: str) -> Dict[str, str]:
        h5_root = Path(data_root) / points_h5_root
        if not h5_root.is_dir():
            raise FileNotFoundError(f"points_h5_root not found: {h5_root}")

        index_paths = sorted(h5_root.glob(f"*/{points_h5_index_name}"))

        if not index_paths:
            raise RuntimeError(
                f"No {points_h5_index_name} files found under: {h5_root}"
            )

        points_h5_index = {}
        for index_path in index_paths:
            group_h5_dir = index_path.parent
            with open(index_path, "r", encoding="utf-8") as f:
                group_index = json.load(f)

            for points_key, meta in group_index.items():
                shard_name = meta["shard"]
                shard_path = group_h5_dir / shard_name
                if points_key in points_h5_index:
                    raise RuntimeError(f"Duplicate points_key in h5 index: {points_key}")
                points_h5_index[points_key] = str(shard_path)

        return points_h5_index

    def __len__(self) -> int:
        return len(self.label_list)

    def load_points_h5(self, label):
        tt = lambda x: torch.from_numpy(x).float()
        points_key = label["points_key"]
        points_path = self.points_h5_index[points_key]
        if points_path in self.points_h5_files:
            h5f = self.points_h5_files[points_path]
        else:
            h5f = h5py.File(points_path, "r")
            self.points_h5_files[points_path] = h5f
        depth_3d = h5f[points_key][:]
        return tt(depth_3d)
    
    # 每个点是4维：[x,y,z,semantic]，直接生成固定长度scene_tokens
    def load_scene_tokens(self, label, intrinsics_old):
        depth_3d = self.load_points_h5(label)
        return build_scene_tokens_from_points(
            depth_3d,
            grid_size=self.grid_size,
            scene_token_points=self.scene_token_points,
        )

    # Diffuser分支使用的原始点云（用于occupancy_grid）
    def load_scene_points_raw(self, label, intrinsics_old):
        return self.load_points_h5(label)

    # 用于predictor的train/val/test
    def build_predictor_condition(self, data_dict: Dict, label: Dict,
                                  transl: torch.Tensor,
                                  intrinsics_old: np.ndarray) -> None:
        # valid_len = self.num_timestamp - int(data_dict["motion_mask"].sum().item())
        # valid_len = max(valid_len, 1)

        full_horizon = 150
        traj_full = transl[:full_horizon]
        if traj_full.shape[0] < full_horizon:
            traj_full = torch.cat([
                traj_full,
                torch.zeros(full_horizon - traj_full.shape[0], 3)
            ], dim=0)

        valid_horizon = min(transl.shape[0], full_horizon)
        last_valid = max(valid_horizon - 1, 0)
        data_dict["gt_init_pos"] = traj_full[0].clone()
        data_dict["gt_traj_150"] = traj_full.clone()
        data_dict["gt_traj_150_mask"] = (torch.arange(full_horizon) <= last_valid).float()
        data_dict["scene_tokens"] = self.load_scene_tokens(label, intrinsics_old)

    # 用于diffuser的train/val/test(已检查)
    def build_diffuser_condition(self, data_dict: Dict, label: Dict,
                                 transl: torch.Tensor,
                                 global_orient: torch.Tensor,
                                 start_t: int,
                                 intrinsics_old: np.ndarray) -> None:
        tt = lambda x: torch.from_numpy(x).float()
        depth_3d = self.load_scene_points_raw(label, intrinsics_old).clone()
        init_trans = transl[0]
        dx = int(init_trans[0] * (1280**2 + 720**2)**0.5 / init_trans[2] + 1280 / 2)
        dy = int(init_trans[1] * (1280**2 + 720**2)**0.5 / init_trans[2] + 720 / 2)
        dx = max(min(dx, 1279), 0)
        dy = max(min(dy, 719), 0)
        metric_depth = depth_3d[dy, dx, 2]
        depth_3d[..., :3] = depth_3d[..., :3] * init_trans[2] / metric_depth
        depth_3d[..., :3] = depth_3d[..., :3] - init_trans.unsqueeze(0).unsqueeze(0)

        if self.mode == "train" and self.use_data_augmentation:
            transl = transl - transl[[0]]
            rot = np.random.uniform(-np.pi, np.pi)
            rot_mat = torch.tensor(
                [[np.cos(rot), 0, np.sin(rot)],
                 [0, 1, 0],
                 [-np.sin(rot), 0, np.cos(rot)]],
                dtype=global_orient.dtype,
            )
            global_orient = rot_mat @ global_orient
            transl = transl @ rot_mat.T
            depth_3d[..., :3] = depth_3d[..., :3] @ rot_mat.T

        mask = torch.ones([720, 1280], dtype=torch.bool)
        bbox = label["bbox_2d"][start_t]
        mask[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])] = 0
        mask = reduce(torch.logical_and, [
            mask,
            depth_3d[..., 0] >= self.grid_size[0] + 1e-5,
            depth_3d[..., 0] < self.grid_size[1] - 1e-5,
            depth_3d[..., 1] >= self.grid_size[2] + 1e-5,
            depth_3d[..., 1] < self.grid_size[3] - 1e-5,
            depth_3d[..., 2] >= self.grid_size[4] + 1e-5,
            depth_3d[..., 2] < self.grid_size[5] - 1e-5,
        ])
        depth_3d = depth_3d.reshape(720 * 1280, -1)
        depth_3d = depth_3d[mask.flatten(), :]
        occupancy_grid = create_occupancy_grid(
            depth_3d, self.grid_size, self.grid_points)
        grid_2d = tt(create_2d_grid(
            num_points=self.grid_points, grid_size=self.grid_size))
        occupancy_grid = occupancy_grid.permute(0, 2, 1)
        occupancy_grid = torch.cat([occupancy_grid, grid_2d], dim=-1)
        occupancy_grid = occupancy_grid.reshape(
            occupancy_grid.shape[0] * occupancy_grid.shape[1], -1)
        data_dict["new_img"] = occupancy_grid

    def __getitem__(self, index: int) -> Dict:
        label = self.label_list[index]
        tt = lambda x: torch.from_numpy(x).float()
        data_dict = {}
        img_id = label["image"].split("/")
        img_id = img_id[-2] + "_" + img_id[-1][:-4] + "_" + str(label["index"])
        meta = {"source": "pedmotion", "img_id": img_id}
        data_dict["meta"] = meta

        f = (1280**2 + 720**2)**0.5
        cx = 0.5 * 1280
        cy = 0.5 * 720
        intrinsics_old = np.eye(3)
        intrinsics_old[0, 0] = f
        intrinsics_old[1, 1] = f
        intrinsics_old[0, 2] = cx
        intrinsics_old[1, 2] = cy

        start_t = label["start_t"]

        global_orient_source = axis_angle_to_matrix(tt(label["global_orient"][start_t]))
        global_orient_target = axis_angle_to_matrix(tt(label["local_orient"][start_t]))

        transl_source = tt(label["global_trans"][[start_t]])
        transl_target = tt(label["local_trans"][[start_t]])

        source_to_target_rotation = global_orient_target @ global_orient_source.T

        source_to_target_translation = (transl_target.T - source_to_target_rotation @ transl_source.T)

        transl = tt(label["global_trans"][start_t:])
        transl = source_to_target_rotation @ transl.T + source_to_target_translation
        transl = transl.T

        if self.mode_train_target == "predictor":
            if self.mode != "pred":
                self.build_predictor_condition(data_dict, label, transl, intrinsics_old)
            else:# 推理模式
                data_dict["img"] = torch.zeros([3, 720, 1280], dtype=torch.float32)
                data_dict["intrinsics"] = tt(intrinsics_old)
                data_dict["gt_init_pos"] = transl[0].clone()
                data_dict["scene_tokens"] = self.load_scene_tokens(label, intrinsics_old)
            return data_dict

        if self.use_image:
            img = torch.zeros([3, 720, 1280], dtype=torch.float32)
        else:
            rgb = cv2.imread(label["image"])
            rgb = np.array(rgb, dtype=np.float32)
            rgb_vis = copy.deepcopy(rgb)
            mean = np.float64(self.img_mean.reshape(1, -1))
            stdinv = 1 / np.float64(self.img_std.reshape(1, -1))
            cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)  # type: ignore
            cv2.subtract(rgb, mean, rgb)  # type: ignore
            cv2.multiply(rgb, stdinv, rgb)  # type: ignore
            img = tt(rgb).permute(2, 0, 1)  # 3, H, W

        global_orient = axis_angle_to_matrix(tt(label["global_orient"][start_t:, :3]))
        global_orient = source_to_target_rotation @ global_orient

        data_dict["img"] = img
        data_dict["intrinsics"] = tt(intrinsics_old)
        data_dict["global_trans"] = transl[:self.num_timestamp]
        data_dict["motion_mask"] = torch.zeros((self.num_timestamp,), dtype=torch.bool)

        data_dict["global_orient"] = matrix_to_rotation_6d(global_orient[:self.num_timestamp])
        data_dict["betas"] = torch.mean(tt(label["betas"][start_t:start_t + self.num_timestamp]),dim=0)

        data_dict["body_pose"] = matrix_to_rotation_6d(axis_angle_to_matrix(tt(label["body_pose"][start_t:start_t + self.num_timestamp]))
        ).reshape(-1, 23 * 6)

        if data_dict["global_trans"].shape[0] < self.num_timestamp:
            motion_length = data_dict["global_trans"].shape[0]
            data_dict["motion_mask"][motion_length:] = True
            data_dict["global_trans"] = torch.cat([
                data_dict["global_trans"],
                torch.zeros(self.num_timestamp - motion_length, 3)
            ], dim=0)
            data_dict["global_orient"] = torch.cat([
                data_dict["global_orient"],
                torch.zeros(self.num_timestamp - motion_length, 6)
            ], dim=0)
            data_dict["body_pose"] = torch.cat([
                data_dict["body_pose"],
                torch.zeros(self.num_timestamp - motion_length, 23 * 6)
            ], dim=0)

        if not self.use_image:
            return data_dict
        
        if self.mode != "pred":# 训练/验证/测试模式
            if self.mode_train_target == "diffuser":
                self.build_diffuser_condition(
                    data_dict,
                    label,
                    transl,
                    global_orient,
                    start_t,
                    intrinsics_old,
                )
            else:
                raise ValueError(
                    f"Unsupported mode_train_target: {self.mode_train_target}"
                )
        else:# 推理模式
            if self.mode_train_target == "diffuser":#联合推理
                data_dict["gt_init_pos"] = transl[0].clone()
                data_dict["scene_tokens"] = self.load_scene_tokens(label, intrinsics_old)
                self.build_diffuser_condition(data_dict, label, transl, global_orient, start_t, intrinsics_old)
            else:
                raise ValueError(f"Unsupported mode_train_target: {self.mode_train_target}")

        return data_dict


def collate_fn_pedmotion_predictor(data):
    img_batch = []
    intrinsics_batch = []
    meta_batch = []
    gt_init_pos_batch = []
    gt_traj_150_batch = []
    gt_traj_150_mask_batch = []
    scene_tokens_batch = []

    for data_dict in data:
        if "img" in data_dict:
            img_batch.append(data_dict["img"])
        if "intrinsics" in data_dict:
            intrinsics_batch.append(data_dict["intrinsics"])
        meta_batch.append(data_dict["meta"])
        if "gt_init_pos" in data_dict:
            gt_init_pos_batch.append(data_dict["gt_init_pos"])
        if "gt_traj_150" in data_dict:
            gt_traj_150_batch.append(data_dict["gt_traj_150"])
        if "gt_traj_150_mask" in data_dict:
            gt_traj_150_mask_batch.append(data_dict["gt_traj_150_mask"])
        if "scene_tokens" in data_dict:
            scene_tokens_batch.append(data_dict["scene_tokens"])


    ret_dict = {
        "meta": meta_batch,
        "batch_size": len(data),
        "scene_tokens": torch.stack(scene_tokens_batch),
    }
    if len(img_batch) > 0:
        ret_dict["img"] = torch.stack(img_batch)
    if len(intrinsics_batch) > 0:
        ret_dict["intrinsics"] = torch.stack(intrinsics_batch)
    if len(gt_init_pos_batch) > 0:
        ret_dict["gt_init_pos"] = torch.stack(gt_init_pos_batch)
    if len(gt_traj_150_batch) > 0:
        ret_dict["gt_traj_150"] = torch.stack(gt_traj_150_batch)
    if len(gt_traj_150_mask_batch) > 0:
        ret_dict["gt_traj_150_mask"] = torch.stack(gt_traj_150_mask_batch)
        
    return ret_dict

def collate_fn_pedmotion_diffuser(data):
    img_batch = []
    intrinsics_batch = []
    global_trans_batch = []
    betas_batch = []
    global_orient_batch = []
    body_pose_batch = []
    meta_batch = []
    motion_mask_batch = []
    new_img_batch = []

    for data_dict in data:
        img_batch.append(data_dict["img"])
        intrinsics_batch.append(data_dict["intrinsics"])
        global_trans_batch.append(data_dict["global_trans"])
        betas_batch.append(data_dict["betas"])
        global_orient_batch.append(data_dict["global_orient"])
        body_pose_batch.append(data_dict["body_pose"])
        meta_batch.append(data_dict["meta"])
        if "motion_mask" in data_dict:
            motion_mask_batch.append(data_dict["motion_mask"])
        if "new_img" in data_dict:
            new_img_batch.append(data_dict["new_img"])

    ret_dict = {
        "img": torch.stack(img_batch),
        "intrinsics": torch.stack(intrinsics_batch),
        "meta": meta_batch,
        "global_trans": torch.stack(global_trans_batch),
        "betas": torch.stack(betas_batch),
        "global_orient": torch.stack(global_orient_batch),
        "body_pose": torch.stack(body_pose_batch),
        "batch_size": len(img_batch),
    }
    if len(motion_mask_batch) > 0:
        ret_dict["motion_mask"] = torch.stack(motion_mask_batch)
    if len(new_img_batch) > 0:
        ret_dict["new_img"] = torch.stack(new_img_batch)
    return ret_dict

def collate_fn_pedmotion_predictor_pred(data):# 单独推理
    """Predict-only collate for standalone predictor inference."""
    return collate_fn_pedmotion_predictor(data)

def collate_fn_pedmotion_diffuser_pred(data):# 联合推理
    """Predict-only collate for joint predictor+diffuser inference."""
    ret_dict = collate_fn_pedmotion_predictor(data)
    new_img_batch = []
    for data_dict in data:
        if "new_img" not in data_dict:
            raise RuntimeError("new_img is required for diffuser prediction collate")
        new_img_batch.append(data_dict["new_img"])
    ret_dict["new_img"] = torch.stack(new_img_batch)
    return ret_dict