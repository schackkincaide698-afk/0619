"""Utilities for building scene tokens from semantic point clouds."""
from typing import List, Tuple

import torch


WALKABLE_CLASSES = (0, 1, 9)
STATIC_OBSTACLE_CLASSES = (2, 3, 4, 5, 6, 7)
DYNAMIC_OBSTACLE_CLASSES = (11, 12, 13, 14, 15, 16, 17, 18)
LOW_PRIORITY_CLASSES = (8, 10)
SEMANTIC_GROUPS: Tuple[Tuple[int, ...], ...] = (
    WALKABLE_CLASSES,
    STATIC_OBSTACLE_CLASSES,
    DYNAMIC_OBSTACLE_CLASSES,
    LOW_PRIORITY_CLASSES,
)
SEMANTIC_GROUP_WEIGHTS = (0.55, 0.25, 0.15, 0.05)
SEMANTIC_FILL_ORDER = (0, 1, 2, 3)


def _sample_or_pad(points: torch.Tensor, num_target: int) -> torch.Tensor:
    if num_target <= 0:
        raise ValueError(f"num_target must be positive, got {num_target}")

    num_points = points.shape[0]
    if num_points == 0:
        raise ValueError("_sample_or_pad expects at least one point")

    if num_points >= num_target:
        indices = torch.linspace(
            0,
            num_points - 1,
            steps=num_target,
            device=points.device,
        ).long()
        return points[indices]

    pad_indices = torch.arange(
        num_target - num_points,
        device=points.device,
    ) % num_points
    return torch.cat([points, points[pad_indices]], dim=0)


def _semantic_mask(points: torch.Tensor, class_ids: Tuple[int, ...]) -> torch.Tensor:
    semantic = points[:, 3].long()
    mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for class_id in class_ids:
        mask = mask | (semantic == class_id)
    return mask


def _sample_indices_by_distance(
    points: torch.Tensor,
    indices: torch.Tensor,
    num_target: int,
    num_bins: int,
) -> torch.Tensor:
    if num_target <= 0 or indices.shape[0] == 0:
        return indices[:0]
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")

    distances = torch.norm(points[indices, :3], dim=-1)
    sorted_indices = indices[torch.argsort(distances)]
    if sorted_indices.shape[0] <= num_target:
        return sorted_indices

    num_bins = min(num_bins, num_target)
    bin_edges = torch.linspace(
        0,
        sorted_indices.shape[0],
        steps=num_bins + 1,
        device=points.device,
    ).long()
    base_quota = num_target // num_bins
    remainder = num_target % num_bins

    sampled_bins = []
    for bin_idx in range(num_bins):
        start = int(bin_edges[bin_idx].item())
        end = int(bin_edges[bin_idx + 1].item())
        bin_indices = sorted_indices[start:end]
        if bin_indices.shape[0] == 0:
            continue

        quota = base_quota + (1 if bin_idx < remainder else 0)
        if bin_indices.shape[0] <= quota:
            sampled_bins.append(bin_indices)
            continue

        keep = torch.linspace(
            0,
            bin_indices.shape[0] - 1,
            steps=quota,
            device=points.device,
        ).long()
        sampled_bins.append(bin_indices[keep])

    if not sampled_bins:
        return sorted_indices[:num_target]

    sampled = torch.cat(sampled_bins, dim=0)
    if sampled.shape[0] > num_target:
        return sampled[:num_target]
    return sampled


def _semantic_distance_stratified_sample(
    points: torch.Tensor,
    num_target: int,
    num_bins: int,
) -> torch.Tensor:
    quotas = [int(num_target * weight) for weight in SEMANTIC_GROUP_WEIGHTS]
    quotas[0] += num_target - sum(quotas)

    group_masks = [_semantic_mask(points, class_ids) for class_ids in SEMANTIC_GROUPS]
    selected_mask = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    selected_indices = []

    for quota, group_mask in zip(quotas, group_masks):
        candidate_indices = torch.nonzero(group_mask & ~selected_mask).flatten()
        sampled = _sample_indices_by_distance(points, candidate_indices, quota, num_bins)
        if sampled.shape[0] > 0:
            selected_mask[sampled] = True
            selected_indices.append(sampled)

    selected_count = sum(indices.shape[0] for indices in selected_indices)
    deficit = num_target - selected_count
    for group_idx in SEMANTIC_FILL_ORDER:
        if deficit <= 0:
            break
        candidate_indices = torch.nonzero(group_masks[group_idx] & ~selected_mask).flatten()
        sampled = _sample_indices_by_distance(points, candidate_indices, deficit, num_bins)
        if sampled.shape[0] > 0:
            selected_mask[sampled] = True
            selected_indices.append(sampled)
            deficit -= sampled.shape[0]

    if deficit > 0:
        candidate_indices = torch.nonzero(~selected_mask).flatten()
        sampled = _sample_indices_by_distance(points, candidate_indices, deficit, num_bins)
        if sampled.shape[0] > 0:
            selected_mask[sampled] = True
            selected_indices.append(sampled)

    if not selected_indices:
        return _sample_or_pad(points, num_target)

    sampled_points = points[torch.cat(selected_indices, dim=0)]
    distances = torch.norm(sampled_points[:, :3], dim=-1)
    sampled_points = sampled_points[torch.argsort(distances)]
    return _sample_or_pad(sampled_points, num_target)


def build_scene_tokens_from_points(
    points_4d: torch.Tensor,
    grid_size: List[float],
    scene_token_points: int,
    distance_bins: int = 4,
) -> torch.Tensor:
    """Build fixed-length raw point scene tokens without voxel aggregation.

    Args:
        points_4d: Tensor shaped [..., 4], where the last dimension is
            [x, y, z, semantic].
        grid_size: Spatial crop as [x_min, x_max, y_min, y_max, z_min, z_max].
        scene_token_points: Number of tokens to return.
        distance_bins: Number of near-to-far bins for distance-stratified
            sampling within each semantic group.

    Returns:
        Tensor shaped [scene_token_points, 4].
    """
    if scene_token_points <= 0:
        raise ValueError(f"scene_token_points must be positive, got {scene_token_points}")
    if points_4d.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got {points_4d.shape}")
    if len(grid_size) != 6:
        raise ValueError(f"grid_size must contain 6 values, got {grid_size}")

    points = points_4d.reshape(-1, 4)
    valid = torch.isfinite(points).all(dim=-1)
    valid = valid & (points[:, 2] > 1e-5)
    points = points[valid]

    if points.shape[0] > 0:
        grid = torch.tensor(grid_size, dtype=points.dtype, device=points.device)
        grid_mask = (
            (points[:, 0] >= grid[0]) &
            (points[:, 0] < grid[1]) &
            (points[:, 1] >= grid[2]) &
            (points[:, 1] < grid[3]) &
            (points[:, 2] >= grid[4]) &
            (points[:, 2] < grid[5])
        )
        points = points[grid_mask]

    if points.shape[0] == 0:
        return torch.zeros(
            (scene_token_points, 4),
            dtype=torch.float32,
            device=points_4d.device,
        )

    tokens = _semantic_distance_stratified_sample(
        points,
        scene_token_points,
        distance_bins,
    )
    return tokens.float()