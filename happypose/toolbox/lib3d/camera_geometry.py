"""Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Standard Library
from typing import Tuple

# Third Party
import torch


def project_points(points_3d, K, TCO):
    assert K.shape[-2:] == (3, 3)
    assert TCO.shape[-2:] == (4, 4)
    batch_size = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat(
            (points_3d, torch.ones(batch_size, n_points, 1).to(device)),
            dim=-1,
        )
    P = K @ TCO[:, :3]
    suv = (P.unsqueeze(1) @ points_3d.unsqueeze(-1)).squeeze(-1)
    suv = suv / suv[..., [-1]]
    return suv[..., :2]


def project_points_robust(points_3d, K, TCO, z_min=0.1):
    assert K.shape[-2:] == (3, 3)
    assert TCO.shape[-2:] == (4, 4)
    batch_size = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat(
            (points_3d, torch.ones(batch_size, n_points, 1).to(device)),
            dim=-1,
        )
    P = K @ TCO[:, :3]
    suv = (P.unsqueeze(1) @ points_3d.unsqueeze(-1)).squeeze(-1)
    z = suv[..., -1]
    suv[..., -1] = torch.max(torch.ones_like(z) * z_min, z)
    suv = suv / suv[..., [-1]]
    return suv[..., :2]


def boxes_from_uv(uv):
    assert uv.shape[-1] == 2
    x1 = uv[..., [0]].min(dim=1)[0]
    y1 = uv[..., [1]].min(dim=1)[0]

    x2 = uv[..., [0]].max(dim=1)[0]
    y2 = uv[..., [1]].max(dim=1)[0]

    return torch.cat((x1, y1, x2, y2), dim=1)


def get_K_crop_resize(
    K: torch.Tensor,
    boxes: torch.Tensor,
    orig_size: Tuple[int, int],
    crop_resize: Tuple[int, int],
) -> torch.Tensor:
    """Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled.

    Args:
    ----
        K: (bsz, 3, 3) float
        boxes: (bsz, 4) float.
    """
    assert K.dim() == 3
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4,)
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.tensor(orig_size, dtype=torch.float)
    crop_resize = torch.tensor(crop_resize, dtype=torch.float)

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K


def cropresize_backtransform_points2d(
    input_wh,
    boxes_2d_crop,
    output_wh,
    points_2d_in_output,
):
    bsz = input_wh.shape[0]
    assert output_wh.shape == (bsz, 2)
    assert input_wh.shape == (bsz, 2)
    assert points_2d_in_output.dim() == 3

    points_2d_normalized = points_2d_in_output / output_wh.unsqueeze(1)
    points_2d = boxes_2d_crop[:, [0, 1]].unsqueeze(
        1,
    ) + points_2d_normalized * input_wh.unsqueeze(1)
    return points_2d


def get_pointcloud(depth, intrinsics, flatten=False, remove_zero_depth_points=True):
    """Projects depth image to pointcloud.

    Args:
    ----
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
        flatten: whether to flatten pointcloud

    Returns:
    -------
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)

    if flatten:
        points = np.reshape(points, [height * width, 3])
    return points
