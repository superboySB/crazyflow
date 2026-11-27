import torch
import math
from isaaclab.utils.math import matrix_from_quat

@torch.jit.script
def depth_to_bev_torch(depth_images: torch.Tensor,
                       quaternions: torch.Tensor,
                       max_range: float = 4.0,
                       resolution: int = 16,
                       fov: float = 90.0) -> torch.Tensor:
    """
    Project depth images to Bird's Eye View (BEV) maps using PyTorch.
    Maps directly to robot's body frame with x-forward, y-left.

    Args:
        depth_images: Tensor of depth images [batch, height, width] or [batch, height*width]
        quaternions: Tensor of orientation quaternions [batch, 4] (w,x,y,z)
        max_range: Maximum range to consider in the BEV map
        resolution: Size of the BEV map (resolution x resolution)
        fov: Field of view in degrees

    Returns:
        bev_maps: Tensor of BEV maps [batch, resolution, resolution]
    """
    device = depth_images.device
    batch_size = depth_images.shape[0]

    if depth_images.dim() == 2:  # [batch, height*width]
        img_size = int(math.sqrt(depth_images.shape[1]))
        depth_images = depth_images.reshape(batch_size, img_size, img_size)

    height, width = depth_images.shape[1], depth_images.shape[2]

    bev_maps = torch.zeros((batch_size, resolution, resolution), dtype=torch.float32, device=device)

    # Grid parameters
    grid_center = resolution // 2
    inv_meters_per_cell = resolution / max_range
    half_fov_rad = math.radians(fov / 2.0)

    # Create viewing angles grid
    rows = torch.linspace(-half_fov_rad, half_fov_rad, height, device=device)
    cols = torch.linspace(-half_fov_rad, half_fov_rad, width, device=device)
    alpha_v, alpha_h = torch.meshgrid(rows, cols, indexing='ij')

    sin_alpha_h = torch.sin(alpha_h)
    sin_alpha_v = torch.sin(alpha_v)
    cos_alpha_h = torch.cos(alpha_h)
    cos_alpha_v = torch.cos(alpha_v)

    # Create direction vectors in camera frame
    # Use z-forward convention: x-right, y-down, z-forward
    dir_local = torch.stack([
        sin_alpha_h,               # x (right)
        sin_alpha_v,               # y (down)
        cos_alpha_h * cos_alpha_v  # z (forward)
    ], dim=-1)

    dir_local_flat = dir_local.reshape(-1, 3)
    n_pixels = dir_local_flat.shape[0]
    depth_flat = depth_images.reshape(batch_size, -1)

    valid_mask = depth_flat > 0
    if not torch.any(valid_mask):
        return bev_maps

    batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand_as(depth_flat)
    flat_batch_indices = batch_indices[valid_mask]
    flat_depths = depth_flat[valid_mask]

    pixel_indices = torch.arange(n_pixels, device=device).expand(batch_size, -1)
    flat_pixel_indices = pixel_indices[valid_mask]
    flat_dirs = dir_local_flat[flat_pixel_indices]

    # Calculate points in camera frame
    points_local = flat_depths.unsqueeze(1) * flat_dirs  # [n_valid_points, 3]

    # For BEV in body frame (x-forward, y-left):
    # - Camera frame z becomes body frame x (forward)
    # - Camera frame -x becomes body frame y (left)
    # Extract body frame coordinates
    body_x = points_local[:, 2]  # Forward = camera z
    body_y = -points_local[:, 0] # Left = -camera x

    # Map to BEV grid (x-forward is up in the grid, y-left is left in the grid)
    grid_y = (grid_center + body_y * inv_meters_per_cell).long()
    grid_x = (grid_center - body_x * inv_meters_per_cell).long()  # Flip so forward is up

    valid_bounds = (flat_depths < max_range) & \
                  (grid_x >= 0) & (grid_x < resolution) & \
                  (grid_y >= 0) & (grid_y < resolution)

    final_batch_indices = flat_batch_indices[valid_bounds]
    final_grid_x = grid_x[valid_bounds]
    final_grid_y = grid_y[valid_bounds]

    # Fill the BEV grid
    bev_flat_size = batch_size * resolution * resolution
    bev_flat_indices = (final_batch_indices * resolution * resolution +
                        final_grid_x * resolution +
                        final_grid_y)

    valid_flat_indices = (bev_flat_indices >= 0) & (bev_flat_indices < bev_flat_size)
    bev_flat_indices = bev_flat_indices[valid_flat_indices]

    ones = torch.ones(bev_flat_indices.size(0), device=device)
    bev_maps_flat = bev_maps.reshape(-1)
    bev_maps_flat.scatter_(0, bev_flat_indices, ones)

    return bev_maps.reshape(batch_size, resolution, resolution)