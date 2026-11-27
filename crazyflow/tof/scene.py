"""Seeded obstacle maps and analytical solid geometry for batched ToF sensing.

The primitive sizes and Poisson separation match CrazyRL's map generator. Mesh
triangles and a CPU KD-tree are replaced by exact primitive intersections and
surface distance queries, so training never copies a batch to the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from crazyflow.tof.config import EnvConfig


@dataclass
class Scene:
    centers: jax.Array
    half_sizes: jax.Array
    kinds: jax.Array
    rotations: jax.Array
    free_points: jax.Array
    counts: np.ndarray
    room_half_size: float = 11.0
    floor: float = 0.01
    ceiling: float = 2.0
    free_counts: jax.Array | None = None


def _poisson(rng: np.random.Generator, size: float, separation: float, count: int) -> np.ndarray:
    """Bridson's 30-attempt Poisson disk sampler, as in CrazyRL."""
    points = [rng.uniform(0, size, 2)]
    active = [0]
    while active and len(points) < count:
        slot = int(rng.integers(len(active)))
        origin = points[active[slot]]
        found = False
        for _ in range(30):
            angle = rng.uniform(0, 2 * np.pi)
            point = origin + rng.uniform(separation, separation * 2) * np.array(
                [np.cos(angle), np.sin(angle)]
            )
            if np.any(point < 0) or np.any(point >= size):
                continue
            if np.min(np.linalg.norm(np.array(points) - point, axis=1)) < separation:
                continue
            points.append(point)
            active.append(len(points) - 1)
            found = True
            break
        if not found:
            active.pop(slot)
    return np.array(points) - size / 2


def _numpy_distance(
    points: np.ndarray,
    centers: np.ndarray,
    sizes: np.ndarray,
    kinds: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    local = np.einsum("kji,pkj->pki", rotations, points[:, None] - centers[None])
    q = np.abs(local) - sizes
    box = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)
    radial = np.linalg.norm(local[..., :2], axis=-1)
    cylq = np.stack([radial - sizes[:, 0], np.abs(local[..., 2]) - sizes[:, 2]], -1)
    cyl = np.linalg.norm(np.maximum(cylq, 0), axis=-1) + np.minimum(np.max(cylq, axis=-1), 0)
    radius, height = sizes[:, 0], sizes[:, 2]
    rz = np.stack([radial, local[..., 2]], -1)
    apex = np.stack([np.zeros_like(height), height], -1)
    edge = np.stack([radius, -2 * height], -1)
    t = np.clip(np.sum((rz - apex) * edge, -1) / np.sum(edge**2, -1), 0, 1)
    side = apex + t[..., None] * edge
    base = np.stack([np.minimum(radial, radius), np.broadcast_to(-height, radial.shape)], -1)
    cone_d = np.minimum(np.linalg.norm(rz - side, axis=-1), np.linalg.norm(rz - base, axis=-1))
    cone_inside = (np.abs(local[..., 2]) <= height) & (
        radial <= radius * (height - local[..., 2]) / (2 * height)
    )
    cone_d *= np.where(cone_inside, -1, 1)
    return np.min(np.where(kinds[None] == 0, box, np.where(kinds[None] == 1, cyl, cone_d)), axis=-1)


def build_scene(cfg: EnvConfig, device: jax.Device) -> Scene:
    """Generate seeded CPU maps and transfer their compact geometry to the device."""
    rng = np.random.default_rng(cfg.map_seed)
    maps = []
    for _ in range(cfg.num_maps):
        xy = _poisson(rng, cfg.map_size, cfg.obstacle_size_max + cfg.obstacle_min_distance, 1024)
        centers, sizes, kinds, rotations = [], [], [], []
        for x, y in xy:
            width, length = rng.uniform(cfg.obstacle_size_min, cfg.obstacle_size_max, 2)
            height = rng.uniform(2.4, 2.5)
            kind = int(rng.integers(2))
            radius = (width + length) / 4
            centers.append([x, y, height / 2])
            sizes.append(
                [width / 2, length / 2, height / 2] if kind == 0 else [radius, radius, height / 2]
            )
            kinds.append(kind)
            rotations.append(np.eye(3))
        column_count = len(centers)
        # CrazyRL's stratified 0.8-jitter sampler, including its neighboring-cell
        # distance check. Its requested 256 floaters normally yields far fewer.
        floaters, grid = [], {}
        low = np.array([-cfg.map_size / 2, -cfg.map_size / 2, 0.3])
        high = np.array([cfg.map_size / 2, cfg.map_size / 2, 1.7])
        target = int(cfg.map_size**2)
        cells_per_dim = max(3, int(np.ceil(target ** (1 / 3) * 1.5)))
        cell_width = (high - low) / cells_per_dim
        cells = (
            np.array(np.meshgrid(*([np.arange(cells_per_dim)] * 3), indexing="ij")).reshape(3, -1).T
        )
        rng.shuffle(cells)

        def add(point: np.ndarray) -> None:
            cell = tuple((point / cfg.obstacle_min_distance).astype(int))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbors = grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), [])
                        if any(
                            np.linalg.norm(point - other) < cfg.obstacle_min_distance * 2
                            for other in neighbors
                        ):
                            return
            floaters.append(point)
            grid.setdefault(cell, []).append(point)

        for cell in cells:
            point = low + (cell + 0.5) * cell_width + rng.uniform(-0.8, 0.8, 3) * cell_width * 0.5
            add(point)
            if len(floaters) >= target:
                break
        for _ in range((target - len(floaters)) * 5):
            if len(floaters) >= target:
                break
            add(rng.uniform(low, high))
        for point in floaters:
            size = rng.uniform(cfg.floater_size_min, cfg.floater_size_max)
            kind = int(rng.integers(3))
            centers.append(point)
            sizes.append([size / 2, size / 4, size / 4] if kind == 0 else [size / 2] * 3)
            kinds.append(kind)
            rotations.append(Rotation.random(random_state=rng).as_matrix())
        arrays = [np.asarray(x, dtype=np.float32) for x in (centers, sizes, rotations)]
        centers, sizes, rotations = arrays
        kinds = np.asarray(kinds, dtype=np.int32)
        # The reference chooses free points on a 0.2m grid, x/y one meter
        # inside the obstacle region and z in [0.5,1.5), before 0.15m goal noise.
        xy_axis = np.arange(-cfg.map_size / 2 + 1, cfg.map_size / 2 - 1, 0.2)
        z_axis = np.arange(0.5, 1.5, 0.2)
        candidates = np.stack(np.meshgrid(xy_axis, xy_axis, z_axis, indexing="xy"), -1).reshape(
            -1, 3
        )
        candidates = candidates[
            rng.choice(len(candidates), min(len(candidates), 20000), replace=False)
        ]
        distances = _numpy_distance(candidates, centers, sizes, kinds, rotations)
        free_indices = np.flatnonzero(distances >= 0)
        if len(free_indices) > 10000:
            free_indices = rng.choice(free_indices, 10000, replace=False)
        free = candidates[free_indices[distances[free_indices] >= cfg.spawn_clearance]]
        if not len(free):
            raise ValueError("Map generation produced no legal spawn points")
        free_count = len(free)
        padded_free = np.zeros((10000, 3), dtype=np.float32)
        padded_free[:free_count] = free
        maps.append(
            (
                centers,
                sizes,
                kinds,
                rotations,
                padded_free,
                (column_count, len(floaters)),
                free_count,
            )
        )
    capacity = max(len(m[0]) for m in maps)

    # A fixed object axis compiles once for every map in the bank.
    def pad(array: np.ndarray, value: float) -> np.ndarray:
        result = np.full((capacity, *array.shape[1:]), value, array.dtype)
        result[: len(array)] = array
        return result

    centers = np.stack([pad(m[0], 1e5) for m in maps])
    sizes = np.stack([pad(m[1], 0.01) for m in maps])
    kinds = np.stack([pad(m[2], 0) for m in maps])
    rotations = np.stack([pad(m[3], 0) for m in maps])
    for mi, m in enumerate(maps):
        rotations[mi, len(m[0]) :] = np.eye(3)

    def put(x: np.ndarray) -> jax.Array:
        return jax.device_put(jnp.asarray(x), device)

    return Scene(
        put(centers),
        put(sizes),
        put(kinds),
        put(rotations),
        put(np.stack([m[4] for m in maps]).astype(np.float32)),
        np.array([m[5] for m in maps]),
        free_counts=put(np.array([m[6] for m in maps], np.int32)),
    )


def primitive_distances(
    local: jax.Array, size: jax.Array, kind: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Signed distance and vector to the nearest surface of box/cylinder/cone."""
    q = jnp.abs(local) - size
    box_distance = jnp.linalg.norm(jnp.maximum(q, 0), axis=-1) + jnp.minimum(jnp.max(q, axis=-1), 0)
    box_closest = jnp.clip(local, -size, size)
    inside = jnp.all(q < 0, axis=-1)
    face = jax.nn.one_hot(jnp.argmax(q, axis=-1), 3)
    box_closest = jnp.where(
        inside[..., None],
        local + face * (jnp.where(local >= 0, 1.0, -1.0) * size - local),
        box_closest,
    )
    radial = jnp.linalg.norm(local[..., :2], axis=-1)
    radius, height = size[..., 0], size[..., 2]
    radial_dir = jnp.where(
        (radial > 1e-8)[..., None],
        local[..., :2] / jnp.maximum(radial[..., None], 1e-8),
        jnp.array([1.0, 0.0]),
    )
    cylq = jnp.stack([radial - radius, jnp.abs(local[..., 2]) - height], -1)
    cylinder_distance = jnp.linalg.norm(jnp.maximum(cylq, 0), axis=-1) + jnp.minimum(
        jnp.max(cylq, axis=-1), 0
    )
    side = jnp.concatenate(
        [
            radial_dir * radius[..., None],
            jnp.clip(local[..., 2:3], -height[..., None], height[..., None]),
        ],
        -1,
    )
    cap = jnp.concatenate(
        [
            radial_dir * jnp.minimum(radial, radius)[..., None],
            jnp.where(local[..., 2:] >= 0, height[..., None], -height[..., None]),
        ],
        -1,
    )
    cylinder_closest = jnp.where(
        (jnp.abs(cylq[..., 0]) < jnp.abs(cylq[..., 1]))[..., None], side, cap
    )
    cylinder_closest = jnp.where(
        jnp.any(cylq > 0, -1)[..., None],
        jnp.concatenate(
            [
                radial_dir * jnp.minimum(radial, radius)[..., None],
                jnp.clip(local[..., 2:3], -height[..., None], height[..., None]),
            ],
            -1,
        ),
        cylinder_closest,
    )
    # Project the radial-height point onto the cone's side segment and base.
    rz = jnp.stack([radial, local[..., 2]], -1)
    apex = jnp.stack([jnp.zeros_like(height), height], -1)
    edge = jnp.stack([radius, -2 * height], -1)
    t = jnp.clip(jnp.sum((rz - apex) * edge, -1) / jnp.sum(edge**2, -1), 0, 1)
    projected_side = apex + t[..., None] * edge
    projected_base = jnp.stack([jnp.minimum(radial, radius), -height], -1)
    near_side = jnp.sum((rz - projected_side) ** 2, -1) < jnp.sum((rz - projected_base) ** 2, -1)
    cone_rz = jnp.where(near_side[..., None], projected_side, projected_base)
    cone_closest = jnp.concatenate([radial_dir * cone_rz[..., :1], cone_rz[..., 1:]], -1)
    cone_inside = (jnp.abs(local[..., 2]) <= height) & (
        radial <= radius * (height - local[..., 2]) / (2 * height)
    )
    cone_distance = jnp.linalg.norm(rz - cone_rz, axis=-1) * jnp.where(cone_inside, -1.0, 1.0)
    distance = jnp.where(
        kind == 0, box_distance, jnp.where(kind == 1, cylinder_distance, cone_distance)
    )
    closest = jnp.where(
        (kind == 0)[..., None],
        box_closest,
        jnp.where((kind == 1)[..., None], cylinder_closest, cone_closest),
    )
    return distance, closest - local


def nearest_obstacle(
    scene: Scene, pos: jax.Array, map_id: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Query exact primitive and boundary distances and nearest surface vectors."""
    center, size = scene.centers[map_id], scene.half_sizes[map_id]
    rotation, kind = scene.rotations[map_id], scene.kinds[map_id]
    local = jnp.einsum("nkji,nkj->nki", rotation, pos[:, None] - center)
    distance, local_vec = primitive_distances(local, size, kind)
    vectors = jnp.einsum("nkij,nkj->nki", rotation, local_vec)
    # Include the floor, ceiling and four walls in occupancy surface queries.
    boundary_d = jnp.stack(
        [
            pos[:, 2] - scene.floor,
            scene.ceiling - pos[:, 2],
            scene.room_half_size - pos[:, 0],
            scene.room_half_size + pos[:, 0],
            scene.room_half_size - pos[:, 1],
            scene.room_half_size + pos[:, 1],
        ],
        -1,
    )
    axes = jnp.array(
        [[0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], pos.dtype
    )
    boundary_vectors = boundary_d[..., None] * axes
    distance = jnp.concatenate([distance, boundary_d], -1)
    vectors = jnp.concatenate([vectors, boundary_vectors], 1)
    index = jnp.argmin(distance, -1)
    return jnp.take_along_axis(distance, index[:, None], 1)[:, 0], jnp.take_along_axis(
        vectors, index[:, None, None], 1
    )[:, 0]


def _ray_primitive(
    origin: jax.Array, direction: jax.Array, size: jax.Array, kind: jax.Array
) -> jax.Array:
    """All quantities broadcast over (world, ray, object); directions need not be unit."""
    safe_d = jnp.where(jnp.abs(direction) > 1e-8, direction, 1e-8)
    a, b = (-size - origin) / safe_d, (size - origin) / safe_d
    enter, leave = jnp.max(jnp.minimum(a, b), -1), jnp.min(jnp.maximum(a, b), -1)
    box = jnp.where((leave >= jnp.maximum(enter, 0)), jnp.maximum(enter, 0), jnp.inf)
    radius, height = size[..., 0], size[..., 2]
    ox, oy, oz = origin[..., 0], origin[..., 1], origin[..., 2]
    dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
    # Cylinder and cone side surfaces are quadratic intersections.
    slope2 = (radius / (2 * height)) ** 2
    cone = kind == 2
    qa = dx**2 + dy**2 - jnp.where(cone, slope2 * dz**2, 0)
    qb = 2 * (ox * dx + oy * dy + jnp.where(cone, slope2 * (height - oz) * dz, 0))
    qc = ox**2 + oy**2 - jnp.where(cone, slope2 * (height - oz) ** 2, radius**2)
    discr = qb**2 - 4 * qa * qc
    quadratic = jnp.abs(qa) > 1e-9
    linear = ~quadratic & (jnp.abs(qb) > 1e-9)
    root = jnp.sqrt(jnp.maximum(discr, 0))
    # The stable quadratic form avoids cancellation of the near intersection
    # for almost parallel rays. A cylinder-axis ray has qa=qb=0 and must not
    # manufacture a side hit at zero; its valid hits are the end caps.
    q = -0.5 * (qb + jnp.where(qb >= 0, root, -root))
    qa_safe = jnp.where(quadratic, qa, 1.0)
    q_safe = jnp.where(q != 0, q, 1.0)
    t1, t2 = q / qa_safe, qc / q_safe

    def valid(t: jax.Array) -> jax.Array:
        return (t >= 0) & (jnp.abs(oz + t * dz) <= height)

    side = jnp.minimum(
        jnp.where(quadratic & (discr >= 0) & valid(t1), t1, jnp.inf),
        jnp.where(quadratic & (discr >= 0) & valid(t2), t2, jnp.inf),
    )
    # Cone generatrix directions make the quadratic term vanish while leaving
    # a linear side equation. Rays lying on a side enter at its height bounds.
    qb_safe = jnp.where(linear, qb, 1.0)
    t_linear = -qc / qb_safe
    side = jnp.minimum(side, jnp.where(linear & valid(t_linear), t_linear, jnp.inf))
    dzsafe = jnp.where(jnp.abs(dz) > 1e-8, dz, 1e-8)
    coincident = ~quadratic & ~linear & (jnp.abs(qc) <= 1e-9)
    height_entry = jnp.maximum(jnp.minimum((-height - oz) / dzsafe, (height - oz) / dzsafe), 0)
    side = jnp.minimum(side, jnp.where(coincident & valid(height_entry), height_entry, jnp.inf))
    bottom, top = (-height - oz) / dzsafe, (height - oz) / dzsafe
    bottom_valid = (bottom >= 0) & ((ox + bottom * dx) ** 2 + (oy + bottom * dy) ** 2 <= radius**2)
    top_valid = (top >= 0) & ((ox + top * dx) ** 2 + (oy + top * dy) ** 2 <= radius**2) & ~cone
    capped = jnp.minimum(
        side,
        jnp.minimum(jnp.where(bottom_valid, bottom, jnp.inf), jnp.where(top_valid, top, jnp.inf)),
    )
    return jnp.where(kind == 0, box, capped)


def raycast(
    scene: Scene, origins: jax.Array, directions: jax.Array, map_id: jax.Array, maximum: float = 4.0
) -> jax.Array:
    """Return image-plane depth. Chunk rays to bound temporary GPU memory."""
    rotations = scene.rotations[map_id]
    centers, sizes, kinds = scene.centers[map_id], scene.half_sizes[map_id], scene.kinds[map_id]

    def render(o: jax.Array, d: jax.Array) -> jax.Array:
        local_o = jnp.einsum("nkji,nrkj->nrki", rotations, o[:, :, None] - centers[:, None])
        local_d = jnp.einsum("nkji,nrj->nrki", rotations, d)
        hit = _ray_primitive(local_o, local_d, sizes[:, None], kinds[:, None])
        closest = jnp.min(hit, -1)
        bounds_low = jnp.array([-scene.room_half_size, -scene.room_half_size, scene.floor])
        bounds_high = jnp.array([scene.room_half_size, scene.room_half_size, scene.ceiling])
        safe_d = jnp.where(jnp.abs(d) > 1e-8, d, 1e-8)
        bounds = jnp.where(d > 0, (bounds_high - o) / safe_d, (bounds_low - o) / safe_d)
        boundary = jnp.min(jnp.where(bounds >= 0, bounds, jnp.inf), -1)
        # Isaac ray_caster_camera uses max_dist=1e6, converts intersections
        # to image-plane distance, then clips at camera_max_distance.
        return jnp.clip(jnp.minimum(closest, boundary), 0, maximum)

    # 4 rays per scan limits the N*ray*object intermediates at large n_worlds.
    n, rays, _ = origins.shape
    pad = (-rays) % 4
    o = jnp.pad(origins, ((0, 0), (0, pad), (0, 0))).reshape(n, -1, 4, 3).swapaxes(0, 1)
    d = (
        jnp.pad(directions, ((0, 0), (0, pad), (0, 0)), constant_values=1)
        .reshape(n, -1, 4, 3)
        .swapaxes(0, 1)
    )
    _, depths = jax.lax.scan(lambda carry, values: (carry, render(*values)), None, (o, d))
    return depths.swapaxes(0, 1).reshape(n, -1)[:, :rays]
