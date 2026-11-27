import math
import omni
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg
import isaacsim.core.utils.prims as prims_utils
from scipy.spatial import KDTree
import random
from pxr import UsdGeom, Sdf, Gf, Vt, Usd, UsdShade, UsdPhysics
import time
import trimesh

def flatten_instancer_and_mesh(
    stage: Usd.Stage,
    source_root_path: str,
    merged_mesh_path: str,
    min_bound: tuple = None,
    max_bound: tuple = None,
    env_space: float = 10.0
) -> str:
    """
    Traverse source_root_path on the given USD Stage, instantiate and merge
    all PointInstancers and Mesh prims into one UsdGeom.Mesh at merged_mesh_path.
    Return the prim path of the new merged Mesh.

    Args:
        stage: USD Stage to operate on
        source_root_path: Path to the root prim containing geometry to flatten
        merged_mesh_path: Path where the merged mesh will be created
        min_bound: Minimum bounds for mesh filtering (default: (-env_space*2, -env_space*2, 0))
        max_bound: Maximum bounds for mesh filtering (default: (env_space*2, env_space*2, 5.0))
        env_space: Environment space parameter for default bounds calculation (default: 10.0)

    Performance-optimized implementation using vectorized operations and bulk USD API calls.
    """

    if min_bound is None:
        min_bound = (-env_space * 2, -env_space * 2, 0.0)
    if max_bound is None:
        max_bound = (env_space * 2, env_space * 2, 5.0)

    print(f"Using mesh bounds: min={min_bound}, max={max_bound}")

    source_prim = stage.GetPrimAtPath(source_root_path)
    if not source_prim.IsValid():
        raise ValueError(f"Source root path '{source_root_path}' does not exist on stage")

    if not merged_mesh_path.startswith('/'):
        raise ValueError(f"Merged mesh path '{merged_mesh_path}' must be absolute")

    print(f"Flattening geometry from '{source_root_path}' into '{merged_mesh_path}'...")
    start_time = time.time()

    instancers = []
    meshes = []

    for prim in Usd.PrimRange(source_prim):
        if prim.IsA(UsdGeom.PointInstancer):
            instancers.append(UsdGeom.PointInstancer(prim))
        elif prim.IsA(UsdGeom.Mesh):
            meshes.append(UsdGeom.Mesh(prim))

    print(f"Found {len(instancers)} PointInstancers and {len(meshes)} standalone Meshes")

    all_vertices = []
    all_face_counts = []
    all_face_indices = []
    vertex_offset = 0

    for instancer in instancers:
        instancer_prim = instancer.GetPrim()
        instancer_world_transform = UsdGeom.Xformable(instancer_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        positions = instancer.GetPositionsAttr().Get()

        if not positions:
            print(f"Info: No positions found in {instancer.GetPath()}") # sometimes only one type of obstacle is generated
            continue

        orientations = instancer.GetOrientationsAttr().Get()
        scales = instancer.GetScalesAttr().Get()
        proto_indices = instancer.GetProtoIndicesAttr().Get()
        prototypes_rel = instancer.GetPrototypesRel()
        prototype_paths = prototypes_rel.GetTargets()
        prototype_meshes = {}
        for i, proto_path in enumerate(prototype_paths):
            proto_prim = stage.GetPrimAtPath(proto_path)
            geometric_data_list = collect_prototype_meshes(proto_prim)
            combined_points = []
            combined_face_counts = []
            combined_face_indices = []
            proto_vertex_offset = 0

            for geom_data in geometric_data_list:
                points = geom_data['points']
                face_counts = geom_data['face_counts']
                face_indices = geom_data['face_indices']

                combined_points.extend(points)
                combined_face_counts.extend(face_counts)

                for idx in face_indices:
                    combined_face_indices.append(idx + proto_vertex_offset)

                proto_vertex_offset += len(points)

            if combined_points:
                prototype_meshes[proto_path] = {
                    'points': Vt.Vec3fArray(combined_points),
                    'face_counts': Vt.IntArray(combined_face_counts),
                    'face_indices': Vt.IntArray(combined_face_indices)
                }

        if not prototype_meshes:
            print(f"  Error: No prototype meshes found for {instancer.GetPath()}")
            continue

        num_instances = len(positions)

        if not orientations or len(orientations) != num_instances:
            orientations = [Gf.Quath(1, 0, 0, 0)] * num_instances
        if not scales or len(scales) != num_instances:
            scales = [Gf.Vec3f(1, 1, 1)] * num_instances

        if not proto_indices:
            proto_indices = [0] * num_instances
        elif len(proto_indices) != num_instances:
            if len(proto_indices) > num_instances:
                proto_indices = proto_indices[:num_instances]
            else:
                proto_indices = list(proto_indices) + [0] * (num_instances - len(proto_indices))

        max_proto_index = len(prototype_paths) - 1

        transform_matrices = []
        for i in range(num_instances):
            pos = positions[i]
            orient = orientations[i]
            scale = scales[i]

            scale_matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(scale[0], scale[1], scale[2]))
            rotate_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(orient)))
            local_transform = rotate_matrix * scale_matrix
            local_transform.SetRow(3, Gf.Vec4d(pos[0], pos[1], pos[2], 1.0))
            final_transform = instancer_world_transform * local_transform
            transform_matrices.append(final_transform)

        instances_processed = 0

        for i in range(num_instances):
            proto_idx = max(0, min(proto_indices[i], len(prototype_paths) - 1))
            proto_path = prototype_paths[proto_idx]

            if proto_path not in prototype_meshes:
                continue  # Skip silently to avoid log spam

            proto_data = prototype_meshes[proto_path]
            proto_points = proto_data['points']
            proto_face_counts = proto_data['face_counts']
            proto_face_indices = proto_data['face_indices']

            transform_matrix = transform_matrices[i]
            transformed_points = Vt.Vec3fArray(len(proto_points))

            for j, point in enumerate(proto_points):
                transformed_point = transform_matrix.Transform(Gf.Vec3d(point[0], point[1], point[2]))
                transformed_points[j] = Gf.Vec3f(transformed_point[0], transformed_point[1], transformed_point[2])

            all_vertices.append(transformed_points)
            all_face_counts.extend(proto_face_counts)

            for idx in proto_face_indices:
                all_face_indices.append(idx + vertex_offset)

            vertex_offset += len(proto_points)
            instances_processed += 1

    for mesh in meshes:
        mesh_path = mesh.GetPath()

        if "/prototypes/" in str(mesh_path):
            continue

        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if not points or not face_counts or not face_indices:
            continue

        mesh_prim = mesh.GetPrim()
        world_transform = UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        transformed_points = Vt.Vec3fArray(len(points))
        for i, point in enumerate(points):
            transformed_point = world_transform.Transform(Gf.Vec3d(point[0], point[1], point[2]))
            transformed_points[i] = Gf.Vec3f(transformed_point[0], transformed_point[1], transformed_point[2])

        all_vertices.append(transformed_points)
        all_face_counts.extend(face_counts)

        adjusted_indices = []
        for idx in face_indices:
            adjusted_indices.append(idx + vertex_offset)
        all_face_indices.extend(adjusted_indices)

        vertex_offset += len(points)

    if not all_vertices:
        print("No geometry found to flatten")
        return ""

    if prims_utils.is_prim_path_valid(merged_mesh_path):
        prims_utils.delete_prim(merged_mesh_path)

    total_vertices = sum(len(verts) for verts in all_vertices)
    merged_vertices = Vt.Vec3fArray(total_vertices)

    vertex_idx = 0
    for vertex_array in all_vertices:
        for vertex in vertex_array:
            merged_vertices[vertex_idx] = vertex
            vertex_idx += 1

    merged_face_counts = Vt.IntArray(all_face_counts)
    merged_face_indices = Vt.IntArray(all_face_indices)

    all_vertices.clear()
    all_face_counts.clear()
    all_face_indices.clear()

    merged_mesh = UsdGeom.Mesh.Define(stage, merged_mesh_path)

    merged_mesh.GetPointsAttr().Set(merged_vertices)
    merged_mesh.GetFaceVertexCountsAttr().Set(merged_face_counts)
    merged_mesh.GetFaceVertexIndicesAttr().Set(merged_face_indices)

    merged_mesh.GetSubdivisionSchemeAttr().Set("none")  # Disable subdivision for performance

    # Add collision properties to the unified mesh
    stage = omni.usd.get_context().get_stage()
    mesh_prim = merged_mesh.GetPrim()

    # Add physics collision API
    collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)

    # Set collision properties
    collision_api.CreateCollisionEnabledAttr().Set(True)

    # Add mesh collision API for trimesh collision
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.CreateApproximationAttr().Set("sdf")

    # Add rigid body properties
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(mesh_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr().Set(False)
    rigid_body_api.CreateKinematicEnabledAttr().Set(True)

    try:
        imageable = UsdGeom.Imageable(merged_mesh.GetPrim())
        imageable.CreateVisibilityAttr().Set("invisible")
    except Exception as e:
        print(f"Warning: Failed to set merged mesh invisible: {e}")

    # Convert to numpy array for vectorized counting (much faster for large arrays)
    face_counts_np = np.array(merged_face_counts, dtype=np.int32)
    total_triangles = np.count_nonzero(face_counts_np == 3)
    total_quads = np.count_nonzero(face_counts_np == 4)

    total_time = time.time() - start_time

    print(f"Mesh flattening complete:")
    print(f"  - Vertices: {total_vertices:,}, Faces: {len(merged_face_counts):,} ({total_triangles:,} triangles, {total_quads:,} quads)")
    print(f"  - Time: {total_time:.3f}s, Rate: {total_vertices/total_time:,.0f} vertices/sec")

    return merged_mesh_path

def collect_prototype_meshes(proto_prim):
    """Collect all valid Mesh prims under a prototype and return their point and face data."""
    mesh_data_list = []
    for prim in Usd.PrimRange(proto_prim):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            fcs = mesh.GetFaceVertexCountsAttr().Get()
            fvs = mesh.GetFaceVertexIndicesAttr().Get()
            if pts and fcs and fvs:
                mesh_data_list.append({'points': pts, 'face_counts': fcs, 'face_indices': fvs})
    return mesh_data_list

class MapGenerator:
    """
    Class to generate maps with walls and obstacles for drone environments.
    """
    def __init__(self, sim, device="cuda:0", map_origin=(0.0, 0.0, 0.0), base_prim="/World/ground"):
        self.device = device
        self.sim = sim
        self.map_origin = map_origin
        self.base_prim = base_prim
        self._generation_in_progress = False

        # Enable viewport grid by default
        self._enable_viewport_grid()

    def _get_prim_path(self, relative_path):
        """
        Construct a primitive path relative to the base primitive.

        Args:
            relative_path: Relative path from base primitive

        Returns:
            Full primitive path
        """
        return f"{self.base_prim}/{relative_path}" if relative_path else self.base_prim

    def _translate_to_origin(self, position):
        """
        Translate a position relative to the map origin.

        Args:
            position: Tuple of (x, y, z) coordinates

        Returns:
            Tuple of translated coordinates
        """
        return (
            position[0] + self.map_origin[0],
            position[1] + self.map_origin[1],
            position[2] + self.map_origin[2]
        )

    def generate_walls(self, scene):
        """
        Generate walls and ground around the environment relative to map origin.

        Args:
            scene: Scene configuration with env_spacing parameter

        Returns:
            List of wall and ground primitive paths
        """
        wall_prims = []

        # Walls - optimized to avoid oblong shape warnings by using reasonable aspect ratios
        # PhysX works best when no dimension is more than 40x larger than the smallest dimension
        wall_height = 3.0
        wall_length = scene.env_spacing + 6.0
        wall_thickness = max(0.2, max(wall_height, wall_length) / 40.0)

        wall_cfg = sim_utils.MeshCuboidCfg(
            size=(wall_length, wall_thickness, wall_height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 0.8)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True,
        )

        # Ground plane configuration using MeshCuboid with better aspect ratio
        # Ensure no dimension exceeds 40x the thickness to avoid oblong warnings
        ground_length = scene.env_spacing + 6.0
        ground_width = scene.env_spacing + 6.0
        ground_thickness = max(0.2, max(ground_length, ground_width) / 40.0)

        ground_cfg = sim_utils.MeshCuboidCfg(
            size=(ground_length, ground_width, ground_thickness),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True,
        )

        # Left wall
        left_wall_path = self._get_prim_path("left_wall")
        wall_cfg.func(left_wall_path, wall_cfg, translation=self._translate_to_origin((0.0, scene.env_spacing / 2.0 + wall_thickness / 2.0 + 3.0, wall_height / 2.0)))
        wall_prims.append(left_wall_path)

        # Right wall
        right_wall_path = self._get_prim_path("right_wall")
        wall_cfg.func(right_wall_path, wall_cfg, translation=self._translate_to_origin((0.0, -scene.env_spacing / 2.0 - wall_thickness / 2.0 - 3.0, wall_height / 2.0)))
        wall_prims.append(right_wall_path)

        # Forward wall
        forward_wall_path = self._get_prim_path("forward_wall")
        wall_cfg.func(forward_wall_path, wall_cfg, translation=self._translate_to_origin((scene.env_spacing / 2.0 + wall_thickness / 2.0 + 3.0, 0.0, wall_height / 2.0)), orientation=(0.707, 0.0, 0.0, 0.707))
        wall_prims.append(forward_wall_path)

        # Backward wall
        backward_wall_path = self._get_prim_path("backward_wall")
        wall_cfg.func(backward_wall_path, wall_cfg, translation=self._translate_to_origin((-scene.env_spacing / 2.0 - wall_thickness / 2.0 - 3.0, 0.0, wall_height / 2.0)), orientation=(0.707, 0.0, 0.0, 0.707))
        wall_prims.append(backward_wall_path)

        # Ceiling - use the same thickness as ground for consistency
        ceiling_path = self._get_prim_path("ceiling")
        ceiling_cfg = ground_cfg.replace(size=(ground_length, ground_width, ground_thickness))
        ceiling_cfg.func(ceiling_path, ceiling_cfg, translation=self._translate_to_origin((0.0, 0.0, 2.0 + ground_thickness / 2.0)))

        # Make the ceiling invisible by default
        stage = omni.usd.get_context().get_stage()
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(ceiling_path))
        imageable.MakeInvisible()

        wall_prims.append(ceiling_path)

        # Ground plane - using MeshCuboid instead of GroundPlaneCfg
        ground_path = self._get_prim_path("ground_plane")
        ground_cfg.func(ground_path, ground_cfg, translation=self._translate_to_origin((0.0, 0.0, 0.01 - ground_thickness / 2.0)))
        wall_prims.append(ground_path)

        return wall_prims

    def generate_obstacles(self, scene, num_obstacles=100, min_distance=1.0, obstacle_size_range=(0.3, 0.8), height_range=(0.5, 3.0)):
        """
        Generate random obstacles using PointInstancer for highly efficient geometry creation.

        Args:
            scene: Scene configuration with env_spacing parameter
            num_obstacles: Number of obstacles to generate
            min_distance: Minimum distance between obstacle surfaces
            obstacle_size_range: Range of obstacle sizes (min, max)
            height_range: Range of obstacle heights (min, max)

        Returns:
            List of obstacle primitive paths and their positions
        """
        # Environment bounds relative to origin
        x_min = -scene.env_spacing / 2.0 + self.map_origin[0]
        x_max = scene.env_spacing / 2.0 + self.map_origin[0]
        y_min = -scene.env_spacing / 2.0 + self.map_origin[1]
        y_max = scene.env_spacing / 2.0 + self.map_origin[1]
        
        # Minimum distance between points (centers)
        min_center_distance = obstacle_size_range[1] + min_distance

        # Generate points using Poisson disk sampling
        points = self.poisson_disk_sampling(
            width=x_max - x_min,
            height=y_max - y_min,
            min_distance=min_center_distance,
            max_points=num_obstacles
        )

        print(f"Generated {len(points)} points for obstacles")

        cuboid_prototype_path = self._get_prim_path("prototypes/cuboid_prototype")
        cylinder_prototype_path = self._get_prim_path("prototypes/cylinder_prototype")

        # Create prototype directories if needed
        prototype_dir = self._get_prim_path("prototypes")
        if not prims_utils.is_prim_path_valid(prototype_dir):
            prims_utils.create_prim(prototype_dir, "Xform")

        cuboid_cfg = sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, 1.0),  # Reference unit size
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True,
        )

        cylinder_cfg = sim_utils.MeshCylinderCfg(
            radius=0.5,  # Reference unit radius
            height=1.0,  # Reference unit height
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
            activate_contact_sensors=True,
        )

        # Create prototypes and optimize collision approximations
        if not prims_utils.is_prim_path_valid(cuboid_prototype_path):
            cuboid_cfg.func(cuboid_prototype_path, cuboid_cfg)
        if not prims_utils.is_prim_path_valid(cylinder_prototype_path):
            cylinder_cfg.func(cylinder_prototype_path, cylinder_cfg)

        # Create PointInstancers
        cuboid_instancer_path = self._get_prim_path("cuboid_instancer")
        cylinder_instancer_path = self._get_prim_path("cylinder_instancer")

        # Get the stage
        stage = omni.usd.get_context().get_stage()

        # Create or get instancers
        if not prims_utils.is_prim_path_valid(cuboid_instancer_path):
            # Create the PointInstancer primitive
            cuboid_instancer = UsdGeom.PointInstancer.Define(stage, cuboid_instancer_path)
            # Add prototype references
            cuboid_instancer.GetPrototypesRel().AddTarget(cuboid_prototype_path)

        if not prims_utils.is_prim_path_valid(cylinder_instancer_path):
            # Create the PointInstancer primitive
            cylinder_instancer = UsdGeom.PointInstancer.Define(stage, cylinder_instancer_path)
            # Add prototype references
            cylinder_instancer.GetPrototypesRel().AddTarget(cylinder_prototype_path)

        # Prepare transform data for instances
        cuboid_positions = []
        cuboid_scales = []
        cuboid_orientations = []
        cylinder_positions = []
        cylinder_scales = []
        cylinder_orientations = []

        # Prepare obstacle info and materials
        obstacle_positions = []
        obstacle_types = []
        cuboid_colors = []
        cylinder_colors = []

        # Process all points and prepare instance data
        for i, point in enumerate(points):
            # Translate point to world coordinates
            x = point[0] + (x_min)
            y = point[1] + (y_min)

            # Generate obstacle dimensions with reasonable aspect ratios to avoid oblong shape warnings
            width = random.uniform(*obstacle_size_range)
            length = random.uniform(*obstacle_size_range)
            height = random.uniform(*height_range)

            # Ensure no dimension is more than 10x larger than the smallest to avoid oblong warnings
            min_dim = min(width, length, height)
            max_allowed = min_dim * 10.0
            width = min(width, max_allowed)
            length = min(length, max_allowed)
            height = min(height, max_allowed)

            # Randomly choose obstacle type
            obstacle_type = random.choice(["cuboid", "cylinder"])

            # Calculate color based on height (gradient from red to blue)
            height_ratio = (height - height_range[0]) / (height_range[1] - height_range[0] + 1e-5)
            color = (
                0.9 * (1.0 - height_ratio),  # Red decreases with height
                0.2,                         # Green constant
                0.9 * height_ratio,          # Blue increases with height
            )

            # Record obstacle information
            if obstacle_type == "cuboid":
                obstacle_info = (x, y, width, length)

                # Add to instancer data with origin offset
                cuboid_positions.append((x, y, height/2.0 + 0.05 + self.map_origin[2]))
                cuboid_scales.append((width, length, height))
                cuboid_orientations.append((1.0, 0.0, 0.0, 0.0))
                cuboid_colors.append(color)
            else:  # cylinder
                radius = (width + length) / 4.0
                obstacle_info = (x, y, radius)

                # Add to instancer data with origin offset
                cylinder_positions.append((x, y, height/2.0 + 0.05 + self.map_origin[2]))
                cylinder_scales.append((radius*2, radius*2, height))  # Scale x2 for diameter
                cylinder_orientations.append((1.0, 0.0, 0.0, 0.0))
                cylinder_colors.append(color)

            obstacle_positions.append(obstacle_info)
            obstacle_types.append(obstacle_type)

        # Apply all instance transformations at once
        obstacle_prims = []

        # Update cuboid instances
        if cuboid_positions:
            # Get the instancer
            cuboid_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(cuboid_instancer_path))

            # Create proper VtArray for positions, scales and orientations
            # Positions - VtVec3fArray
            pos_vtarray = Vt.Vec3fArray(len(cuboid_positions))
            for i, pos in enumerate(cuboid_positions):
                pos_vtarray[i] = Gf.Vec3f(pos[0], pos[1], pos[2])

            # Scales - VtVec3fArray
            scale_vtarray = Vt.Vec3fArray(len(cuboid_scales))
            for i, scale in enumerate(cuboid_scales):
                scale_vtarray[i] = Gf.Vec3f(scale[0], scale[1], scale[2])

            # Orientations - VtQuathArray (half-float quaternion)
            orient_vtarray = Vt.QuathArray(len(cuboid_orientations))
            for i, orient in enumerate(cuboid_orientations):
                # Convert to Quaternion with half-precision (Quath)
                orient_vtarray[i] = Gf.Quath(orient[0], orient[1], orient[2], orient[3])

            # Set instance indices (all using prototype 0)
            proto_indices = Vt.IntArray(len(cuboid_positions), 0)  # Initialize with zeros

            # Set all attributes
            cuboid_instancer.CreateProtoIndicesAttr().Set(proto_indices)
            cuboid_instancer.CreatePositionsAttr().Set(pos_vtarray)
            cuboid_instancer.CreateScalesAttr().Set(scale_vtarray)
            cuboid_instancer.CreateOrientationsAttr().Set(orient_vtarray)

            # Add instance paths to output list
            cuboid_count = len(cuboid_positions)
            for i in range(cuboid_count):
                obstacle_prims.append(f"{cuboid_instancer_path}/instance_{i}")

        # Update cylinder instances
        if cylinder_positions:
            # Get the instancer
            cylinder_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(cylinder_instancer_path))

            # Create proper VtArray for positions, scales and orientations
            # Positions - VtVec3fArray
            pos_vtarray = Vt.Vec3fArray(len(cylinder_positions))
            for i, pos in enumerate(cylinder_positions):
                pos_vtarray[i] = Gf.Vec3f(pos[0], pos[1], pos[2])

            # Scales - VtVec3fArray
            scale_vtarray = Vt.Vec3fArray(len(cylinder_scales))
            for i, scale in enumerate(cylinder_scales):
                scale_vtarray[i] = Gf.Vec3f(scale[0], scale[1], scale[2])

            # Orientations - VtQuathArray (half-float quaternion)
            orient_vtarray = Vt.QuathArray(len(cylinder_orientations))
            for i, orient in enumerate(cylinder_orientations):
                # Convert to Quaternion with half-precision (Quath)
                orient_vtarray[i] = Gf.Quath(orient[0], orient[1], orient[2], orient[3])

            # Set instance indices (all using prototype 0)
            proto_indices = Vt.IntArray(len(cylinder_positions), 0)  # Initialize with zeros

            # Set all attributes
            cylinder_instancer.CreateProtoIndicesAttr().Set(proto_indices)
            cylinder_instancer.CreatePositionsAttr().Set(pos_vtarray)
            cylinder_instancer.CreateScalesAttr().Set(scale_vtarray)
            cylinder_instancer.CreateOrientationsAttr().Set(orient_vtarray)

            # Add instance paths to output list
            cylinder_count = len(cylinder_positions)
            for i in range(cylinder_count):
                obstacle_prims.append(f"{cylinder_instancer_path}/instance_{i}")

        return obstacle_prims, obstacle_positions

    def poisson_disk_sampling(self, width, height, min_distance, max_points=None, k=30):
        """
        Generate Poisson disk sampling points in a rectangle.

        Args:
            width, height: Dimensions of the rectangle
            min_distance: Minimum distance between points
            max_points: Maximum number of points to generate (optional)
            k: Number of attempts to place a new point near an existing one

        Returns:
            List of points [(x, y)]
        """
        # Cell size for the grid
        cell_size = min_distance / math.sqrt(2)

        # Grid dimensions
        grid_width = int(math.ceil(width / cell_size))
        grid_height = int(math.ceil(height / cell_size))

        # Grid to store sample indices - using sparse dictionary for efficiency
        grid = {}

        # List of samples
        samples = []

        # List of active samples (indices)
        active = []

        # Initial sample
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        initial_sample = (x, y)

        # Get the grid coordinates for a point
        def get_cell(point):
            return (int(point[0] / cell_size), int(point[1] / cell_size))

        # Add first sample
        samples.append(initial_sample)
        active.append(0)

        cell_coords = get_cell(initial_sample)
        grid[cell_coords] = 0

        # While there are active samples and we haven't reached max_points
        while active and (max_points is None or len(samples) < max_points):
            # Choose a random active sample
            active_index = random.randrange(len(active))
            sample_index = active[active_index]
            sample = samples[sample_index]

            # Try to generate a new sample near the active one
            found = False
            for _ in range(k):
                # Generate a random point between min_distance and 2*min_distance from the sample
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(min_distance, 2 * min_distance)
                new_x = sample[0] + math.cos(angle) * distance
                new_y = sample[1] + math.sin(angle) * distance

                # Check if the point is in bounds
                if not (0 <= new_x < width and 0 <= new_y < height):
                    continue

                # Check if the point is far enough from existing samples
                cell_x, cell_y = get_cell((new_x, new_y))

                # Check nearby cells
                valid = True
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        neighbor_cell = (cell_x + dx, cell_y + dy)

                        if neighbor_cell in grid:
                            neighbor_sample = samples[grid[neighbor_cell]]
                            distance = math.sqrt((new_x - neighbor_sample[0])**2 + (new_y - neighbor_sample[1])**2)
                            if distance < min_distance:
                                valid = False
                                break

                if not valid:
                    continue

                # Add the new sample
                new_sample = (new_x, new_y)
                new_sample_index = len(samples)
                samples.append(new_sample)
                active.append(new_sample_index)

                # Add to grid
                grid[get_cell(new_sample)] = new_sample_index

                found = True
                break

            if not found:
                # Remove the active sample we just processed
                active.pop(active_index)

        # If we need to limit the number of points
        if max_points is not None and len(samples) > max_points:
            samples = samples[:max_points]

        return samples

    def generate_floaters(self, scene, num_floaters=100, min_distance=1.0, floater_size_range=(0.3, 0.8), height_range=(0.5, 3.0)):
        """
        Generate random floating objects using PointInstancer for efficiency.
        Uses stratified sampling to ensure uniform distribution across the entire environment.

        Args:
            scene: Scene configuration with env_spacing parameter
            num_floaters: Number of floating objects to generate
            min_distance: Minimum distance between floating objects
            floater_size_range: Range of floating object sizes (min, max)
            height_range: Range of floating object heights above ground (min, max)

        Returns:
            List of floater primitive paths and their positions
        """
        # Environment bounds relative to origin
        x_min = -scene.env_spacing / 2.0 + self.map_origin[0]
        x_max = scene.env_spacing / 2.0 + self.map_origin[0]
        y_min = -scene.env_spacing / 2.0 + self.map_origin[1]
        y_max = scene.env_spacing / 2.0 + self.map_origin[1]
        z_min = height_range[0] + self.map_origin[2]
        z_max = height_range[1] + self.map_origin[2]

        # Calculate volume dimensions
        volume_width = x_max - x_min
        volume_height = y_max - y_min
        volume_depth = z_max - z_min

        # List to store generated points
        points_3d = []

        # Create spatial index for distance checks
        spatial_grid = {}
        spatial_cell_size = min_distance

        # Helper function to get spatial grid cell for a point
        def get_spatial_cell(pos):
            return (int(pos[0] / spatial_cell_size),
                    int(pos[1] / spatial_cell_size),
                    int(pos[2] / spatial_cell_size))

        # Helper function to check if a point is valid (far enough from existing points)
        def is_valid_point(pos):
            # Check if point is within bounds
            if not (x_min <= pos[0] < x_max and y_min <= pos[1] < y_max and z_min <= pos[2] < z_max):
                return False

            # Get the spatial cell for the point
            cell = get_spatial_cell(pos)

            # Check surrounding cells
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                        if neighbor_cell in spatial_grid:
                            for other_pos in spatial_grid[neighbor_cell]:
                                dist = math.sqrt(sum((a-b)**2 for a, b in zip(pos, other_pos)))
                                if dist < min_distance * 2: # Times 2 to maintain a reasonable density
                                    return False
            return True

        # Add a valid point to our collection
        def add_point(pos):
            points_3d.append(pos)
            cell = get_spatial_cell(pos)
            if cell not in spatial_grid:
                spatial_grid[cell] = []
            spatial_grid[cell].append(pos)

        # Calculate grid dimensions for sampling
        cells_per_dim = max(3, int(math.ceil(num_floaters ** (1/3) * 1.5)))
        total_cells = cells_per_dim ** 3

        print(f"Using {cells_per_dim}³ grid for stratified sampling ({total_cells} cells)")

        # Cell sizes
        cell_width = volume_width / cells_per_dim
        cell_height = volume_height / cells_per_dim
        cell_depth = volume_depth / cells_per_dim

        # Create and shuffle cell indices for randomized processing
        cell_indices = [(i, j, k)
                        for i in range(cells_per_dim)
                        for j in range(cells_per_dim)
                        for k in range(cells_per_dim)]
        random.shuffle(cell_indices)

        # Fill cells with stratified sampling
        jitter_factor = 0.8  # Controls jitter amount (0.8 = 80% of cell size)
        cells_to_fill = min(total_cells, num_floaters)
        filled_cells = 0

        for i, j, k in cell_indices:
            if filled_cells >= cells_to_fill:
                break

            # Calculate cell center
            cell_center = (
                x_min + (i + 0.5) * cell_width,
                y_min + (j + 0.5) * cell_height,
                z_min + (k + 0.5) * cell_depth
            )

            # Apply random jitter within the cell
            jitter = (
                random.uniform(-jitter_factor, jitter_factor) * cell_width * 0.5,
                random.uniform(-jitter_factor, jitter_factor) * cell_height * 0.5,
                random.uniform(-jitter_factor, jitter_factor) * cell_depth * 0.5
            )

            # Create point with jitter
            pos = (cell_center[0] + jitter[0],
                   cell_center[1] + jitter[1],
                   cell_center[2] + jitter[2])

            # Add if valid
            if is_valid_point(pos):
                add_point(pos)
                filled_cells += 1

        print(f"Stratified sampling: placed {filled_cells}/{cells_to_fill} points")

        # Fill any remaining points with random sampling if needed
        if filled_cells < num_floaters:
            remaining = num_floaters - filled_cells
            max_attempts = remaining * 5
            attempts = 0
            added = 0

            while added < remaining and attempts < max_attempts:
                pos = (random.uniform(x_min, x_max),
                       random.uniform(y_min, y_max),
                       random.uniform(z_min, z_max))

                if is_valid_point(pos):
                    add_point(pos)
                    added += 1

                attempts += 1

            print(f"Random sampling: added {added}/{remaining} points")

        print(f"Generated {len(points_3d)} floater points")

        # Create prototypes - place them outside the environment boundary
        prototype_paths = {
            "cuboid_floater": self._get_prim_path("prototypes/cuboid_floater_prototype"),
            "cylinder_floater": self._get_prim_path("prototypes/cylinder_floater_prototype"),
            "cone_floater": self._get_prim_path("prototypes/cone_floater_prototype")
        }

        instancer_paths = {
            "cuboid_floater": self._get_prim_path("cuboid_floater_instancer"),
            "cylinder_floater": self._get_prim_path("cylinder_floater_instancer"),
            "cone_floater": self._get_prim_path("cone_floater_instancer")
        }

        # Create prototype directory if needed
        prototype_dir = self._get_prim_path("prototypes")
        if not prims_utils.is_prim_path_valid(prototype_dir):
            prims_utils.create_prim(prototype_dir, "Xform")

        prototype_configs = {
            # Use cuboid as simple 3D floating obstacle - only 12 triangular faces (6 quads)
            "cuboid_floater": sim_utils.MeshCuboidCfg(
                size=(1.0, 0.5, 0.5),  # cuboid shape - minimal mesh complexity
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
                activate_contact_sensors=True,
            ),
            # Use cylinder for rounded floating obstacles
            "cylinder_floater": sim_utils.MeshCylinderCfg(
                radius=0.5,  # Reference unit radius
                height=1.0,  # Reference unit height
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
                activate_contact_sensors=True,
            ),
            # Keep cone for pointed floating obstacles
            "cone_floater": sim_utils.MeshConeCfg(
                radius=0.5,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=False, kinematic_enabled=True),
                activate_contact_sensors=True,
            )
        }

        # Create prototypes if they don't exist - at offset location
        for shape_type, path in prototype_paths.items():
            if not prims_utils.is_prim_path_valid(path):
                prototype_configs[shape_type].func(path, prototype_configs[shape_type])

        # Get the stage
        stage = omni.usd.get_context().get_stage()

        # Create or get instancers
        for shape_type, path in instancer_paths.items():
            if not prims_utils.is_prim_path_valid(path):
                instancer = UsdGeom.PointInstancer.Define(stage, path)
                instancer.GetPrototypesRel().AddTarget(prototype_paths[shape_type])

        # Prepare transform data by shape type
        transform_data = {
            "cuboid_floater": {"positions": [], "scales": [], "orientations": []},
            "cylinder_floater": {"positions": [], "scales": [], "orientations": []},
            "cone_floater": {"positions": [], "scales": [], "orientations": []}
        }

        floater_positions = []

        # Process points and prepare instance data
        for pos in points_3d:
            # Random size and shape type
            size = random.uniform(*floater_size_range)
            shape_type = random.choice(["cuboid_floater", "cylinder_floater", "cone_floater"])

            # Random orientation using improved method
            u1, u2, u3 = random.random(), random.random(), random.random()
            qw = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
            qx = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
            qy = math.sqrt(u1) * math.sin(2 * math.pi * u3)
            qz = math.sqrt(u1) * math.cos(2 * math.pi * u3)
            orientation = (qw, qx, qy, qz)

            # Add to appropriate instancer data
            transform_data[shape_type]["positions"].append(pos)

            if shape_type == "cuboid_floater":
                transform_data[shape_type]["scales"].append((size, size, size))
            elif shape_type == "cylinder_floater":
                transform_data[shape_type]["scales"].append((size, size, size))  # radius scaled uniformly
            else:  # cone_floater
                transform_data[shape_type]["scales"].append((size, size, size))

            transform_data[shape_type]["orientations"].append(orientation)
            floater_positions.append(pos)

        # Apply all instance transformations
        floater_prims = []

        for shape_type, data in transform_data.items():
            if not data["positions"]:
                continue

            instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_paths[shape_type]))

            # Create VtArrays for the transform data
            pos_array = Vt.Vec3fArray(len(data["positions"]))
            scale_array = Vt.Vec3fArray(len(data["scales"]))
            orient_array = Vt.QuathArray(len(data["orientations"]))

            # Fill the arrays
            for i, (pos, scale, orient) in enumerate(zip(
                data["positions"], data["scales"], data["orientations"])):
                pos_array[i] = Gf.Vec3f(*pos)
                scale_array[i] = Gf.Vec3f(*scale)
                orient_array[i] = Gf.Quath(*orient)

            # Set instance indices (all using prototype 0)
            proto_indices = Vt.IntArray(len(data["positions"]), 0)

            # Set all attributes
            instancer.CreateProtoIndicesAttr().Set(proto_indices)
            instancer.CreatePositionsAttr().Set(pos_array)
            instancer.CreateScalesAttr().Set(scale_array)
            instancer.CreateOrientationsAttr().Set(orient_array)

            # Add instance paths to output list
            for i in range(len(data["positions"])):
                floater_prims.append(f"{instancer_paths[shape_type]}/instance_{i}")

        return floater_prims, floater_positions

    def points_filter(self, min_bounds, max_bounds, points):
        """
        Filter points based on specified bounds with optimized performance for millions of points.

        Args:
            min_bounds: Minimum bounds (x, y, z) - tuple or array-like
            max_bounds: Maximum bounds (x, y, z) - tuple or array-like
            points: List of points or numpy array to filter [(x, y, z), ...] or shape (N, 3)

        Returns:
            Filtered numpy array of points within the specified bounds, shape (M, 3)
        """
        # Handle empty input
        if len(points) == 0:
            return np.array([]).reshape(0, 3)

        # Convert to numpy array if not already (optimized for large datasets)
        if not isinstance(points, np.ndarray):
            points_array = np.array(points, dtype=np.float32)
        else:
            points_array = points.astype(np.float32, copy=False)  # Avoid copy if already float32

        # Ensure 2D array shape (N, 3)
        if points_array.ndim == 1:
            points_array = points_array.reshape(1, -1)

        # Convert bounds to numpy arrays for vectorized operations
        min_bounds = np.array(min_bounds, dtype=np.float32)
        max_bounds = np.array(max_bounds, dtype=np.float32)

        # Check all dimensions simultaneously using broadcasting
        mask = np.all(
            (points_array >= min_bounds) & (points_array <= max_bounds),
            axis=1
        )
        # Return filtered points using boolean indexing
        return points_array[mask]

    def generate_occupancy_map_from_mesh(self, scene, mesh_path="/map_mesh", num_surface_samples=200000, num_free_samples=10000, voxel_pitch=0.2):
        """
        Generate an occupancy map from a unified mesh using trimesh.

        Args:
            scene: Scene configuration with env_spacing parameter
            mesh_path: Path to the unified mesh primitive
            num_surface_samples: Number of surface samples for occupied points
            num_free_samples: Number of free space samples

        Returns:
            KDTree of occupied points, occupied points array, free points array
        """
        print(f"Generating occupancy map from mesh: {mesh_path}")
        start_time = time.time()

        # Get the mesh from USD stage
        stage = omni.usd.get_context().get_stage()
        mesh_prim = stage.GetPrimAtPath(mesh_path)

        if not mesh_prim.IsValid():
            print(f"Warning: Mesh prim at {mesh_path} not found, using fallback points")
            fallback_points = np.array([[0, 0, 0]], dtype=np.float32)
            return KDTree(fallback_points), fallback_points, fallback_points

        # Convert USD mesh to trimesh
        mesh_geom = UsdGeom.Mesh(mesh_prim)
        vertices = mesh_geom.GetPointsAttr().Get()
        face_vertex_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

        if not vertices or not face_vertex_indices:
            print("Warning: Empty mesh data, using fallback points")
            fallback_points = np.array([[0, 0, 0]], dtype=np.float32)
            return KDTree(fallback_points), fallback_points, fallback_points

        # Convert to numpy arrays
        vertices_np = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=np.float32)

        # Convert face data to triangles (trimesh expects triangular faces)
        faces = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:
                # Triangle face
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 1],
                    face_vertex_indices[idx + 2]
                ])
            elif count == 4:
                # Quad face - split into two triangles
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 1],
                    face_vertex_indices[idx + 2]
                ])
                faces.append([
                    face_vertex_indices[idx],
                    face_vertex_indices[idx + 2],
                    face_vertex_indices[idx + 3]
                ])
            idx += count

        faces_np = np.array(faces, dtype=np.int32)

        # Create trimesh object
        try:
            mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
            print(f"Created trimesh with {len(vertices_np)} vertices and {len(faces_np)} faces")
        except Exception as e:
            print(f"Error creating trimesh: {e}, using fallback points")
            fallback_points = np.array([[0, 0, 0]], dtype=np.float32)
            return KDTree(fallback_points), fallback_points, fallback_points

        # Surface sampling for occupied points
        surface_time = time.time()
        try:
            # Use even surface sampling for better distribution
            occupied_points, _ = trimesh.sample.sample_surface_even(
                mesh, num_surface_samples, seed=int(np.random.randint(0, 2**31))
            )
            print(f"Surface sampling: {time.time() - surface_time:.3f} seconds")
        except Exception as e:
            print(f"Error in surface sampling: {e}, using mesh vertices as occupied points")
            occupied_points = vertices_np

        # Randomly sample twice the target number of points to limit computation
        free_time = time.time()
        min_bounds = self._translate_to_origin((
            -scene.env_spacing / 2.0 + 1.0,
            -scene.env_spacing / 2.0 + 1.0,
            0.5
        ))
        max_bounds = self._translate_to_origin((
            scene.env_spacing / 2.0 - 1.0,
            scene.env_spacing / 2.0 - 1.0,
            1.5
        ))
        bbox_min = np.array(min_bounds, dtype=np.float32)
        bbox_max = np.array(max_bounds, dtype=np.float32)

        # Generate a regular 3D grid of candidate points within bounds
        x_vals = np.arange(bbox_min[0], bbox_max[0], voxel_pitch, dtype=np.float32)
        y_vals = np.arange(bbox_min[1], bbox_max[1], voxel_pitch, dtype=np.float32)
        z_vals = np.arange(bbox_min[2], bbox_max[2], voxel_pitch, dtype=np.float32)
        xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing='xy')
        grid_pts = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1)

        # Randomly sample twice the target number of points to limit computation
        num_candidates = min(len(grid_pts), num_free_samples * 2)
        candidate_indices = np.random.choice(len(grid_pts), size=num_candidates, replace=False)
        sample_pts = grid_pts[candidate_indices]

        # Efficient free space sampling via downward ray intersection with even/odd parity test
        ray_directions = np.tile([0.0, 0.0, -1.0], (len(sample_pts), 1))
        # Batch compute all intersection locations and ray indices
        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=sample_pts,
            ray_directions=ray_directions,
            multiple_hits=True
        )
        # Count hits per ray
        counts = np.bincount(index_ray, minlength=len(sample_pts))
        # Free points are outside mesh (even number of intersections - 0, 2, 4, ...)
        # Occupied points are inside mesh (odd number of intersections - 1, 3, 5, ...)
        free_mask = (counts % 2) == 0
        free_points = sample_pts[free_mask]
        print(f"Downward ray even/odd parity: tested {len(sample_pts)} rays, found {len(free_points)} free points")

        # Downsample free points to exact target count if necessary
        if len(free_points) > num_free_samples:
            chosen = np.random.choice(len(free_points), size=num_free_samples, replace=False)
            free_points = free_points[chosen]
            print(f"Downsampled to {len(free_points)} free points")

        # Handle empty points case
        if len(occupied_points) == 0:
            print("No occupied points found, using fallback")
            occupied_points = np.array([[0, 0, 0]], dtype=np.float32)

        if len(free_points) == 0:
            print("No free points found, using fallback")
            free_points = np.array([[0, 0, 1]], dtype=np.float32)

        # Create KDTree for efficient nearest neighbor queries
        kdtree = KDTree(occupied_points)

        # Filter free points by minimum clearance distance to obstacles
        clearance_time = time.time()
        min_clearance = 0.20  # Minimum clearance distance in meters

        if len(free_points) > 0 and len(occupied_points) > 0:
            # Query distances to nearest occupied points for all free points
            distances, _ = kdtree.query(free_points, k=1)

            # Keep only free points that have sufficient clearance
            clearance_mask = distances >= min_clearance
            filtered_free_points = free_points[clearance_mask]

            print(f"Clearance filtering: kept {len(filtered_free_points)}/{len(free_points)} free points "
                  f"with min clearance {min_clearance}m, time: {time.time() - clearance_time:.3f} seconds")
        else:
            filtered_free_points = free_points
        print(f"Filtered free points: {len(filtered_free_points)}, time: {time.time() - free_time:.3f} seconds")

        total_time = time.time() - start_time
        print(f"Total trimesh-based occupancy map generation: {total_time:.3f} seconds")

        return kdtree, occupied_points, filtered_free_points



    def create_environment(self, scene, num_obstacles=100, min_distance=1.0, obstacle_size_range=(0.3, 0.8),
                      obstacle_height_range=(0.5, 3.0), num_floaters=50, floaters_size_range=(0.1, 0.5), floaters_height_range=(1.0, 4.0)):
        """
        Create a full environment with ground, walls, obstacles, and floating objects.

        Args:
            scene: Scene configuration with env_spacing parameter
            num_obstacles: Number of obstacles to generate
            min_distance: Minimum distance between obstacles
            obstacle_size_range: Range of obstacle sizes (min, max)
            obstacle_height_range: Range of obstacle heights (min, max)
            num_floaters: Number of floating objects to generate
            floaters_size_range: Range of floater sizes (min, max)
            floaters_height_range: Range of floating object heights (min, max)
        """
        if self._generation_in_progress:
            print(f"Warning: Generation already in progress for {self.base_prim}")
            return None

        self._generation_in_progress = True

        try:
            print(f"Generating environment for {self.base_prim}...")
            start_time = time.time()

            # Clean up existing environment first
            self._cleanup_environment()

            # Generate walls and ground (integrated)
            walls_time = time.time()
            walls = self.generate_walls(scene)
            print(f"Walls and ground generation: {time.time() - walls_time:.3f} seconds")

            # Generate obstacles
            obstacles_time = time.time()
            obstacles, obstacle_positions = self.generate_obstacles(
                scene,
                num_obstacles=num_obstacles,
                min_distance=min_distance,
                obstacle_size_range=obstacle_size_range,
                height_range=obstacle_height_range
            )
            print(f"Obstacles generation: {time.time() - obstacles_time:.3f} seconds")

            # Generate floating objects
            floaters_time = time.time()
            # Only generate floaters if requested
            if num_floaters > 0:
                floaters, floater_positions = self.generate_floaters(
                    scene,
                    num_floaters=num_floaters,
                    min_distance=min_distance,
                    floater_size_range=floaters_size_range,
                    height_range=floaters_height_range
                )
            else:
                floaters, floater_positions = [], []
            print(f"Floaters generation: {time.time() - floaters_time:.3f} seconds")

            # Generate flattened mesh for the entire environment
            flatten_time = time.time()
            stage = omni.usd.get_context().get_stage()
            merged_mesh_path = "/map_mesh"
            if prims_utils.is_prim_path_valid(merged_mesh_path):
                prims_utils.delete_prim(merged_mesh_path)
                print(f"Deleted existing merged mesh at: {merged_mesh_path}")

            # Calculate bounds based on environment spacing
            env_space = scene.env_spacing / 2.0
            min_bound = (-env_space * 2, -env_space * 2, 0.0)
            max_bound = (env_space * 2, env_space * 2, 5.0)

            flattened_mesh_path = flatten_instancer_and_mesh(
                stage,
                "/World/ground",
                merged_mesh_path,
                min_bound=min_bound,
                max_bound=max_bound,
                env_space=env_space
            )
            print(f"Environment flattening: {time.time() - flatten_time:.3f} seconds")

            # Generate occupancy map from the unified mesh
            occ_time = time.time()
            kdtree, points, free_points = self.generate_occupancy_map_from_mesh(
                scene,
                mesh_path=merged_mesh_path,
                num_surface_samples=800000,
                num_free_samples=10000
            )
            print(f"Occupancy map generation: {time.time() - occ_time:.3f} seconds")

            total_time = time.time() - start_time
            print(f"Total environment generation time for {self.base_prim}: {total_time:.3f} seconds")
            return {
                "walls": walls,
                "obstacles": obstacles,
                "obstacle_positions": obstacle_positions,
                "floaters": floaters,
                "floater_positions": floater_positions,
                "kdtree": kdtree,
                "points": points,
                "free_points": free_points,
                "flattened_mesh": flattened_mesh_path,
                "generation_time": total_time,
                "map_origin": self.map_origin,
                "base_prim": self.base_prim
            }
        finally:
            self._generation_in_progress = False

    def _cleanup_environment(self):
        """Clean up existing environment components."""
        # Remove existing instancers and prototypes
        instancers = [
            "cuboid_instancer", "cylinder_instancer",
            "cuboid_floater_instancer", "cylinder_floater_instancer", "cone_floater_instancer"
        ]

        for instancer in instancers:
            instancer_path = self._get_prim_path(instancer)
            if prims_utils.is_prim_path_valid(instancer_path):
                prims_utils.delete_prim(instancer_path)

        # Remove prototypes
        prototypes_path = self._get_prim_path("prototypes")
        if prims_utils.is_prim_path_valid(prototypes_path):
            prims_utils.delete_prim(prototypes_path)

        # Remove walls and ground plane
        walls_and_ground = ["left_wall", "right_wall", "forward_wall", "backward_wall", "ceiling", "ground_plane"]
        for wall in walls_and_ground:
            wall_path = self._get_prim_path(wall)
            if prims_utils.is_prim_path_valid(wall_path):
                prims_utils.delete_prim(wall_path)

    def is_generation_in_progress(self):
        """Check if map generation is currently in progress."""
        return self._generation_in_progress

    def _enable_viewport_grid(self):
        """
        Enable the viewport grid by default using carb settings interface.
        """
        try:
            import carb
            settings = carb.settings.get_settings()
            # Enable the viewport grid
            settings.set("/app/viewport/grid/enabled", True)
            print("Viewport grid enabled by default")
        except ImportError:
            print("Warning: carb module not available, cannot enable grid automatically")
        except Exception as e:
            print(f"Warning: Failed to enable viewport grid: {e}")
