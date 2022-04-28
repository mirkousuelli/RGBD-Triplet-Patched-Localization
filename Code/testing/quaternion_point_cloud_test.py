import numpy as np
import open3d as o3d
import copy
from utils.utils import *
from camera.Frame import Frame

first = 0
second = 60
frame1 = Frame(
	get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, first),
	get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, first),
	get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
	first
)
frame2 = Frame(
	get_rgb_triplet_dataset_path("../../Dataset", "Testing", 2, second),
	get_depth_triplet_dataset_path("../../Dataset", "Testing", 2, second),
	get_pose_triplet_dataset_path("../../Dataset", "Testing", 2),
	second
)

color_raw_1 = o3d.io.read_image('../../Dataset/Testing/2/Colors/00000-color.png')
depth_raw_1 = o3d.io.read_image('../../Dataset/Testing/2/Depths/00000-depth.png')
rgbd_image_1 = o3d.geometry.RGBDImage.create_from_tum_format(color_raw_1, depth_raw_1)
dopo = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_1, depth_raw_1, convert_rgb_to_intensity=False)
rgbd_image_1.color = dopo.color
pcd_1 = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image_1,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
)

pcd_1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh_t1 = copy.deepcopy(mesh).translate(
	(frame1.t[0], frame1.t[1], frame1.t[2])
)
R1 = mesh.get_rotation_matrix_from_quaternion(
	(frame1.pose[0], frame1.pose[1], frame1.pose[2], frame1.pose[3])
)
mesh_t1.rotate(R1, center=mesh_t1.get_center())
mesh_t2 = copy.deepcopy(mesh).translate(
	(frame2.t[0], frame2.t[1], frame2.t[2])
)
R2 = mesh.get_rotation_matrix_from_quaternion(
	(frame2.pose[0], frame2.pose[1], frame2.pose[2], frame2.pose[3])
)
mesh_t2.rotate(R2, center=mesh_t2.get_center())

pcd_1.translate(mesh_t1.get_center() - pcd_1.get_center())

print()
print(f'Center of mesh: {mesh.get_center()}')
print()
print(f'Center of mesh t1: {mesh_t1.get_center()}')
print(f'Rotation mesh t1: {mesh_t1.get_rotation_matrix_from_xyz(mesh_t1.get_center())}')
print()
print(f'Center of mesh t2: {mesh_t2.get_center()}')
print(f'Rotation mesh t2: {mesh_t2.get_rotation_matrix_from_xyz(mesh_t2.get_center())}')
print()
print(f'Image 1: {pcd_1.get_center()}')
print(f'Rotation mesh t1: {mesh_t1.get_rotation_matrix_from_xyz(mesh_t1.get_center())}')
o3d.visualization.draw_geometries([mesh_t1, pcd_1])
