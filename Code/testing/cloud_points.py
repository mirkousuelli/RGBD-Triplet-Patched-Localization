# cloud points prova
import open3d as o3d
from matplotlib import pyplot as plt


color_raw = o3d.io.read_image('../../Dataset/Colors/00000-color.png')
depth_raw = o3d.io.read_image('../../Dataset/Depths/00000-depth.png')
rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)
dopo = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
print(rgbd_image)

rgbd_image.color = dopo.color

plt.subplot(1, 2, 1)
plt.title('RGB image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Depth image')
plt.imshow(rgbd_image.depth)
plt.show()

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Flip it, otherwise the point cloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.25)
vis.run()