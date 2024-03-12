import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/MergedPointCloud_Skoda.ply")
# o3d.visualization.draw_geometries([pcd])


print("Every 5th points are selected")
uni_down_pcd = pcd.uniform_down_sample(every_k_points=10)
# o3d.visualization.draw_geometries([uni_down_pcd])

print("Statistical outlier removal")
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=5.0)
display_inlier_outlier(pcd, ind)

inlier_cloud = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([inlier_cloud])
