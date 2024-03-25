import open3d as o3d
import numpy as np
import copy
import glob


def convertTuple(tup):
    str = ''
    for item in tup:
        str = str + item
    return str


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])
    return source_temp, target


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(
                                                                radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and compute normals.")

    source = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/Cropped/3D_Cam_PointCloud_Cropped3.ply")
    target = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/Cropped/3D_Cam_PointCloud_Cropped4.ply")
    target.estimate_normals()
    # target.orient_normals_consistent_tangent_plane(100)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# SLOW RANSAC ALGORITHM
# def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 1.5
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("Since the downsampling voxel size is %.3f," % voxel_size)
#     print("we use a liberal distance threshold %.3f." % distance_threshold)
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
#         [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
#         o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
#     return result


# FASTER ZHOU ALGORITHM
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.05
    print(":: Point-to-plane ICP registration is applied on original point")
    print("clouds to refine the alignment. This time we use a strict")
    print("distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


voxel_size = 0.0002  # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

# Compute FAST Global registration
result_ransac = execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print(result_ransac)
draw_registration_result(source_down, target_down, result_ransac.transformation)

# Compute local ICP registration (PointToPlane())
result_icp_PointToPlane = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
print(result_icp_PointToPlane)
draw_registration_result_original_color(source, target, result_icp_PointToPlane.transformation)

# Compute Color Multi-Scale ICP transformation
voxel_radius = [0.0004, 0.0003, 0.0002, 0.0001]
max_iter = [200, 150, 100, 50]
current_transformation = np.identity(4)
print("3. Colored point cloud registration")
for scale in range(4):
    iter = max_iter[scale]
    radius = voxel_radius[scale]
    print([iter, radius, scale])

    print("3-1. Downsample with a voxel size %.2f" % radius)
    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)

    print("3-2. Estimate normal.")
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

    print("3-3. Applying colored point cloud registration")
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, radius, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
    current_transformation = result_icp.transformation
    print(result_icp)
source_temp, target = draw_registration_result_original_color(source, target, result_icp.transformation)
pcd_combined = o3d.geometry.PointCloud()
pcd_combined = source_temp + target
o3d.io.write_point_cloud("Data/Output/PointClouds/3D_Cam/Combined_3DCam_PointCloud.ply", pcd_combined)
