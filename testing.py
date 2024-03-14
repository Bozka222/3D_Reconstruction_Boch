import numpy as np
import open3d as o3d
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(
                                                                radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    print(result.transformation)
    return result


source = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/Cropped/3D_Cam_PointCloud_Cropped3.ply")
target = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/Cropped/3D_Cam_PointCloud_Cropped4.ply")

# o3d.visualization.draw_geometries([source])
# o3d.visualization.draw_geometries([target])
voxel_size = 0.050
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print(result_ransac)
global_registration = result_ransac.transformation
draw_registration_result(source_down, target_down, global_registration)

# trans_init = np.asarray([[1.0, 0.0, 0.0, -0.00026],
#                          [0.0, 1.0, 0.0, -0.0032],
#                          [0.0, 0.0, 1.0, -0.00016],
#                          [0.0, 0.0, 0.0, 1.0]])
# draw_registration_result(source, target, trans_init)

threshold = 0.02
print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, global_registration)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, global_registration,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

# threshold = 0.02
# print("Apply point-to-plane ICP")
# reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
#                                                       o3d.pipelines.registration.TransformationEstimationPointToPlane())
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# draw_registration_result(source, target, reg_p2p.transformation)
