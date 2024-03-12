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
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    # demo_icp_pcds = o3d.data.DemoICPPointClouds()
    # source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    # target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    source = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/3D_Camera_PointCloud2.ply")
    target = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/3D_Camera_PointCloud1.ply")
    source.estimate_normals()
    source.orient_normals_consistent_tangent_plane(50)
    target.estimate_normals()
    target.orient_normals_consistent_tangent_plane(50)

    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


voxel_size = 0.0001  # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
print(result_ransac)
draw_registration_result(source_down, target_down, result_ransac.transformation)

result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size)
print(result_icp)
draw_registration_result(source, target, result_icp.transformation)



# print("Initial alignment")
# evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
# print(evaluation)

# voxel_size = 0.02
# max_correspondence_distance_coarse = voxel_size * 15
# max_correspondence_distance_fine = voxel_size * 1.5
#
#
# def load_point_clouds():
#     pcds = []
#     for i in range(7):
#         pcd = o3d.io.read_point_cloud("Data/Output/PointClouds/3D_Cam/3D_Camera_PointCloud%d.ply" % i)
#         # pcd_down = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=voxel_size)
#         pcds.append(pcd)
#     return pcds
#
#
# def pairwise_registration(source, target):
#     print("Apply point-to-plane ICP")
#     icp_coarse = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance_coarse, np.identity(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPlane())
#     icp_fine = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance_fine,
#         icp_coarse.transformation,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane())
#     transformation_icp = icp_fine.transformation
#     information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
#         source, target, max_correspondence_distance_fine,
#         icp_fine.transformation)
#     return transformation_icp, information_icp
#
#
# def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
#     pose_graph = o3d.pipelines.registration.PoseGraph()
#     odometry = np.identity(4)
#     pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
#     n_pcds = len(pcds)
#     for source_id in range(n_pcds):
#         for target_id in range(source_id + 1, n_pcds):
#             transformation_icp, information_icp = pairwise_registration(
#                 pcds[source_id], pcds[target_id])
#             print("Build o3d.registration.PoseGraph")
#             if target_id == source_id + 1:  # odometry case
#                 odometry = np.dot(transformation_icp, odometry)
#                 pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
#                 pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                                                  target_id,
#                                                                                  transformation_icp,
#                                                                                  information_icp,
#                                                                                  uncertain=False))
#             else:  # loop closure case
#                 pose_graph.edges.append(
#                     o3d.pipelines.registration.PoseGraphEdge(source_id,
#                                                              target_id,
#                                                              transformation_icp,
#                                                              information_icp,
#                                                              uncertain=True))
#     return pose_graph
#
#
# o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
# pcds_down = load_point_clouds()
# o3d.visualization.draw_geometries(pcds_down)
# pcds_with_normals = []
#
# for pcd in pcds_down:
#     pcd.estimate_normals()
#     pcd.orient_normals_consistent_tangent_plane(50)
#     pcds_with_normals.append(pcd)
#
#
# print("Full registration ...")
# pose_graph = full_registration(pcds_with_normals, max_correspondence_distance_coarse, max_correspondence_distance_fine)
#
# print("Optimizing PoseGraph ...")
# option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=max_correspondence_distance_fine,
#                                                    edge_prune_threshold=0.25,
#                                                    reference_node=0)
# o3d.pipelines.registration.global_optimization(pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
#                                      o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)
#
# print("Transform points and display")
# for point_id in range(len(pcds_with_normals)):
#     print(pose_graph.nodes[point_id].pose)
#     pcds_with_normals[point_id].transform(pose_graph.nodes[point_id].pose)
# o3d.visualization.draw_geometries(pcds_with_normals)
#
# print("Make a combined point cloud")
# pcds = load_point_clouds()
# pcd_combined = o3d.geometry.PointCloud()
# for point_id in range(len(pcds)):
#     pcds[point_id].transform(pose_graph.nodes[point_id].pose)
#     pcd_combined += pcds[point_id]
# # pcd_combined_down = o3d.geometry.voxel_down_sample(pcd_combined, voxel_size=voxel_size)
#
# o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined)
# o3d.visualization.draw_geometries([pcd_combined])
