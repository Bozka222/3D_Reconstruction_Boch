import cv2 as cv
import os
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def draw_epipolar_lines(pts1, pts2, img1, img2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.subplot(121)
    plt.imshow(img5)
    plt.subplot(122)
    plt.imshow(img3)
    plt.show()

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines'''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def reprojection_loss_function(opt_variables, points_2d, num_pts):
    '''opt_variables --->  Camera Projection matrix + All 3D points'''
    P = opt_variables[0:12].reshape(3, 4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]
        # print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])

    return np.array(rep_error).ravel()


def bundle_adjustment(points_3d, points_2d, img, projection_matrix):
    opt_variables = np.hstack((projection_matrix.ravel(), points_3d.ravel(order="F")))
    num_points = len(points_2d[0])

    corrected_values = least_squares(reprojection_loss_function, opt_variables, args=(points_2d, num_points))

    print("The optimized values \n" + str(corrected_values))
    P = corrected_values.x[0:12].reshape(3, 4)
    points_3d = corrected_values.x[12:].reshape((num_points, 4))

    return P, points_3d


def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3, 4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]

        print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])


curr_dir_path = os.getcwd()
images_dir = curr_dir_path + "/Data/Input/Images_With_BG"
filenames = []
if __name__ == "__main__":

    cv_file = cv.FileStorage()
    cv_file.open('Data/Input/stereoMap.xml', cv.FileStorage_READ)
    K = cv_file.getNode('KL').mat()
    print(K)

    # Variables 
    iter = 0
    prev_img = None
    prev_kp = None
    prev_desc = None
    K = np.array(K, dtype=np.float32)
    R_t_0 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
    R_t_1 = np.empty((3, 4))
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3, 4))
    X = np.array([])
    Y = np.array([])
    Z = np.array([])

    for filename in os.listdir(images_dir):
        
        file = os.path.join(images_dir, filename)
        img = cv.imread(file, 0)
        filenames.append(filename)

        resized_img = img
        sift = cv.SIFT.create()
        kp, desc = sift.detectAndCompute(resized_img, None)
        
        if iter == 0:
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc
        else:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(prev_desc, desc, k=2)
            good = []
            pts1 = []
            pts2 = []
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    good.append(m)
                    pts1.append(prev_kp[m.queryIdx].pt)
                    pts2.append(kp[m.trainIdx].pt)
                    
            pts1 = np.array(pts1)
            pts2 = np.array(pts2)
            F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
            print("The fundamental matrix \n" + str(F))

            # We select only inlier points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            draw_epipolar_lines(pts1, pts2, prev_img, resized_img)

            E = np.matmul(np.matmul(np.transpose(K), F), K)

            print("The new essential matrix is \n" + str(E))

            retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
            
            print("I+0 \n" + str(R_t_0))

            print("Mullllllllllllll \n" + str(np.matmul(R, R_t_0[:3, :3])))

            R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
            R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())

            print("The R_t_0 \n" + str(R_t_0))
            print("The R_t_1 \n" + str(R_t_1))

            P2 = np.matmul(K, R_t_1)

            print("The projection matrix 1 \n" + str(P1))
            print("The projection matrix 2 \n" + str(P2))

            pts1 = np.transpose(pts1)
            pts2 = np.transpose(pts2)

            print("Shape pts 1\n" + str(pts1.shape))

            points_3d = cv.triangulatePoints(P1, P2, pts1, pts2)
            points_3d /= points_3d[3]

            # P2, points_3D = bundle_adjustment(points_3d, pts2, resized_img, P2)
            opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
            num_points = len(pts2[0])
            rep_error_fn(opt_variables, pts2, num_points)

            X = np.concatenate((X, points_3d[0]))
            Y = np.concatenate((Y, points_3d[1]))
            Z = np.concatenate((Z, points_3d[2]))

            points = np.zeros((len(X), 3))
            points[:, 0] = X
            points[:, 1] = Y
            points[:, 2] = Z

            R_t_0 = np.copy(R_t_1)
            P1 = np.copy(P2)
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc

        iter = iter + 1

pcd = o3d.geometry.PointCloud()
pc_points = np.array(points, np.float32)
pc_color = np.array([], np.float32)

# # RGB to BGR for pc_points
# img_color = cv.cvtColor(img, cv.COLOR_RGB2BGR)
# point_color = np.transpose(pts1)
#
# for i in range(len(point_color)):
#     u = np.int32(point_color[i][1])
#     v = np.int32(point_color[i][0])
#
#     pc_color = np.append(pc_color, np.array(np.float32(img_color[u][v] / 255)))
#     pc_color = np.reshape(pc_color, (-1, 3))

pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.colors = o3d.utility.Vector3dVector(pc_color)
o3d.visualization.draw_geometries([pcd], window_name="Stereo point cloud", width=1280, height=720)
o3d.io.write_point_cloud("Data/Output/PointClouds/BundleAdjustment.ply", pcd, format="PLY")
