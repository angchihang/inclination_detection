import open3d as o3d
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

def extract_plane_points(pcd, voxel_size):
    pcd.paint_uniform_color([0,1,0])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pca = PCA(n_components=3, svd_solver='full')
    linearity_list = []
    planarity_list = []
    eigen_value_list = []
    eigenvector_list = []
    for i in tqdm(np.arange(np.asarray(pcd.points).shape[0])):
        [k, idx, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[i], 3*voxel_size, 30)
        if (k > 3):
            pca.fit(np.asarray(pcd.points)[idx[1:]])
            eigenvalue = (pca.explained_variance_)
            eigenvector = pca.components_

            # extract hand-crafted feature by eigen value
            linearity = (eigenvalue[0] - eigenvalue[1]) / eigenvalue[0]
            planarity = (eigenvalue[1] - eigenvalue[2]) / eigenvalue[0]
            alpha_3d = eigenvalue[2] / eigenvalue[0]

            assert linearity + planarity + alpha_3d - 1 < 0.001        
        else:
            linearity = 0.33
            planarity = 0.33
            alpha_3d = 0.33

        linearity_list.append(linearity)
        planarity_list.append(planarity)
        eigen_value_list.append(eigenvalue)
        eigenvector_list.append(eigenvector)
    plane_points_list = []
    plane_point_v_list = []
    plane_points_ev_list = []

    for i in tqdm(np.arange(len(linearity_list))):
        if (planarity_list[i] > 0.4):
            plane_points_list.append(i)
            plane_point_v_list.append(eigen_value_list[i])
            plane_points_ev_list.append(eigenvector_list[i])
    plane_points_list = np.array(plane_points_list)
    plane_point_v_list = np.array(plane_point_v_list)
    plane_points_ev_list = np.array(plane_points_ev_list)
    return plane_points_list, plane_point_v_list, plane_points_ev_list

def find_similar_ev_points(src_ev: np.ndarray, tgt_points_id: np.ndarray, tgt_evs: np.ndarray):
    '''Given a source point eigen vector, return the point id(s) in target points that have similar first eigenvector
    
    Args:
        src_ev (np.ndarray): source eigen vector
        tgt_points_id (np.ndarray):  target points id(s) from original pcd
        tgt_evs (np.ndarray):    the eigen vectors of target points
    
    Returns:
        np.ndarray: point id(s) that have similar first eigen vector
    '''
    deg_threshold = 10
    C_list = []
    src_ev1 = src_ev[0]

    for index in range(len(tgt_evs)):
        tgt_ev1 = tgt_evs[index][0]
        if (np.abs(np.dot(src_ev1, tgt_ev1)) > np.cos(deg_threshold*np.pi/180)):
            C_list.append(tgt_points_id[index])
    return np.array(C_list)

def pick_points(pcd, prompt:str = "pick points"):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print(prompt)
    return vis.get_picked_points()

def angle_between_plane(n1: np.ndarray, n2: np.ndarray):
    angle = np.arccos(np.abs(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))) * 180 / np.pi
    return angle

def extract_plane(voxel_down_pcd, plane_points_list_1, plane_points_list_2):
    ground_pcd_1 = o3d.geometry.PointCloud()
    ground_pcd_1.points = o3d.utility.Vector3dVector(np.asarray(voxel_down_pcd.points)[plane_points_list_1])
    
    segment_plane1 = ground_pcd_1.segment_plane(distance_threshold=0.005, ransac_n=4, num_iterations=1000)

    ground_pcd_2 = o3d.geometry.PointCloud()
    ground_pcd_2.points = o3d.utility.Vector3dVector(np.asarray(voxel_down_pcd.points)[plane_points_list_2])
    
    segment_plane2 = ground_pcd_2.segment_plane(distance_threshold=0.005, ransac_n=4, num_iterations=1000)

    return ground_pcd_1, ground_pcd_2, segment_plane1, segment_plane2

def fit_plane_ransac(points, n_iterations=100, distance_threshold=0.01):
    best_plane = None
    best_inliers = []
    
    for _ in range(n_iterations):
        # Step 1: Randomly select 3 points
        random_indices = np.random.choice(len(points), 3, replace=False)
        random_subset = points[random_indices]

        # Step 2: Fit a plane to the random subset
        normal = fit_plane(random_subset)
        p0 = random_subset[0]

        # Step 3: Evaluate consensus
        distances = compute_distances(points, p0, normal)
        inliers = distances < distance_threshold

        # Step 4: Update best model if current model is better
        if inliers.sum() > len(best_inliers):
            best_inliers = inliers

    # Step 5: Select best model
    best_plane_normal = fit_plane(points[best_inliers])

    return best_plane_normal, best_inliers

def fit_plane(points):
    p1, p2, p3 = points[0], points[1], points[2]
    vec1 = p1 - p2
    vec2 = p1 - p3
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(normal)
    return normal

def compute_distances(points, p0, normal):
    # Compute the signed distance from each point to the plane
    p0pn_vec = points - p0
    dist = np.dot(p0pn_vec, normal.reshape((3,1))) / np.linalg.norm(normal)
    return dist.flatten()