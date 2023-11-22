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

def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()