from utils import *
import open3d as o3d
import numpy as np

def parse_conf():
    return {
        "pcd_path": "./data/column_metashape.ply",
        "voxel_size": 0.02,
        "visualization": False
    }

def ground_extraction(pcd_path:str, voxel_size:float, visualization:bool=False):
    '''take point cloud and downsampling voxel size as input, required manually pick ground point, then use the plane segmentation to extract the ground plane equation.
    '''
    pcd = o3d.io.read_point_cloud(pcd_path)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print("after down sampling, the point cloud is {}".format(voxel_down_pcd))
    filtered_pcd, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    print("after radius oulier removal, the point cloud is {}".format(filtered_pcd))
    plane_points_id_list, plane_points_v_list, plane_points_ev_list = extract_plane_points(filtered_pcd, voxel_size)

    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(np.asarray(filtered_pcd.points)[plane_points_id_list])
    plane_pcd.colors = o3d.utility.Vector3dVector(np.asarray(filtered_pcd.colors)[plane_points_id_list])
    picked_points = pick_points(plane_pcd)
    print("the id of plane point picked is {}".format(pick_points[0]))

    C_list = find_similar_ev_points(plane_points_ev_list[picked_points[0]], plane_points_id_list, plane_points_ev_list)
    print("number of similar normal = {}".format(len(C_list)))
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(np.asarray(filtered_pcd.points)[C_list])
    ground_pcd.paint_uniform_color([1,0,0])

    segment_plane = ground_pcd.segment_plane(distance_threshold=0.02, ransac_n=10, num_iterations=100)
    params = segment_plane[0]
    print("the equation of plane is {:.4f}x + {:.4f}y +{:.4f}z +{:.4f} = 0".format(params[0], params[1], params[2], params[3]))

    np.asarray(ground_pcd.colors)[segment_plane[1]] = np.array([0,1,0])
    if (visualization):
        o3d.visualization.draw_geometries([ground_pcd])
    return params

if __name__=="__main__":
    ground_extraction(**parse_conf())