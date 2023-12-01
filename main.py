from utils import *
import open3d as o3d
import numpy as np

def parse_conf():
    return {
        "pcd_path": "./data/book.ply",
        "voxel_size": 0.1,
        "visualization": True
    }

def main(pcd_path, voxel_size, visualization):
    pcd = o3d.io.read_point_cloud(pcd_path)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    plane_points_id_list, plane_points_v_list, plane_points_ev_list = extract_plane_points(voxel_down_pcd, voxel_size)

    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(np.asarray(voxel_down_pcd.points)[plane_points_id_list])
    points = pick_points(plane_pcd)

    assert len(points) == 2, 'Please pick two points.'
    plane1_point_id, plane2_point_id = points

    plane_points_list_1 = find_similar_ev_points(plane_points_ev_list[plane1_point_id], plane_points_id_list, plane_points_ev_list)
    plane_points_list_2 = find_similar_ev_points(plane_points_ev_list[plane2_point_id], plane_points_id_list, plane_points_ev_list)
    print("plane 1 point's number = {}, plane 2 point's number = {}".format(len(plane_points_list_1), len(plane_points_list_2)))

    ground_pcd_1 = o3d.geometry.PointCloud()
    ground_pcd_1.points = o3d.utility.Vector3dVector(np.asarray(voxel_down_pcd.points)[plane_points_list_1])
    
    segment_plane1 = ground_pcd_1.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    n1 = segment_plane1[0][:3]

    ground_pcd_2 = o3d.geometry.PointCloud()
    ground_pcd_2.points = o3d.utility.Vector3dVector(np.asarray(voxel_down_pcd.points)[plane_points_list_2])
    
    segment_plane2 = ground_pcd_2.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    n2 = segment_plane2[0][:3]

    print("n1 = {}, n2 = {}".format(n1, n2))

    if (visualization):
        inlier1 = segment_plane1[1]
        inlier_cloud1 = ground_pcd_1.select_by_index(inlier1)
        inlier_cloud1.paint_uniform_color([0, 1, 0])

        inlier2 = segment_plane2[1]
        inlier_cloud2 = ground_pcd_2.select_by_index(inlier2)
        inlier_cloud2.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([inlier_cloud2, inlier_cloud1])

    angle = angle_between_plane(n1, n2)
    print("angle between plane = {:.2f} deg".format(angle))

if __name__=="__main__":
    main(**parse_conf())