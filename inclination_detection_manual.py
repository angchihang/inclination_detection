from utils import *
import open3d as o3d
import numpy as np

def parse_conf():
    return {
        "pcd_path": "./data/book.ply",
        "voxel_size": 0.05,
        "visualization": False
    }

def main(pcd_path, voxel_size, visualization):
    pcd = o3d.io.read_point_cloud(pcd_path)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    firstPlanePointsIds = pick_points(voxel_down_pcd, "pick > 3 points for first plane")

    assert len(firstPlanePointsIds) >= 3, 'Please pick at least 3 points.'

    firstPlanePoints = np.asarray(voxel_down_pcd.points)[firstPlanePointsIds]    
    n1, fistPlanebest_inliers = fit_plane_ransac(firstPlanePoints)
    print("best plane normal = {}".format(n1))
    # print("best inlier = {}".format(fistPlanebest_inliers))

    secondPlanePointsIds = pick_points(voxel_down_pcd, "pick > 3 points for first plane")

    assert len(secondPlanePointsIds) >= 3, 'Please pick at least 3 points.'

    secondPlanePoints = np.asarray(voxel_down_pcd.points)[secondPlanePointsIds]
    n2, secondPlanebest_inliers = fit_plane_ransac(secondPlanePoints)
    print("best plane normal = {}".format(n2))

    
    print("n1 = {}, n2 = {}".format(n1, n2))

    angle = angle_between_plane(n1, n2)
    print("angle between plane = {:.2f} deg".format(angle))

if __name__=="__main__":
    main(**parse_conf())