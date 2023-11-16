import numpy as np
import cv2 as cv
import PIL.Image
import PIL.ExifTags
import json
import matplotlib.pyplot as plt
import open3d as o3d

CAMERA_CONFIG_PATH = "./camera_model.json"

class Frame:
    def __init__(self, input_path, resize_scale=4) -> None:
        self.input_path = input_path
        self.resize_scale = resize_scale
        self.load(input_path)
        self.get_camera_p(camera_model=self.camera_model)
        self.f = self.f / self.p
        pass

    def load(self, input_path):
        
        img = PIL.Image.open(input_path)
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS
        }
        self.camera_model = exif['Model']
        self.f = float(exif['FocalLength']) / (self.resize_scale)
        self.cx = exif['ExifImageWidth'] / (2 * self.resize_scale)
        self.cy = exif['ExifImageHeight'] / (2 * self.resize_scale)
        self.width = exif['ExifImageWidth']
        self.height = exif['ExifImageHeight']
        self.img = cv.resize(np.array(img), (int(self.width / self.resize_scale), int(self.height / self.resize_scale)))

    def get_camera_p(self, camera_model):
        with open(CAMERA_CONFIG_PATH, 'r') as f:
            cams = json.load(f)
            for cam in cams:
                if (cam['model'] == camera_model):
                    p = (cam['cmos_width'] / self.width + cam['cmos_height'] / self.height) / 2
                    self.p = p
    
    def get_intrinsic(self):
        K = np.array([
            [self.f, 0, self.cx],
            [0, self.f, self.cy],
            [0, 0, 1]
        ])
        return K
    
    def getNormalizationTrans(self):
        return np.array([[2/self.w, 0, -1], [0, 2/self.h, -1], [0, 0, 1]])
    
def essential_matrix_test(E, src_point ,tgt_point, camera_intrinsic):
    product = np.linalg.multi_dot([np.hstack((tgt_point, 1)), np.linalg.inv(camera_intrinsic).T, E, np.linalg.inv(camera_intrinsic), np.hstack((src_point, 1)).reshape((3,1))])
    if (product < 0.05):
        return True
    else:
        return False


srcList = []
tgtList = []
def main(visualization = False):
    src_frame = Frame("./data/small scale/column_left.JPG", resize_scale=1)
    tgt_frame = Frame("./data/small scale/column_right.JPG", resize_scale=1)


    src_gray = cv.cvtColor(np.array(src_frame.img),cv.COLOR_BGR2GRAY)
    tgt_gray = cv.cvtColor(np.array(tgt_frame.img),cv.COLOR_BGR2GRAY)

    # def onMouse(event, x, y, flags, param):
    #     global srcList
    #     if event == cv.EVENT_LBUTTONDOWN:
    #             srcList.append([x,y])
    #             print("point ({}, {}) picked".format(x, y))
    #             cv.circle(src_gray,(x,y),5,(255,0,0),-1)

    # cv.namedWindow(winname='src_image')
    # cv.setMouseCallback("src_image", onMouse)
    # while True: 
    #     cv.imshow('src_image',src_gray)
    #     if cv.waitKey(20) & 0xFF == 27:
    #         break
    # cv.destroyAllWindows()
    # src_points_to_estimate = np.array(srcList)
    # print(src_points_to_estimate)

    # def onMouse(event, x, y, flags, param):
    #     global tgtList
    #     if event == cv.EVENT_LBUTTONDOWN:
    #             tgtList.append([x,y])
    #             print("point ({}, {}) picked".format(x, y))
    #             cv.circle(tgt_gray,(x,y),5,(255,0,0),-1)

    # cv.namedWindow(winname='tgt_image')
    # cv.setMouseCallback("tgt_image", onMouse)
    # while True: 
    #     cv.imshow('tgt_image',tgt_gray)
    #     if cv.waitKey(20) & 0xFF == 27:
    #         break
    # cv.destroyAllWindows()
    # tgt_points_to_estimate = np.array(tgtList)
    # print(tgt_points_to_estimate)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src_frame.img,None)
    kp2, des2 = sift.detectAndCompute(tgt_frame.img,None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)
    gmatches = sorted(gmatches, key=lambda x: x.distance)

    src_points = np.array([kp1[m.queryIdx].pt for m in gmatches])
    tgt_points = np.array([kp2[m.trainIdx].pt for m in gmatches])

    camera_intrinsic = src_frame.get_intrinsic()
    print("camera intrinsic = {}".format(camera_intrinsic))
    E, mask = cv.findEssentialMat(src_points, tgt_points, camera_intrinsic)
    # print(essential_matrix_test(E, src_points[0], tgt_points[0], camera_intrinsic))
    _, R, t, mask_, triangulatedPoints = cv.recoverPose(E, src_points, tgt_points, camera_intrinsic, distanceThresh=50)
    print("relative R = {}".format(R))
    print("relative t = {}".format(t))
    triangulatedPoints = triangulatedPoints.T[np.ma.make_mask(mask_.flatten())]
    # print(triangulatedPoints[:, :3])
    triangulatedPoints = triangulatedPoints[:, :3] / triangulatedPoints[:, -1].reshape(len(triangulatedPoints), 1)

    # P_src = np.dot(camera_intrinsic, np.hstack((np.eye(3), np.zeros((3,1)))))
    # P_tgt = np.dot(camera_intrinsic, np.hstack((R, t)))
    # for (p1, p2) in zip(src_points_to_estimate, tgt_points_to_estimate):
    #     p1, p2 = np.array(p1), np.array(p2)
    #     print("src point, tgt point = {}, {}".format(p1, p2))
    #     X = cv.triangulatePoints(P_src, P_tgt, p1, p2)
    #     X = X/X[3]
    #     print(X/X[3])

    # points = cv.triangulatePoints(
    #     P_src, 
    #     P_tgt, 
    #     src_points.transpose(), 
    #     tgt_points.transpose() 
    #     ).transpose()  # shape: (N, 4)
    # points = points[:, :3] / points[:, 3:]
    
    if (visualization == True):
        src_camera = o3d.geometry.LineSet.create_camera_visualization(view_width_px=src_frame.width, view_height_px=src_frame.height, 
                                                                                intrinsic=src_frame.get_intrinsic(), extrinsic=np.eye(4))
        extrinsic = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
        tgt_camera = o3d.geometry.LineSet.create_camera_visualization(view_width_px=tgt_frame.width, view_height_px=tgt_frame.height, 
                                                                                intrinsic=tgt_frame.get_intrinsic(), extrinsic=extrinsic)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(triangulatedPoints[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector([[1,0,0]])
        pcd.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([src_camera, tgt_camera, pcd])

if __name__=="__main__":
    main(True)