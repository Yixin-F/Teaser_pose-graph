#!/usr/bin/env python3
import open3d as o3d
import teaserpp_python
import numpy as np 
import copy
import math
from helpers import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import rospy
from sensor_msgs.msg import PointCloud2,PointField
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import struct
import time
import tf.transformations as tr
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool
import sys

VOXEL_SIZE = 10 # (mm)
VISUALIZE = False  # True for test
FRAME_SIZE = 14 # Number of frames for reconstruction
WORLD_FRAME_START = True # world coordinate
RELO_MODE = 1  # TODO: check it, "0" for gloal matching, "1" for local matching
RECON_MODE = 0 # TODO: check it, "0" for "C path", "1" for "S path"
CROP_USE = True # crop
TRUSS_USE = False
REVISIT = 7 

X_INTERVAL = 600 # (mm)
Y_INTERVAL = 1200 # (mm)
OFFSET = -400 # (mm)

FOV_X = 1200 # (mm)
FOV_Y = 2400 # (mm)
FOV_OFFSET = 100 # (mm)
FOV_DIFF = 300 # (mm)

FEAT_OFFSET = 80 # (mm)

# C path with 14 waypoints
C_path =[
    [0, 0], [0, X_INTERVAL * 1], [0, X_INTERVAL * 2], [0, X_INTERVAL * 3], [0, X_INTERVAL * 4], [0, X_INTERVAL * 5], [0, X_INTERVAL * 6],
    [Y_INTERVAL, X_INTERVAL * 6], [Y_INTERVAL, X_INTERVAL * 5], [Y_INTERVAL, X_INTERVAL * 4], [Y_INTERVAL, X_INTERVAL * 3], [Y_INTERVAL, X_INTERVAL * 2], [Y_INTERVAL, X_INTERVAL * 1],
    [Y_INTERVAL, -50]
]

# S path with 21 waypoints
S_path = [
    [OFFSET, 0], 
    [OFFSET, X_INTERVAL * 1], [OFFSET, X_INTERVAL * 2], [OFFSET, X_INTERVAL * 3], [OFFSET, X_INTERVAL * 4], [OFFSET, X_INTERVAL * 5], 
    [OFFSET, X_INTERVAL * 6], [OFFSET + Y_INTERVAL, X_INTERVAL * 6],
    [OFFSET + Y_INTERVAL, X_INTERVAL * 5], [OFFSET + Y_INTERVAL, X_INTERVAL * 4], [OFFSET + Y_INTERVAL, X_INTERVAL * 3],
    [OFFSET + Y_INTERVAL, X_INTERVAL * 2], [OFFSET + Y_INTERVAL, X_INTERVAL * 1], [OFFSET + Y_INTERVAL, 0],
    [OFFSET + Y_INTERVAL * 2, 0],
    [OFFSET + Y_INTERVAL * 2, X_INTERVAL * 1], [OFFSET + Y_INTERVAL * 2, X_INTERVAL * 2], [OFFSET + Y_INTERVAL * 2, X_INTERVAL * 3],
    [OFFSET + Y_INTERVAL * 2, X_INTERVAL * 4], [OFFSET + Y_INTERVAL * 2, X_INTERVAL * 5], [OFFSET + Y_INTERVAL * 2, X_INTERVAL * 6]
]

# TODO: check for simulated relocated pose, you should get them from joint state
R_path = [
    [Y_INTERVAL, 300], [Y_INTERVAL, 900], [Y_INTERVAL, 1100]
]

def get_truss_T(xy_pos):
    eye_pos = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    eye_pos[0, 3] = xy_pos[0]
    eye_pos[1, 3] = xy_pos[1]

    return eye_pos

def get_roi(a_pcd, a_truss, b_pcd, b_truss):
    diff_truss = []
    diff_truss.append([(b_truss[0] - a_truss[0]), (b_truss[1] - a_truss[1])])
    a_min_x = - FOV_Y / 2 + (b_truss[0] - a_truss[0])
    a_max_x = FOV_Y / 2
    a_min_y = - FOV_X / 2 + (b_truss[1] - a_truss[1])
    a_max_y = FOV_X / 2
    b_min_x = - FOV_Y / 2
    b_max_x = FOV_Y / 2 - (b_truss[0] - a_truss[0])
    b_min_y = - FOV_X / 2
    b_max_y = FOV_X / 2 - (b_truss[1] - a_truss[1])
    a_min_bound = np.array([a_min_x - FOV_OFFSET, a_min_y - FOV_OFFSET, -10000.0])
    a_max_bound = np.array([a_max_x + FOV_OFFSET, a_max_y + FOV_OFFSET, 10000.0])
    b_min_bound = np.array([b_min_x - FOV_OFFSET, b_min_y - FOV_OFFSET, -10000.0])
    b_max_bound = np.array([b_max_x + FOV_OFFSET, b_max_y + FOV_OFFSET, 10000.0])

    if a_max_x - a_min_x <= FOV_DIFF or a_max_y - a_min_y <= FOV_DIFF or b_max_x - b_min_x < FOV_DIFF or b_max_y - b_min_y < FOV_DIFF:
        return a_pcd, b_pcd, False
    else:
        print(f"source_min_bound: {b_min_bound}, source_max_bound: {b_max_bound}")
        print(f"target_min_bound: {a_min_bound}, target_max_bound: {a_max_bound}")
        a_cropped_pcd = a_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(a_min_bound, a_max_bound))
        b_cropped_pcd = b_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(b_min_bound, b_max_bound))

        return a_cropped_pcd, b_cropped_pcd, True

# compute transform difference
def compute_transform_difference(T1, T2):
    T1_inv = np.linalg.inv(T1)
    T_diff = np.dot(T1_inv, T2)
    
    R_diff = T_diff[:3, :3]
    t_diff = T_diff[:3, 3]
    
    rotation_diff = R.from_matrix(R_diff).as_rotvec()
    
    return rotation_diff, t_diff

class SLAMNode:
    def __init__(self):
        self.source_pcd = None
        self.target_pcd = None
        self.buffer = []
        self.buffer_threshold = FRAME_SIZE
        self.out_flag = True

        self.waypoints = []

        self.relative_size = 0
        self.constrains_size = 0
        self.relative_pose_buffer = []

        self.relocalization_buffer = []
        self.pose_buffer = []
        self.pose_buffer_xyz = []
        self.opt_buffer = []
        self.before_optimized_pcd_dense = o3d.geometry.PointCloud()
        self.before_optimized_pcd_sparse = None
        self.optimized_pcd_dense = o3d.geometry.PointCloud() 
        self.relo_pcd = o3d.geometry.PointCloud() 
        self.optimized_pcd_sparse = None
        self.relo_flag = False
        self.relo_idx = 0
        self.near = 0

        self.kdtree = None # KDTree()

        self.source_sub = rospy.Subscriber('point_cloud_topic', PointCloud2, self.source_callback, queue_size = 100)
        self.pub = rospy.Publisher('reconstruction_topic', PointCloud2, queue_size = 10)
        self.opt_pub = rospy.Publisher('opt_reconstruction_topic', PointCloud2, queue_size = 10)
        self.pose_sub = rospy.Subscriber('opt_pose', Float32MultiArray, self.pose_callback, queue_size = 100)
        self.init_pub = rospy.Publisher("init_pose", PoseWithCovarianceStamped, queue_size = 10)
        self.reg_pub = rospy.Publisher("reg_pose", PoseWithCovarianceStamped, queue_size = 10)
        self.relo_pub = rospy.Publisher('relo_pose', PoseWithCovarianceStamped, queue_size = 10)
        self.pub_flag = rospy.Publisher('bool_topic', Bool, queue_size = 10)
        self.pub_tmp = rospy.Publisher('bool_topic_tmp', Bool, queue_size = 10)
        self.sub_tmp = rospy.Subscriber('bool_topic_tmp', Bool, self.source_callback3, queue_size = 10)

    def source_callback3(self, msg):
        if msg.data:
            print(f"recieved relo flag: {msg.data}")
            print("recived optimized poses and cloud ...")
            self.relo_pcd += self.optimized_pcd_sparse
            for i in range (len(self.relocalization_buffer)):
                print(f"begin to relo frame {i}")
                relo_pose = self.relocalization(self.relocalization_buffer[i], R_path[i])
                cloud_tmp = copy.deepcopy(self.relocalization_buffer[i])
                cloud_trans = cloud_tmp.transform(relo_pose)

                self.relo_pcd += cloud_trans
                self.pub_relo_pose(relo_pose, i)
                self.relo_idx += 1

            if VISUALIZE:
                o3d.visualization.draw_geometries([self.relo_pcd])

            globalpc = o3d.geometry.PointCloud()
            points_np = np.asarray(self.relo_pcd.points)
            globalpc.points = o3d.utility.Vector3dVector(points_np)
            o3d.io.write_point_cloud("reconstructionReview_relo.pcd", globalpc)
            print("save relo result...")

    def source_callback(self, msg):
        self.source_pcd = self.point_cloud2_to_open3d(msg)

        if len(self.buffer) < (self.buffer_threshold):
            if RECON_MODE == 0:
                self.waypoints = C_path
                print("use 'C' path to reconstruct ...")
            elif RECON_MODE == 1:
                self.waypoints = S_path
                print("use 'S' path to reconstruct ...")
            else:
                print("please choose correct path mode ...")
                sys.exit()

            print("Downsample pcds using voxel_size as %d (mm)" % (VOXEL_SIZE))
            self.buffer.append(self.source_pcd)
            print(f"need {self.buffer_threshold} frames to reconstruct the scene, have received {len(self.buffer)} frames")

            print(f"publish truss waypoint {len(self.buffer) - 1} ...")
            self.pub_init_pose(get_truss_T(self.waypoints[len(self.buffer) - 1]), len(self.buffer) - 1)

        elif len(self.buffer) == (self.buffer_threshold):
            if self.out_flag:
                print(f"have received {self.buffer_threshold} frames, begin to get relative poses and initialize constrains")
                start_time = time.time()
                self.relative_poses()
                end_time = time.time()
                print(f"reconstrunction cost time: {end_time - start_time} (s)")
                bool_msg = Bool()
                bool_msg.data = True 
                rospy.loginfo(f"Publishing OPtimization Flag: {bool_msg.data}")
                self.pub_flag.publish(bool_msg)
                print("reconstruction finished, waiting for motion planning...")
            
            if len(self.relocalization_buffer) < len(R_path):  # TODO: bug
                self.relocalization_buffer.append(self.source_pcd)
                print(f"need {len(R_path)} relo frames, have revieved {len(self.relocalization_buffer)} relo frames")
            
            self.out_flag = False

    def pose_callback(self, msg):
        transformation_matrix = np.array(msg.data, dtype = np.float32).reshape(4, 4)
        self.pose_buffer.append(transformation_matrix)
        self.pose_buffer_xyz.append([transformation_matrix[0, 3], transformation_matrix[1, 3], transformation_matrix[2, 3]])
        print(f"pose {len(self.pose_buffer) - 1}: trans: {transformation_matrix[0, 3]}, {transformation_matrix[1, 3]}, {transformation_matrix[2, 3]}")
        print(f"need {self.relative_size} optimized poses, received {len(self.pose_buffer)} optimized poses")

        if len(self.pose_buffer) == (self.relative_size):
            self.get_optimized_cloud()
            points = np.array(self.pose_buffer_xyz)
            self.kdtree = KDTree(points)

            self.relo_flag = True
            bool_msg = Bool()
            bool_msg.data = self.relo_flag 
            self.pub_tmp.publish(bool_msg)
            print("set relo flag True")

    # Add the provided code with the FrameToFrameRegistration function here
    def FrameToFrameRegistration(self, A_pcd, B_pcd, st_id, voxel_size, truss_pos = None):
        print("----Frame to Frame Registration ----")
        start_time = time.time()        

        if VISUALIZE:
            print("downsampled source (blue) and target (red) pcds")
            A_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
            B_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
            o3d.visualization.draw_geometries([A_pcd, B_pcd])

        target_id = int(math.floor(st_id))
        source_id = int(round((float(st_id) - target_id), 3) * 1000)
        delta_x =  self.waypoints[source_id][0] - self.waypoints[target_id][0]
        delta_y =  self.waypoints[source_id][1] - self.waypoints[target_id][1]
        start_time_truss = time.time()
        if TRUSS_USE:
            if st_id != -1:
                target_id = int(math.floor(st_id))
                source_id = int(round((float(st_id) - target_id), 3) * 1000)
                delta_x =  self.waypoints[source_id][0] - self.waypoints[target_id][0]
                delta_y =  self.waypoints[source_id][1] - self.waypoints[target_id][1]
                print(f"delta_x: {delta_x}, delta_y: {delta_y} to {st_id} between source {source_id} and target {target_id}")
                init_transform = np.array([[1, 0, 0, delta_x],
                                        [0, 1, 0, delta_y],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
                A_pcd.transform(init_transform)
            else:
                if RELO_MODE == 0:
                    print("turn to global relo mode...")
                    delta_x = -1
                    delta_y = -1
                    cropped_min = np.array([truss_pos[0] - FOV_Y / 2, truss_pos[1] - FOV_X / 2, -10000.0])
                    cropped_max = np.array([truss_pos[0] + FOV_Y / 2, truss_pos[1] + FOV_X / 2, 10000.0])
                    B_pcd = B_pcd.crop(o3d.geometry.AxisAlignedBoundingBox(cropped_min, cropped_max))
                    init_transform = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
                    A_pcd.transform(init_transform)

                elif RELO_MODE == 1:
                    print("turn to local relo mode...")
                    print(truss_pos, self.waypoints[self.near])
                    delta_x =  truss_pos[0] - self.waypoints[self.near][0]
                    delta_y =  truss_pos[1] - self.waypoints[self.near][1]
                
                    print(f"delta_x: {delta_x}, delta_y: {delta_y} between relo and target {self.near}")
                    init_transform = np.array([[1, 0, 0, delta_x],
                                        [0, 1, 0, delta_y],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
                    A_pcd.transform(init_transform)
        
        end_time_truss = time.time()
        print(f"truss use time: {end_time_truss - start_time_truss}")

        if VISUALIZE:
            print("Truss compenstate, source (blue), target (red) pcds...")
            o3d.visualization.draw_geometries([A_pcd, B_pcd])

        A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
        B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M

        # extract FPFH features       
        A_feats = extract_fpfh(A_pcd, voxel_size)
        B_feats = extract_fpfh(B_pcd, voxel_size)

        # establish correspondences by nearest neighbour search in feature space
        print("Find correspondences...")
        corrs_A, corrs_B = find_correspondences(
                A_feats, B_feats, mutual_filter = True)
        A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
        B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

        start_time_filter = time.time()
        for i in range(A_corr.shape[1] - 1, -1, -1):
            if abs(delta_x) > 0 and abs(A_corr[1, i] - B_corr[1, i]) > FEAT_OFFSET:
                A_corr = np.delete(A_corr, i, axis = 1)
                B_corr = np.delete(B_corr, i, axis = 1)
            elif abs(delta_y) > 0 and abs(A_corr[0, i] - B_corr[0, i]) > FEAT_OFFSET:
                A_corr = np.delete(A_corr, i, axis = 1)
                B_corr = np.delete(B_corr, i, axis = 1)
            else:
                continue
        
        if A_corr.shape[1] != B_corr.shape[1]:
            sys.exit()

        num_corrs = A_corr.shape[1]
        print(f'FPFH generates {num_corrs} positive correspondences.')
        end_time_filter = time.time()
        print(f"filter time: {end_time_filter - start_time_filter}")

        # visualize the point clouds together with feature correspondences
        points = np.concatenate((A_corr.T, B_corr.T), axis = 0)
        lines = []
        for i in range(num_corrs):
            lines.append([i, i + num_corrs])
            colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
            line_set = o3d.geometry.LineSet(
            points = o3d.utility.Vector3dVector(points),
            lines = o3d.utility.Vector2iVector(lines))

        if VISUALIZE:
            print("FPFH Correspondences: source (blue), target (red), match (green)")
            line_set.colors = o3d.utility.Vector3dVector(colors) # painting
            o3d.visualization.draw_geometries([A_pcd, B_pcd, line_set])

        # robust global registration using TEASER++
        print("Excute teaser++ to get initial pose ...")
        NOISE_BOUND = voxel_size
        teaser_solver = get_teaser_solver(NOISE_BOUND)
        teaser_solver.solve(A_corr, B_corr)
        solution = teaser_solver.getSolution()
        R_teaser = solution.rotation
        t_teaser = solution.translation
        T_teaser = Rt2T(R_teaser, t_teaser)
        A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)

        print("Excute ICP to get refined pose ...")
        icp_sol = o3d.pipelines.registration.registration_icp(
            A_pcd, B_pcd, NOISE_BOUND, T_teaser,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 10))
        T_icp = icp_sol.transformation
        A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)

        r_diff, t_diff = compute_transform_difference(T_teaser, T_icp)
        print(f"Difference between teaser and (teaser + icp): rotation: {r_diff}, translation: {t_diff}")

        if VISUALIZE:
            print("Registration results: teaser (blue), teaser_icp (green), target (red)")
            A_pcd_T_teaser.paint_uniform_color([0, 0, 1]) # blue
            A_pcd_T_icp.paint_uniform_color([0, 1, 0]) # green
            o3d.visualization.draw_geometries([A_pcd_T_teaser, A_pcd_T_icp, B_pcd])
    
        end_time = time.time()
        time_use = end_time - start_time
        print(f"The frame to frame registration cost {time_use: .3f} (s)")

        if TRUSS_USE:
            final_transform = np.dot(T_icp, init_transform)
            if truss_pos == None:
                return final_transform
            if truss_pos != None:
                final_transform_relo = np.dot(self.pose_buffer[self.near], final_transform)
                return final_transform_relo
    
        return T_icp
       
    # relocalization        
    def relocalization(self, cloud, truss_pos):
        if self.optimized_pcd_sparse != None:
            if RELO_MODE == 0:
                print("global relo mode")
                T_relo = self.FrameToFrameRegistration(cloud, self.optimized_pcd_sparse, -1, VOXEL_SIZE, truss_pos)
                return T_relo
            else:
                print("local relo mode")
                truss_pos.append(0)
                distances, indices = self.kdtree.query(truss_pos, k = 1)
                self.near = indices
                print(f"nearest cloud is frame {self.near} in buffer")

                T_relo = self.FrameToFrameRegistration(cloud, self.buffer[self.near], -1, VOXEL_SIZE, truss_pos)
                return T_relo
        else:
            print("reconstruction is not finished...")
            self.relo_flag = False
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def point_cloud2_to_open3d(self, msg):
        Recpoints = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans = True)
        Recpoints_array = np.array(list(Recpoints))
        Recpoints_array *= 1000
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(Recpoints_array)
        pcd_down = pcd.voxel_down_sample(voxel_size = VOXEL_SIZE)
        return pcd_down
    
    def get_optimized_cloud(self):
        if len(self.buffer) != self.relative_size:
            print("error occured, because the pose_buffer is not equal to cloud_buffer")
        print("begin to get optimized cloud now...")
        for i in range(0, len(self.buffer)): 
            cloud_tmp = copy.deepcopy(self.buffer[i])
            cloud_i = cloud_tmp.transform(self.pose_buffer[i])
            self.opt_buffer.append(cloud_i)
            self.optimized_pcd_dense += cloud_i
        
        self.optimized_pcd_sparse = self.optimized_pcd_dense.voxel_down_sample(voxel_size = VOXEL_SIZE)

        # publish
        globalpc = o3d.geometry.PointCloud()
        points_np = np.asarray(self.optimized_pcd_sparse.points)
        globalpc.points = o3d.utility.Vector3dVector(points_np)
        o3d.io.write_point_cloud("reconstructionReview_opt.pcd", globalpc)
 
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
    
        fpoints = globalpc.points

        data = []
        for point in fpoints:
            data.append(struct.pack('f', point[0]))
            data.append(struct.pack('f', point[1]))
            data.append(struct.pack('f', point[2]))

        cloud_msg = PointCloud2(
            header = header,
            height = 1,
            width = len(fpoints),
            fields = fields,
            is_bigendian = False,
            point_step = 12,
            row_step = 12 * len(fpoints),
            is_dense = True,
            data=b''.join(data)
        )
        self.opt_pub.publish(cloud_msg)
        rospy.loginfo("Global Optimized PCD Published")
        if VISUALIZE:
            o3d.visualization.draw_geometries([self.optimized_pcd_sparse])

    def pub_reg_pose(self, set_pose, idx):
        p = PoseWithCovarianceStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = str(idx)
        p.pose.pose.position.x = set_pose[0, 3]
        p.pose.pose.position.y = set_pose[1, 3]
        p.pose.pose.position.z = set_pose[2, 3] 
        
        q = tr.quaternion_from_matrix(set_pose)
 
        (p.pose.pose.orientation.x,
        p.pose.pose.orientation.y,
        p.pose.pose.orientation.z,
        p.pose.pose.orientation.w) = q

        p.pose.covariance[6 * 0 + 0] = 0.5 * 0.5
        p.pose.covariance[6 * 1 + 1] = 0.5 * 0.5
        p.pose.covariance[6 * 3 + 3] = math.pi / 12.0 * math.pi / 12.0
 
        self.reg_pub.publish(p) 
        rospy.sleep(0.5)

    def pub_init_pose(self, path_pose, idx):
        p = PoseWithCovarianceStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = str(idx)
        p.pose.pose.position.x = path_pose[0, 3]
        p.pose.pose.position.y = path_pose[1, 3]
        p.pose.pose.position.z = 0 
 
        (p.pose.pose.orientation.x,
        p.pose.pose.orientation.y,
        p.pose.pose.orientation.z,
        p.pose.pose.orientation.w) = (0, 0, 0, 0)

        p.pose.covariance[6 * 0 + 0] = 0.5 * 0.5
        p.pose.covariance[6 * 1 + 1] = 0.5 * 0.52
        p.pose.covariance[6 * 3 + 3] = math.pi / 12.0 * math.pi / 12.0
 
        self.init_pub.publish(p) 
        rospy.sleep(0.5) 

    def pub_relo_pose(self, set_pose, idx):
        p = PoseWithCovarianceStamped()
        p.header.stamp = rospy.Time.now()
        p.header.frame_id = str(idx)
        p.pose.pose.position.x = set_pose[0, 3]
        p.pose.pose.position.y = set_pose[1, 3]
        p.pose.pose.position.z = set_pose[2, 3] 
        
        q = tr.quaternion_from_matrix(set_pose)
 
        (p.pose.pose.orientation.x,
        p.pose.pose.orientation.y,
        p.pose.pose.orientation.z,
        p.pose.pose.orientation.w) = q

        p.pose.covariance[6 * 0 + 0] = 0.5 * 0.5
        p.pose.covariance[6 * 1 + 1] = 0.5 * 0.5
        p.pose.covariance[6 * 3 + 3] = math.pi / 12.0 * math.pi / 12.0
 
        self.relo_pub.publish(p) 
        rospy.sleep(0.5)
    
    def relative_poses(self):
        """ 
            example 14 frames:
           0   1   2   3   4  5  6 
           13  12  11  10  9  8  7
        """
        if len(self.buffer) != FRAME_SIZE:
            print(f"error occured when getting relative poses")
        
        if len(self.buffer) <= FRAME_SIZE:
            self.relative_size = len(self.buffer)
        else:
            self.relative_size = FRAME_SIZE
        print(f"buffer_size: {len(self.buffer)}, reconstruction_size: {FRAME_SIZE}, relative_size: {self.relative_size}")

        for i in range(self.relative_size - 1):
            # adjent constrains
            source_id = i + 1
            target_id = i
            st_id = float(float(float(source_id) / 1000.0) + target_id)
            # print(f"st_id: {st_id}")

            if CROP_USE:
                a_cropped_pcd, b_cropped_pcd, flag = get_roi(self.buffer[target_id], self.waypoints[target_id], 
                                                            self.buffer[source_id], self.waypoints[source_id])
                if not flag:
                    print(f"the FOV between source {source_id} and target {target_id} pcds is too small, so skip ...")
                    continue

                T_icp = self.FrameToFrameRegistration(b_cropped_pcd, a_cropped_pcd, st_id, VOXEL_SIZE, None)
            else:
                T_icp = self.FrameToFrameRegistration(self.buffer[source_id], self.buffer[target_id], st_id, VOXEL_SIZE, None)

            self.constrains_size += 1
            self.relative_pose_buffer.append(T_icp)
            print(f"add and publish this adjent constrains between frame {source_id} and frame {target_id}, now have {self.constrains_size} contrains")
            self.pub_reg_pose(T_icp, st_id)

            # extral constrains
            if (target_id < REVISIT):
                source_id = REVISIT - (target_id % REVISIT) + REVISIT - 1
                # source_id = self.relative_size - 1 - i
                target_id = i
                st_id = float(float(float(source_id) / 1000.0) + target_id)
            else:
                source_id = -1
                target_id = 1

            if source_id > (target_id + 1) and source_id < self.relative_size - 1:
                if CROP_USE: 
                    a_cropped_pcd_, b_cropped_pcd_, flag_ = get_roi(self.buffer[target_id], self.waypoints[target_id], 
                                                                self.buffer[source_id], self.waypoints[source_id])
            
                    if not flag_:
                        print(f"the FOV between source {source_id} and target {target_id} pcds is too small, so skip ...")
                        continue

                    T_icp = self.FrameToFrameRegistration(b_cropped_pcd_, a_cropped_pcd_, st_id, VOXEL_SIZE, None)
                else:
                    T_icp = self.FrameToFrameRegistration(self.buffer[source_id], self.buffer[target_id], st_id, VOXEL_SIZE, None)

                self.pub_reg_pose(T_icp, st_id)
                self.constrains_size += 1
                print(f"add and publish this extral constrains between frame {source_id} and frame {target_id}, now have {self.constrains_size} contrains")
            else:
                print(f"the distance between source {source_id} and target {target_id} pcds is too small, so skip ...")

        # reconstruct global pcd
        """
                    cloud buffer: 0   1   2   3   4   52
            relative pose buffer:   0   1   2   3   4
        """
        print(f"cloud_buffer size: {len(self.buffer)}, relative_pose_buffer size: {len(self.relative_pose_buffer)}")
        if WORLD_FRAME_START: 
            self.before_optimized_pcd_dense += self.buffer[0]
            for i in range(1, self.relative_size):
                cloud_tmp = copy.deepcopy(self.buffer[i])
                for j in range(i - 1, -1, -1): # 3 -> 2: i = 1, j = 0 // i = 2, j = 1, j = 0
                    cloud_trans = cloud_tmp.transform(self.relative_pose_buffer[j])
                self.before_optimized_pcd_dense += cloud_trans
        else:
            self.before_optimized_pcd_dense += self.buffer[self.relative_size - 1] # 4 > 3
            for i in range(self.relative_size - 2, -1, -1):  # 3: i = 1, j = 1 // i = 0, j = 0, j = 1
                cloud_tmp = copy.deepcopy(self.buffer[i])
                for j in range(i, 1, self.relative_size - 1):
                    cloud_trans = cloud_tmp.transform(np.linalg.inv(self.relative_pose_buffer[j]))
                self.before_optimized_pcd_dense += cloud_trans

        self.before_optimized_pcd_sparse = self.before_optimized_pcd_dense.voxel_down_sample(voxel_size = VOXEL_SIZE)

        if VISUALIZE:
            print("Global PCD (blue) before Optimization")
            self.before_optimized_pcd_sparse.paint_uniform_color([0.0, 0.0, 1.0])
            o3d.visualization.draw_geometries([self.before_optimized_pcd_sparse])

        # publish
        globalpc = o3d.geometry.PointCloud()
        points_np = np.asarray(self.before_optimized_pcd_sparse.points)
        globalpc.points = o3d.utility.Vector3dVector(points_np)
        o3d.io.write_point_cloud("reconstructionReview.pcd", globalpc)
 
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
    
        fpoints = globalpc.points

        data = []
        for point in fpoints:
            data.append(struct.pack('f', point[0]))
            data.append(struct.pack('f', point[1]))
            data.append(struct.pack('f', point[2]))

        cloud_msg = PointCloud2(
            header = header,
            height = 1,
            width = len(fpoints),
            fields = fields,
            is_bigendian = False,
            point_step = 12,
            row_step = 12 * len(fpoints),
            is_dense = True,
            data=b''.join(data)
        )
        self.pub.publish(cloud_msg)
        rospy.loginfo("Global PCD Published")

def main():
    rospy.init_node('slam_node')
    slam_node = SLAMNode()

    rospy.spin()
    
if __name__ == '__main__':
    main()
