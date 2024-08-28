#!/usr/bin/env python3
import open3d as o3d
import teaserpp_python
import numpy as np 
import copy
from helpers import *
from scipy.io import savemat
import rospy
from sensor_msgs.msg import PointCloud2,PointField
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist, Point
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import struct
import time
import tf.transformations as tr
# from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist, Point
# from tf import transformations
#from std_msgs.msg import PointCloud2

VOXEL_SIZE = 10
VISUALIZE = True
WORLD_FRAME_START = True

# Add the provided code with the FrameToFrameRegistration function here

def FrameToFrameRegistration(A_pcd, B_pcd, voxel_size):
    start_time = time.time()
    print("-----------------------------")
    A_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
    B_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red

    # voxel downsample both clouds
    # A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    # B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd,B_pcd]) # plot downsampled A and B 

    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(A_pcd,VOXEL_SIZE)
    B_feats = extract_fpfh(B_pcd,VOXEL_SIZE)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

    num_corrs = A_corr.shape[1]
    print(f'FPFH generates {num_corrs} putative correspondences.')

    # visualize the point clouds together with feature correspondences
    points = np.concatenate((A_corr.T,B_corr.T),axis=0)
    lines = []
    for i in range(num_corrs):
        lines.append([i,i+num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])

    # robust global registration using TEASER++
    NOISE_BOUND = VOXEL_SIZE
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr,B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)

    # Visualize the registration results
    A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
    # o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd])

    # local refinement using ICP
    icp_sol = o3d.pipelines.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, T_teaser,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T_icp = icp_sol.transformation
    

    
    # visualize the registration after ICP refinement
    A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)

    end_time = time.time()
    print(f"time use: {end_time - start_time}")
    # o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd])
    return T_icp


    
    

class SLAMNode:
    def __init__(self):
        self.pc_buffer = []
        self.source_pcd = None
        self.target_pcd = None
        self.source_received = False
        self.target_received = False
        self.buffer = []
        self.buffer_threshold = 14

        self.source_sub = rospy.Subscriber('point_cloud_topic', PointCloud2, self.source_callback)
        self.pub = rospy.Publisher('reconstruction_topic', PointCloud2, queue_size=10)
        #self.initialpose_pub = rospy.Publisher('initial_pose_topic', PoseWithCovarianceStamped, queue_size=1)

    def source_callback(self, msg):
        print("enter source callback")
        self.source_pcd = self.point_cloud2_to_open3d(msg)
        self.buffer.append(self.source_pcd)
        if len(self.buffer) >= (self.buffer_threshold):
            self.relative_poses()
            
            self.process_point_clouds()
            self.buffer.clear()
            print("clear the frame buffer, waiting for next round")

    def point_cloud2_to_open3d(self, msg):
        Recpoints = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        #RecpointsMul = [(x * 1000, y * 1000, z * 1000) for x, y, z in Recpoints]
        Recpoints_array = np.array(list(Recpoints))
        Recpoints_array *= 1000
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(Recpoints_array)
        # ts =int(time.time())
        # o3d.io.write_point_cloud("CuboidPcd" + str(ts) + ".pcd", pcd)
        pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        return pcd_down
    
    def pub_initial_position(self, set_pose, idx):
        rospy.loginfo("start test inital pose...")
        setpose_pub = rospy.Publisher("initialpose",PoseWithCovarianceStamped,latch=True, queue_size=10)
        rospy.loginfo("start set pose...")
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

        # p.pose.covariance[6 * 0 + 0] = 0.5 * 0.5
        # p.pose.covariance[6 * 1 + 1] = 0.5 * 0.5
        # p.pose.covariance[6 * 3 + 3] = math.pi / 12.0 * math.pi / 12.0
 
        setpose_pub.publish(p) 
        rospy.sleep(1)

    
    def relative_poses(self):
        """
           0   1   2   3   4  5  6 
           13  12  11  10  9  8  7
        """
        for i in range(3):
            if i<13:  # for the first 13 scans
                source_id = i
                target_id = i+1
                T_icp = FrameToFrameRegistration(self.buffer[source_id],self.buffer[target_id],VOXEL_SIZE)
                #print(T_icp)
                # publish initial camera pose
                self.pub_initial_position(np.linalg.inv(T_icp), i)
            elif i == 14:
                source_id = 5
                target_id = 8
                T_icp = FrameToFrameRegistration(self.buffer[source_id],self.buffer[target_id],VOXEL_SIZE)
                #print(T_icp)
                # publish initial camera pose
                self.pub_initial_position(np.linalg.inv(T_icp), i)
        
        # for i in range(17):
        #     if i<14:  # for the first 13 scans
        #         source_id = i
        #         target_id = i+1
        #         T_icp = FrameToFrameRegistration(self.buffer[source_id],self.buffer[target_id],VOXEL_SIZE)
        #         #print(T_icp)
        #         # publish initial camera pose
        #         self.pub_initial_position(np.linalg.inv(T_icp), i)
        #     elif i == 14:
        #         source_id = 5
        #         target_id = 8
        #         T_icp = FrameToFrameRegistration(self.buffer[source_id],self.buffer[target_id],VOXEL_SIZE)
        #         #print(T_icp)
        #         # publish initial camera pose
        #         self.pub_initial_position(np.linalg.inv(T_icp), i)
        #     elif i == 15:
        #         source_id = 2
        #         target_id = 11
        #         T_icp = FrameToFrameRegistration(self.buffer[source_id],self.buffer[target_id],VOXEL_SIZE)
        #         #print(T_icp)
        #         # publish initial camera pose
        #         self.pub_initial_position(np.linalg.inv(T_icp), i) 
        #     elif i == 16:
        #         source_id = 1
        #         target_id = 12
        #         T_icp = FrameToFrameRegistration(self.buffer[source_id],self.buffer[target_id],VOXEL_SIZE)
        #         #print(T_icp)
        #         # publish initial camera pose
        #         self.pub_initial_position(np.linalg.inv(T_icp), i)         


    def process_point_clouds(self):
        print("enter process_point_cloud function")
        refined_poses = []
        global_point_cloud = o3d.geometry.PointCloud()
        for i in range(len(self.buffer)-1):
            source_id = i
            target_id = i+1
            T_icp = FrameToFrameRegistration(self.buffer[source_id],self.buffer[target_id],VOXEL_SIZE)
            #print(T_icp)
            # publish initial camera pose
            #self.pub_initial_position(np.linalg.inv(T_icp), i)
            refined_poses.append(T_icp)
        if WORLD_FRAME_START:
            for i in range(1,self.buffer_threshold):
                for j in range(i-1,-1,-1):
                    self.buffer[i].transform(np.linalg.inv(refined_poses[j]))
                global_point_cloud += self.buffer[i]
            global_point_cloud += self.buffer[0]
        else:
            for i in range(self.buffer_threshold-1):
                for j in range(i,self.buffer_threshold-1):
                    self.buffer[i].transform(refined_poses[j])
                global_point_cloud += self.buffer[i]
            global_point_cloud += self.buffer[len(self.buffer)-1]

        # obtain init pose

        # publish 

        
        #o3d.visualization.draw_geometries([global_point_cloud])    
        globalpc = o3d.geometry.PointCloud()
        points_np = np.asarray(global_point_cloud.points)
        offset_Pcr = np.array([375, 602, -183.7])
        points_np -= offset_Pcr
        # translation = np.array([4000,0,3196])
        # points_np[:,0] -=translation[0] #x = x - 4000
        # points_np[:,1] *=- 1  #y = -y
        # points_np[:,2] =translation[2] - points_np[:,2] #z = 3196 - z
        globalpc.points = o3d.utility.Vector3dVector(points_np)
        o3d.io.write_point_cloud("reconstructionReview.pcd",globalpc)
 
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
            header=header,
            height=1,
            width=len(fpoints),
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * len(fpoints),
            is_dense=True,
            data=b''.join(data)
    )
        self.pub.publish(cloud_msg)
        print("published")
        rospy.loginfo("global pcd sent")
        o3d.visualization.draw_geometries([global_point_cloud])
        
def main():
    rospy.init_node('slam_node')
    slam_node = SLAMNode()
    rospy.spin()
    
if __name__ == '__main__':
    main()
