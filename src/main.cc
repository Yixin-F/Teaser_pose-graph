/*
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2024-05-08 14:55:32
 * @LastEditTime: 2024-06-07 16:30:47
 * @LastEditors: Yixin 917449595@qq.com
 * @Description:
 * @FilePath: /baseline/examples/test_pose_graph.cc
 */

#include "optimizer/PoseGraphOptimization.hpp"
#include "optimizer/factor/PoseGraphSE3Factor.hpp"
#include "optimizer/factor/PoseGraphSO3Factor.hpp"
#include "utils/IOFuntion.hpp"
#include "utils/UtilTransformer.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include "ros/ros.h"
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "std_msgs/Float32MultiArray.h"
#include <std_msgs/Bool.h>

using namespace simulator;

void eigen_2_pose(std::vector<Eigen::Matrix4d> &vec_pose, IO::MapOfPoses &poses);
void eigen_2_constraints(Eigen::Matrix4d &loop_pose, IO::VectorOfConstraints &constraints);
void init_vertex(std::vector<Eigen::Matrix4d> &vec_pose);
void pose_constraints(std::map<float, Eigen::Vector3d> id_trans, std::map<float, Eigen::Quaterniond> id_quat, IO::VectorOfConstraints &constraints);

std::map<int, Eigen::Vector3d> id_trans;
std::map<int, Eigen::Quaterniond> id_quat;

std::map<float, Eigen::Vector3d> id_trans_constrains;
std::map<float, Eigen::Quaterniond> id_quat_constrains;

bool do_opt = false;
bool do_init = false;

void boolCallback(const std_msgs::Bool::ConstPtr& msg) {
    do_opt = msg->data;
    ROS_INFO("recieved optimization flag: [%s]", msg->data ? "true" : "false");
}

void GetInitPose(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& pose) {
    std::cout << "get one single pose " << pose->header.frame_id << std::endl;
    geometry_msgs::PoseWithCovarianceStamped ob_pose;
    std::string frame_id = pose->header.frame_id;
    ob_pose.header = pose->header;
    ob_pose.pose = pose->pose;
    
    std::cout << frame_id << ": "
              << ob_pose.pose.pose.position.x << ", "
              << ob_pose.pose.pose.position.y << ", "
              << ob_pose.pose.pose.position.z << std::endl;
    
    Eigen::Vector3d translation;
    Eigen::Quaterniond quat;

    translation << ob_pose.pose.pose.position.x, ob_pose.pose.pose.position.y, ob_pose.pose.pose.position.z;
    quat.x() = ob_pose.pose.pose.orientation.x;
    quat.y() = ob_pose.pose.pose.orientation.y;
    quat.z() = ob_pose.pose.pose.orientation.z;
    quat.w() = ob_pose.pose.pose.orientation.w;

    float idx = std::stof(frame_id);
    if ((idx - std::floor(idx)) == 0.0) {
        id_trans.insert(std::make_pair(stoi(frame_id), translation));
        id_quat.insert(std::make_pair(stoi(frame_id), quat));
        do_init = true;
        std::cout<< "get truss way pose " << "id: " << stoi(frame_id) << " and now have " << id_trans.size() << " poses" << std::endl;
    }
    else {
        id_trans_constrains.insert(std::make_pair(idx, translation));
        id_quat_constrains.insert(std::make_pair(idx, quat));
        std::cout<< "get constrain pose " << "id: " << idx << " and now have " << id_trans_constrains.size() << " poses" << std::endl;
    }
}

int main(int argc, char *argv[]) {   
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "camera_pose_optimizer");
    std::cout << "-------- optimizer start --------------" << std::endl;

    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("init_pose", 10, GetInitPose);
    ros::Subscriber reg_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("reg_pose", 10, GetInitPose);
    ros::Publisher pub = nh.advertise<std_msgs::Float32MultiArray>("opt_pose", 10);
    ros::Subscriber sub_flag = nh.subscribe("bool_topic", 10, boolCallback);

    std::vector<Eigen::Matrix4d> vec_truss_pose;

    int pub_num = 0;

    // wait for optimization
    while (ros::ok()) {
        ros::spinOnce();

        if (do_init) {
            init_vertex(vec_truss_pose);
            do_init = false;
        }

        if (do_opt) {
            simulator::IO::VectorOfConstraints constraints;
            pose_constraints(id_trans_constrains, id_quat_constrains, constraints);

            simulator::IO::MapOfPoses poses;
            eigen_2_pose(vec_truss_pose, poses);

            // do optimizer
            std::vector<Eigen::Matrix4d> Twcs_posegraph;
            simulator::optimizer::PoseGraphOptimization::optimizer(poses, constraints, Twcs_posegraph);
            std::cout << "this optimization process is finished ..." << std::endl;

            // publish
            for (auto pose : Twcs_posegraph) {
                std_msgs::Float32MultiArray optimized_pose_i;
                for (size_t j = 0; j < 4; j++) {
                    for (size_t k = 0; k < 4; k++) {
                        optimized_pose_i.data.emplace_back(pose(j,k));
                    }
                }
                std::cout << "publish optimized pose " << pub_num++ << std::endl;
                ros::Duration(0.2).sleep();
                pub.publish(optimized_pose_i);
            }

            do_opt = false;  // reset and wait for new constrains
        }
    }
    
    return 0;
}

void init_vertex(std::vector<Eigen::Matrix4d> &vec_pose){
    // TODO: get initial poses from joint states continously
    std::cout << "need to initialize " << id_trans.size() << " vertexs" << std::endl;
    vec_pose.clear();
    for (auto &vertex : id_trans) {
        Eigen::Matrix4d pose_tmp;
        pose_tmp(0, 3) = vertex.second(0);
        pose_tmp(1, 3) = vertex.second(1);
        pose_tmp(2, 3) = vertex.second(2);
        vec_pose.emplace_back(pose_tmp);
    }
}

void pose_constraints(std::map<float, Eigen::Vector3d> id_trans, std::map<float, Eigen::Quaterniond> id_quat, 
                      IO::VectorOfConstraints &constraints) {
    std::cout << "need to initialize " << id_trans_constrains.size() << " edges" << std::endl;
    for (auto &edge: id_trans_constrains) {
        float id = edge.first;
        Eigen::Vector3d trans = edge.second;
        Eigen::Quaterniond quat = id_quat_constrains[edge.first];

        IO::Constraint3d constraint_id;

        int target_id = (int)(std::floor(id));
        int source_id = (int)(std::round((id - target_id) * 1000.0));

        std::cout << "add edge between frame " << source_id << " and frame " << target_id << ", trans: " <<  trans.transpose() << std::endl;
        // std::cout << (id - target_id) << " " << (id - target_id) * 1000.0 << " " << (int)((id - target_id) * 1000.0) << std::endl;

        constraint_id.id_begin = target_id;  // TODO: check
        constraint_id.id_end = source_id;

        // constraint_id.id_begin = source_id;  // TODO: check
        // constraint_id.id_end = target_id;

        constraint_id.t_be.p.x() = trans(0);
        constraint_id.t_be.p.y() = trans(1);
        constraint_id.t_be.p.z() = trans(2);
        constraint_id.t_be.q.x() = quat.x();
        constraint_id.t_be.q.y() = quat.y();
        constraint_id.t_be.q.z() = quat.z();
        constraint_id.t_be.q.w() = quat.w();
        
        constraints.push_back(constraint_id);
    }
}

void eigen_2_pose(std::vector<Eigen::Matrix4d> &vec_pose, IO::MapOfPoses &poses) {
    std::cout << "transfer " << vec_pose.size() << " poses to vertexs..." << std::endl; 
    for (int i = 0; i < vec_pose.size(); i++) {
        // rotation
        Eigen::Matrix3d R = vec_pose[i].block(0, 0, 3, 3);
        Eigen::Quaterniond quat(R);
        Eigen::Vector3d trans = vec_pose[i].block(0, 3, 3, 1);
        // pose3d
        IO::Pose3d pose;
        pose.p.x() = trans(0);
        pose.p.y() = trans(1);
        pose.p.z() = trans(2);
        pose.q.x() = quat.x();
        pose.q.y() = quat.y();
        pose.q.z() = quat.z();
        pose.q.w() = quat.w();
        pose.q.normalize();
        poses[i] = pose;
    }
}
