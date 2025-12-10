#!/usr/bin/env python3
from oculus_reader import OculusReader
from tf.transformations import quaternion_from_matrix
import rospy
import tf2_ros
import geometry_msgs.msg
import numpy as np


class OculusPublisher:
    def __init__(self):
        rospy.init_node('pub_pose_node')
        self.right_handle_pose_pub = rospy.Publisher('right_handle_pose', geometry_msgs.msg.PoseStamped, queue_size=1)
        self.left_handle_pose_pub = rospy.Publisher('left_handle_pose', geometry_msgs.msg.PoseStamped, queue_size=1)
        self.br = tf2_ros.TransformBroadcaster()
        self.oculus_reader = OculusReader()
        # 全局频率为100Hz
        self.rate = rospy.Rate(100)
        
    def xyzrpy2Mat(self,x, y, z, roll, pitch, yaw):
        transformation_matrix = np.eye(4)
        A = np.cos(yaw)
        B = np.sin(yaw)
        C = np.cos(pitch)
        D = np.sin(pitch)
        E = np.cos(roll)
        F = np.sin(roll)
        DE = D * E
        DF = D * F
        transformation_matrix[0, 0] = A * C
        transformation_matrix[0, 1] = A * DF - B * E
        transformation_matrix[0, 2] = B * F + A * DE
        transformation_matrix[0, 3] = x
        transformation_matrix[1, 0] = B * C
        transformation_matrix[1, 1] = A * E + B * DF
        transformation_matrix[1, 2] = B * DE - A * F
        transformation_matrix[1, 3] = y
        transformation_matrix[2, 0] = -D
        transformation_matrix[2, 1] = C * F
        transformation_matrix[2, 2] = C * E
        transformation_matrix[2, 3] = z
        transformation_matrix[3, 0] = 0
        transformation_matrix[3, 1] = 0
        transformation_matrix[3, 2] = 0
        transformation_matrix[3, 3] = 1
        return transformation_matrix
    
    def adjustment_matrix(self,transform):

        if transform.shape != (4, 4):
            raise ValueError("Input transform must be a 4x4 numpy array.")
        
        adj_mat = np.array([
            [0,0,-1,0],
            [-1,0,0,0],
            [0,1,0,0],
            [0,0,0,1]
        ])
        
        r_adj = self.xyzrpy2Mat(0,0,0,   -np.pi , 0, -np.pi/2)

        transform = adj_mat @ transform  
        
        transform = np.dot(transform, r_adj)  
        
        return transform
    def right_handle_publish_transform(self, transform, name):
        translation = transform[:3, 3]

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'vr_device'
        t.child_frame_id = name
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        quat = quaternion_from_matrix(transform)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)

        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'vr_device'
        pose_msg.pose.position.x = translation[0]
        pose_msg.pose.position.y = translation[1]
        pose_msg.pose.position.z = translation[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.right_handle_pose_pub.publish(pose_msg)

    def left_handle_publish_transform(self, transform, name):
        translation = transform[:3, 3]

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = 'vr_device'
        t.child_frame_id = name
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        quat = quaternion_from_matrix(transform)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)

        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'vr_device'
        pose_msg.pose.position.x = translation[0]
        pose_msg.pose.position.y = translation[1]
        pose_msg.pose.position.z = translation[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.left_handle_pose_pub.publish(pose_msg)
        
    def run(self):
        try:
            while not rospy.is_shutdown():
                transformations, _ = self.oculus_reader.get_transformations_and_buttons()
                if 'r' not in transformations or 'l' not in transformations:
                    self.rate.sleep()
                    continue
                
                right_controller_pose = self.adjustment_matrix(transformations['r'])
                left_controller_pose = self.adjustment_matrix(transformations['l'])
                
                self.right_handle_publish_transform(right_controller_pose, 'right_controller')
                self.left_handle_publish_transform(left_controller_pose, 'left_controller')
                self.rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            self.oculus_reader.stop()
            


if __name__ == '__main__':
    publisher = OculusPublisher()
    try:
        publisher.run()
    except rospy.ROSInterruptException:
        pass