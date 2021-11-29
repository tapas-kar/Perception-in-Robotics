#!/usr/bin/env python

"""
Code adopted from the turtlebot3 example node for turtlebot3_pointop_key. (https://github.com/ROBOTIS-GIT/turtlebot3/blob/master/turtlebot3_example/nodes/turtlebot3_pointop_key)
"""

from math import radians, copysign, sqrt, pow, pi, atan2
import numpy as np

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion

import tf
from tf.transformations import euler_from_quaternion



class robot_Controller():
    def __init__(self):
        rospy.init_node('turtlebot3_controller', anonymous=False)
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        self.rate = rospy.Rate(10)
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'

        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")
    
    
    def go_to_coordinate(self, target_x, target_y, target_z=None, linear_speed= 1):
        (position, rotation) = self.get_odom()
        print(position, rotation)
        last_rotation = 0
        angular_speed = 1
        if target_z > 180 or target_z < -180:
            print("Z (angle) should be between -180 to 180")
            return
        target_z = np.deg2rad(target_z)
        goal_distance = sqrt(pow(target_x - position.x, 2) + pow(target_y - position.y, 2))
        distance = goal_distance

        position = Point()
        move_cmd = Twist()
        
        # move to the new (target_x, target_y) location
        while distance > 0.05:
            (position, rotation) = self.get_odom()
            x_start = position.x
            y_start = position.y
            path_angle = atan2(target_y - y_start, target_x- x_start)

            if path_angle < -pi/4 or path_angle > pi/4:
                if target_y < 0 and y_start < target_y:
                    path_angle = -2*pi + path_angle
                elif target_y >= 0 and y_start > target_y:
                    path_angle = 2*pi + path_angle
            if last_rotation > pi-0.1 and rotation <= 0:
                rotation = 2*pi + rotation
            elif last_rotation < -pi+0.1 and rotation > 0:
                rotation = -2*pi + rotation
            move_cmd.angular.z = angular_speed * path_angle-rotation

            distance = sqrt(pow((target_x - x_start), 2) + pow((target_y - y_start), 2))

            # move faster if we are farther away from the goal
            if(distance < 0.3):
                move_cmd.linear.x = 0.2
            else:
                move_cmd.linear.x = 0.4

            if move_cmd.angular.z > 0:
                move_cmd.angular.z = min(move_cmd.angular.z, 1.5)   
            else:
                move_cmd.angular.z = max(move_cmd.angular.z, -1.5)

            last_rotation = rotation
            self.cmd_vel.publish(move_cmd)
            self.rate.sleep()
        (position, rotation) = self.get_odom()

        # rotate to the "target_z" degrees IF requested
        if(True):
            while abs(rotation - target_z) > 0.05:
                (position, rotation) = self.get_odom()
                if target_z >= 0:
                    if rotation <= target_z and rotation >= target_z - pi:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = 0.5
                    else:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = -0.5
                else:
                    if rotation <= target_z + pi and rotation > target_z:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = -0.5
                    else:
                        move_cmd.linear.x = 0.00
                        move_cmd.angular.z = 0.5
                self.cmd_vel.publish(move_cmd)
                self.rate.sleep()

        rospy.loginfo("Reached co-ordinate [{}, {}, {}]".format(target_x, target_y, target_z))
        self.cmd_vel.publish(Twist())
        self.rate.sleep()


    def get_odom(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return
        print(Point(*trans), rotation[2])
        return (Point(*trans), rotation[2])


    def shutdown(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
