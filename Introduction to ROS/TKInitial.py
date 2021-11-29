#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist

class Turtle:
    def __init__(self):
        rospy.init_node('my_initials', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.PI = 3.1415926535897

    def move_straight(self, speed, distance):
        vel_msg = Twist()

        # Checking if the movement is forward or backwards
        if distance > 0:
            vel_msg.linear.x = abs(speed)
        else:
            vel_msg.linear.x = -abs(speed)

        distance = abs(distance)

        # Since we are moving just in x-axis
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0

        while not rospy.is_shutdown():

            # Setting the current time for distance calculus
            t0 = rospy.Time.now().to_sec()
            current_distance = 0

            # Loop to move the turtle to a specified distance
            while current_distance < distance:

                # Publish the velocity
                self.velocity_publisher.publish(vel_msg)

                # Takes actual time for velocity calculus
                t1 = rospy.Time.now().to_sec()

                # Calculates distance Pose Stamped
                current_distance = speed * (t1 - t0)

            # After the loop, stops the robot
            vel_msg.linear.x = 0

            # Force the robot to stop
            self.velocity_publisher.publish(vel_msg)

            return

    def rotate(self, speed, angle, clockwise):
        vel_msg = Twist()

        # Converting the angles to radians
        angular_speed = speed * 2 * self.PI / 360
        relative_angle = speed * 2 * self.PI / 360

        # We will not use linear components
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0

        # Checking if the movement is Clockwise or Counter-Clockwise
        if clockwise:
            vel_msg.angular.z = -abs(angular_speed)
        else:
            vel_msg.angular.z = abs(angular_speed)

        # Setting the current time for distance calculus
        t0 = rospy.Time.now().to_sec()
        current_angle = 0

        while current_angle < relative_angle:
            self.velocity_publisher.publish(vel_msg)
            t1 = rospy.Time.now().to_sec()
            current_angle = angular_speed * (t1 - t0)

        # Forcing our robot to stop
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
        return

    if __name__ == "__main__":
        turtle = Turtle()
        turtle.rotate(30, 100, False)
        turtle.move_linear(1, 3)
        turtle.move_linear(1, -6)
        turtle.move_linear(1, 3)
        turtle.rotate(30, 45, True)
        turtle.move_linear(1, 3)
        turtle.move_linear(1, -3)
        turtle.rotate(30, 100, True)
        turtle.move_linear(1, 3)
        turtle.move_linear(1, -3)