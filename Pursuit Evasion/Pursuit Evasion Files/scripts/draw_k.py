#!/usr/bin/env python

from robot_controller import robot_Controller


def move(controller, x, y, z):
    controller.go_to_coordinate(x, y, z)


if __name__ == '__main__':
    controller = robot_Controller()
    move(controller, -5, 2, 90)
    move(controller, -5, 4, -90)
    move(controller, -5, 0, 90)
    move(controller, -5, 2, 45)
    move(controller, -4, 3, -135)
    move(controller, -5, 2, -45)
    move(controller, -5, 2, -45)
    move(controller, -4, 1, 135)
    robot_Controller.shutdown()


