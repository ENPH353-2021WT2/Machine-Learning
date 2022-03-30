#! /usr/bin/env python

import enum

class Robot_State(enum.Enum):
	DRIVE_FORWARD = 1
	LEFT_TURN = 2
	RIGHT_TURN = 3
	PEDESTRIAN = 4