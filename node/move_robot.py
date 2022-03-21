#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import String
import time
from robot_states import Robot_State

class Robot_Controller:
	# Set class constants here
	COMPETITION_TIME = 5
	DEBUG = False

	def __init__(self):
		rospy.init_node('camera_interpreter')
		self.drive_state = Robot_State(1)
		self.startup_flag = True
		self.stop_flag = False
		self.bridge = CvBridge()
		self.driving_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
		self.license_pub = rospy.Publisher('/license_plate', String, queue_size=1)
		self.road_pub = rospy.Publisher('/R1/road_image', Image, queue_size=1)
		time.sleep(1)
		self.startup_time = time.time()
		self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.linefind)
		rospy.spin()


	#Callback function receives image and proccesses it into a command
	def linefind(self, data):
		# If on startup, sends start timer command
		if(self.startup_flag):
			self.license_pub.publish(str('TeamRed,multi21,0,XR58'))
			self.startup_flag = False

		# After elapsed time, sends stop command
		if (time.time() > self.startup_time + self.COMPETITION_TIME) and (self.stop_flag == False):
			self.license_pub.publish(str('TeamRed,multi21,-1,XR58'))
			self.stop_flag = True

		#Processes camera feed to just show the road.
		roadPhoto = self.roadIsolation(data)

		#Logic for driving
		contours,hierarchy = cv2.findContours(roadPhoto, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
		try:
			biggestContour=max(contours, key=cv2.contourArea)
		except:
			forward = 0
			turn = 3
			self.sendDriveCommand(forward, turn)
		if cv2.contourArea(biggestContour) < 1000: #why 1000
			forward = 0
			turn = 3
			self.sendDriveCommand(forward, turn)
			return
		moment_array=cv2.moments(biggestContour)
		try:
			cx = int(moment_array['m10']/moment_array['m00'])
			cy = int(moment_array['m01']/moment_array['m00'])
		except ZeroDivisionError:
			print("divide by zero in moments")

		#Determining Driving Command
		forward = 0.25
		maxTurnAmount = 10
		minTurnAmount = -maxTurnAmount
		width = len(roadPhoto[0])
		turn = self.turnRange(cx, 0, width, minTurnAmount, maxTurnAmount)
		self.sendDriveCommand(forward, -turn)
		rospy.loginfo("Forward value: " + str(forward)  + "Turn value: " + str(turn))

		#Drawing diagnostic photo
		self.publishRoadCenter(roadPhoto, (cx, cy))

	def roadIsolation(self, imgmsg):
		cv_image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding='passthrough')
		cv_image_cropped = cv_image[-400:-1]

		imHSV = cv2.cvtColor(cv_image_cropped, cv2.COLOR_RGB2HSV)
		darkerGray = (0,0,60)
		lighterGray = (0,0,150)
		mask = cv2.inRange(imHSV, darkerGray, lighterGray)
		imFiltered = cv2.bitwise_and(imHSV, imHSV, mask=mask)

		#There is no direct HSV2GRAY
		imFiltered_RGB = cv2.cvtColor(imFiltered, cv2.COLOR_HSV2RGB)
		imFiltered_gray = cv2.cvtColor(imFiltered_RGB, cv2.COLOR_RGB2GRAY)

		threshVal = 0
		maxVal = 255
		ret,thresh = cv2.threshold(imFiltered_gray,threshVal,maxVal,cv2.THRESH_BINARY)

		outputImage = thresh
		return outputImage

	#called from lineDrive, publishes to cmd_vel so the car can drive.
	def sendDriveCommand(self, forwardAmount, turnAmount):
		# pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
		move = Twist()
		move.linear.x = forwardAmount
		move.angular.z = turnAmount
		if self.DEBUG:
			move.linear.x = 0
			move.angular.z = 0
		self.driving_pub.publish(move)

	#https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
	def turnRange(self, value, fromMin, fromMax, toMin, toMax):
		fromRange = fromMax - fromMin
		toRange = toMax - toMin
		mappingValue = (value - fromMin) / float(fromRange)
		output = toMin + mappingValue * toRange
		return output 

	#http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
	def publishRoadCenter(self, isolatedRoad, CoM):
		radius = 50
		thickness = -1
		white = (0,0,0)
		black = (255,255,255)
		road_drawn = cv2.circle(isolatedRoad, CoM, radius, black, thickness)
		road_drawn = cv2.circle(road_drawn, CoM, radius // 2, white, thickness)
		image_message = self.bridge.cv2_to_imgmsg(road_drawn, encoding="passthrough")
		self.road_pub.publish(image_message)


if __name__ == '__main__':
	Robot_Controller()