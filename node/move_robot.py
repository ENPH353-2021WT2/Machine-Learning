#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import String
import time
from robot_states import Robot_State
import numpy as np
from matplotlib import pyplot as plt

class Robot_Controller:
	"""
    A class used to control a self-driving car in ROS/Gazebo.
    Application entry point.

    ...

    Attributes
    ----------
	image_sub : Subscriber
		Robot camera feed. This is the only piece of feedback used to
		control the car
	driving_pub : Publisher
		Publishes speed/turn commands to robot
	license_pub : Publisher
		Publishes license plate numbers and start/stop commands
	road_pub : Publisher
		Publishes diagnostic photo of road isolated from environment

    COMPETITION_TIME : int
    	Represents competition duration in seconds. It is used to turn on/off a
    	timing command which is sent to the scorekeeper
    DEBUG : boolean
		Allows for easier debugging. If True, the car stops moving 
		but continues to trasmit its processed feed.
	LEFT_TURN_TIME : int
		Time needed to complete a left turn in seconds
	RED_THRESHOLD : int
		Experimentally determined for deciding when to stop robot for crosswalk
	PED_WHITE_THRESHOLD : int
		Experimentally determined for when pedestrian movement is detected

	startup_flag : boolean
		Used to send start signal to scorekeeper.
	stop_flag : boolean
		Used to send stop signal to scorekeeper.
	left_turn_flag : boolean
		Used to turn car left without changing state
	ignore_red_flag : boolean
		Used to bypass other red line after having stopped for pedestrian
	startup_time : float
		keeps track of competition duration.
	pedestrian_start_time : float
		Keeps track of time after having passed pedestrian crosswalk
	bridge : CvBridge
		Used to convert cv2 images to/from imgmsg for pub/sub
    """

	COMPETITION_TIME = 5
	DEBUG = False
	LEFT_TURN_TIME = 1.5
	RED_THRESHOLD = 5000
	PED_WHITE_THRESHOLD = 1000

	def __init__(self):
		"""Sets up all instance variables, mainly pub/sub and timing.

        There is a time.sleep(1) to prevent any messages from being 
        published before being registered with the master node.
        """
		rospy.init_node('camera_interpreter')
		self.drive_state = Robot_State.DRIVE_FORWARD
		self.startup_flag = True
		self.stop_flag = False
		self.left_turn_flag = True
		self.ignore_red_flag = False
		self.first_run = True
		# Size of raw image (720, 1280, 3)
		self.prev_frame = np.zeros((720, 1280, 3))
		self.bridge = CvBridge()
		self.driving_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
		self.license_pub = rospy.Publisher('/license_plate', String, queue_size=1)
		self.road_pub = rospy.Publisher('/R1/road_image', Image, queue_size=1)
		self.crosswalk_pub = rospy.Publisher('/R1/crosswalk_image', Image, queue_size=1)
		time.sleep(1)
		self.startup_time = time.time()
		self.pedestrian_start_time = 0
		self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.linefind)
		rospy.spin()


	
	def linefind(self, data):
		"""Callback function that receives car video feed for processing
		Note that PID is used for driving (I and D yet to be implemented)

        Parameters
        ----------
        data : imgmsg
            the photo (as imgmsg) passed by image_sub
        """
		# print(self.drive_state)
		# If on startup, sends start timer command
		if(self.startup_flag):
			self.license_pub.publish(str('TeamRed,multi21,0,XR58'))
			self.startup_flag = False

		# After elapsed time, sends stop command
		if (time.time() > self.startup_time + self.COMPETITION_TIME) and (self.stop_flag == False):
			self.license_pub.publish(str('TeamRed,multi21,-1,XR58'))
			self.stop_flag = True

		
		### STATE MACHINE BELOW ###

		# Pedestrian State
		if self.drive_state == Robot_State.PEDESTRIAN:
			forward = 0
			turn = 0
			self.sendDriveCommand(forward, turn)
			
			current_frame = self.getPedestrianFrame(data)
			# If first loop, use same image
			if self.first_run:
				self.prev_frame = current_frame
				self.first_run = False

			# print(current_frame.shape)
			
			# cv2.imshow("frame", self.prev_frame)
			# cv2.waitKey(3)
			gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
			gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)


			# Compute difference image
			difference_img = cv2.absdiff(gray_current, gray_prev)
			ret, thresh = cv2.threshold(difference_img, 30, 255, cv2.THRESH_BINARY)
			image_message = self.bridge.cv2_to_imgmsg(thresh, encoding="passthrough")
			self.crosswalk_pub.publish(image_message)

			# Detects pedestrian movement in front of robot
			if time.time() > self.pedestrian_start_time + 1.5:
				Y, X = np.where(thresh==255)
				print("Number of white points: ", len(X))

				if len(X) >= self.PED_WHITE_THRESHOLD:
					self.drive_state = Robot_State.DRIVE_FORWARD
					self.ignore_red_flag = True
					self.pedestrian_start_time = time.time()
					print("Pedestrian detected!!!")

			# Waits for 6 seconds and then goes in event that pedestrian is stuck
			if time.time() > self.pedestrian_start_time + 6:
				self.drive_state = Robot_State.DRIVE_FORWARD
				self.ignore_red_flag = True
				self.pedestrian_start_time = time.time()
				print("Time's up!")

			# Updates previous frame
			self.prev_frame = current_frame

		
		# Driving State
		elif self.drive_state == Robot_State.DRIVE_FORWARD:


			# Want robot to turn left on startup, for a specified amount of time
			if time.time() > self.startup_time + self.LEFT_TURN_TIME:
				self.left_turn_flag = False

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
			maxTurnAmount = 5
			minTurnAmount = -maxTurnAmount
			width = len(roadPhoto[0])
			turn = self.turnRange(cx, 0, width, minTurnAmount, maxTurnAmount)
			forward = 0.4
			self.sendDriveCommand(forward, -turn)
			# rospy.loginfo("Forward value: " + str(forward)  + "Turn value: " + str(turn))

			#Drawing diagnostic photo
			self.publishRoadCenter(roadPhoto, (cx, cy))

			#Detects crosswalk red line
			redPoints = self.detectCrosswalk(data)

			# Stops ignoring red points after 4 seconds of passing pedestrian
			if time.time() > self.pedestrian_start_time + 4:
				self.ignore_red_flag = False

			# If detects crosswalk, changes state
			if redPoints >= self.RED_THRESHOLD and not self.ignore_red_flag:
				self.drive_state = Robot_State.PEDESTRIAN
				self.pedestrian_start_time = time.time()
				self.prev_frame = self.getPedestrianFrame()



	def roadIsolation(self, imgmsg):
		"""Processes driving feed to show just the road.

        Parameters
        ----------
        imgmsg : imgmsg
            a photo of the robot's driving feed
        """
        #Convert from imgmsg
		cv_image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding='passthrough')
		width = len(cv_image[0])

		# Depending on left turn or not, will crop the image accordingly
		if not self.left_turn_flag:
			cv_image_cropped = cv_image[-400:-200,500:width-500]
		else:
			cv_image_cropped = cv_image[-400:-1,600:width]

		#convert photo to HSV and isolate for road with mask
		imHSV = cv2.cvtColor(cv_image_cropped, cv2.COLOR_RGB2HSV)
		darkerGray = (0,0,60)
		lighterGray = (0,0,150)
		mask = cv2.inRange(imHSV, darkerGray, lighterGray)
		imFiltered = cv2.bitwise_and(imHSV, imHSV, mask=mask)

		#convert to grayscale in 2 steps
		imFiltered_RGB = cv2.cvtColor(imFiltered, cv2.COLOR_HSV2RGB)
		imFiltered_gray = cv2.cvtColor(imFiltered_RGB, cv2.COLOR_RGB2GRAY)

		#change to pure black/white photo for maximum contrast
		threshVal = 0
		maxVal = 255
		ret,thresh = cv2.threshold(imFiltered_gray,threshVal,maxVal,cv2.THRESH_BINARY)

		outputImage = thresh
		return outputImage

	def sendDriveCommand(self, forwardAmount, turnAmount):
		"""Performs the actual publishing of driving commands to R1/cmd_vel

        Parameters
        ----------
        forwardAmount : float
            the linear speed in m/s of the vehicle
		turnAmont : float
			the angular velocity in rad/s of the vehicle
        """
		move = Twist()
		move.linear.x = forwardAmount
		move.angular.z = turnAmount
		if self.DEBUG:
			move.linear.x = 0
			move.angular.z = 0
		self.driving_pub.publish(move)

	def turnRange(self, value, fromMin, fromMax, toMin, toMax):
		"""Maps a value from a given range to a new range.
		Source: https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another

        Parameters
        ----------
        value : float
        	the initial value living in the range fromMax - fromMin
        fromMin : float
        	the lower bound of the initial range
        fromMax : float
        	the upper bound of the initial range
        toMin : float
        	the lower bound of the output range
        toMax : float
        	the upper bound of the output range
        """
		fromRange = fromMax - fromMin
		toRange = toMax - toMin

		#Value is scaled to range [0,1]
		mappingProportion = (value - fromMin) / float(fromRange)

		#Value is mapped to new range [toMin, toMax]
		output = toMin + mappingProportion * toRange
		return output 

	def publishRoadCenter(self, isolatedRoad, CoM):
		"""Publishes a diagnostic photo of the road + center of mass to
		/R1/road_image

        Parameters
        ----------
        isolatedRoad : Image
        	Driving feed processed to just a road in black-and-white
        CoM : (int, int)
        	(x, y) coodinate of center of mass of the photo. A circle is
        	drawn at this location and is what the car is aiming at
        """
		radius = 50
		thickness = -1
		white = (0,0,0)
		black = (255,255,255)
		road_drawn = cv2.circle(isolatedRoad, CoM, radius, black, thickness)
		road_drawn = cv2.circle(road_drawn, CoM, radius // 2, white, thickness)
		image_message = self.bridge.cv2_to_imgmsg(road_drawn, encoding="passthrough")
		self.road_pub.publish(image_message)

	def detectCrosswalk(self, imgmsg):
		"""Processes driving feed to detect red crosswalk line

        Parameters
        ----------
        imgmsg : imgmsg
            a photo of the robot's driving feed
        """
        #Convert from imgmsg
		cv_image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding='passthrough')
		width = len(cv_image[0])

		# Crop image for optimal crosswalk detection
		cv_image_cropped = cv_image[-400:-1]

		#convert photo to HSV and isolate for road with mask
		imHSV = cv2.cvtColor(cv_image_cropped, cv2.COLOR_RGB2HSV)
		lowerBound = (0,100,20)
		upperBound = (10,255,255)
		mask = cv2.inRange(imHSV, lowerBound, upperBound)

		# binary mask (convert to black and white)
		val, thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
		Y, X = np.where(thresh==255)

		# print('Number of Red Points: ' + str(len(X)))

		#publish image message
		image_message = self.bridge.cv2_to_imgmsg(thresh, encoding="passthrough")
		# self.crosswalk_pub.publish(image_message)
		return len(X)

	def getPedestrianFrame(self, imgmsg):
		cv_image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding='passthrough')
		height = len(cv_image)
		width = len(cv_image[0])
		cv_image = cv_image[200:height-200,300:width-300]
		return cv_image

if __name__ == '__main__':
	Robot_Controller()
