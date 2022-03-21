#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import String
import time

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
    	Represents competition duration. It is used to turn on/off a
    	timing command which is sent to the scorekeeper
    DEBUG : boolean
		Allows for easier debugging. If True, the car stops moving 
		but continues to trasmit its processed feed.
	startup_flag : boolean
		Used to send start signal to scorekeeper.
	stop_flag : boolean
		Used to send stop signal to scorekeeper.
	startup_time : float
		keeps track of competition duration.
	bridge : CvBridge
		Used to convert cv2 images to/from imgmsg for pub/sub
    """

	COMPETITION_TIME = 5
	DEBUG = False

	def __init__(self):
		"""Sets up all instance variables, mainly pub/sub and timing.

        There is a time.sleep(1) to prevent any messages from being 
        published before being registered with the master node.
        """
		rospy.init_node('camera_interpreter')
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


	
	def linefind(self, data):
		"""Callback function that receives car video feed for processing
		Note that PID is used for driving (I and D yet to be implemented)

        Parameters
        ----------
        data : imgmsg
            the photo (as imgmsg) passed by image_sub
        """

		# If on startup, sends start timer command
		if(self.startup_flag):
			self.license_pub.publish(str('TeamRed,multi21,0,XR58'))
			self.startup_flag = False

		# After elapsed time, sends stop command
		if (time.time() > self.startup_time + self.COMPETITION_TIME) and (self.stop_flag == False):
			self.license_pub.publish(str('TeamRed,multi21,-1,XR58'))
			self.stop_flag = True

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
		"""Processes driving feed to show just the road.

        Parameters
        ----------
        imgmsg : imgmsg
            a photo of the robot's driving feed
        """
        #Convert from imgmsg
		cv_image = self.bridge.imgmsg_to_cv2(imgmsg, desired_encoding='passthrough')
		cv_image_cropped = cv_image[-400:-1]

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

if __name__ == '__main__':
	Robot_Controller()
