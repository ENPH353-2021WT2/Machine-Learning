#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def setup():
	rospy.init_node('camera_interpreter')
	rospy.Subscriber('/R1/pi_camera/image_raw', Image, linefind)
	rospy.spin()

#Callback function receives image and proccesses it into a command
def linefind(data):
	bridge = CvBridge()
	cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

	#convert input image to grayscale
	imHSV = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
	darkerGray = (0,0,60)
	lighterGray = (0,0,150)

	mask = cv2.inRange(imHSV, darkerGray, lighterGray)

	imFiltered = cv2.bitwise_and(imHSV, imHSV, mask=mask)
	#crop relevant area

	imFiltered_RGB = cv2.cvtColor(imFiltered, cv2.COLOR_HSV2RGB)
	imFiltered_gray = cv2.cvtColor(imFiltered, cv2.COLOR_RGB2GRAY) #y-values first

	# print(imFiltered_gray[0][0].shape)

	threshVal = 0
	maxVal = 255
	ret,thresh = cv2.threshold(imFiltered_gray,threshVal,maxVal,cv2.THRESH_BINARY_INV)
	#having issues doing decision making for when line is not detected
	contours,hierarchy = cv2.findContours(thresh[-400:-1], cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]

	try:
		biggestContour=max(contours, key=cv2.contourArea)
	except:
		forward = 0
		turn = 3
		sendDriveCommand(forward, turn)
	if cv2.contourArea(biggestContour) < 1000: #why 1000
		forward = 0
		turn = 3
		sendDriveCommand(forward, turn)
		return

	moment_array=cv2.moments(biggestContour)
	try:
		cx = int(moment_array['m10']/moment_array['m00'])
		cy = int(moment_array['m01']/moment_array['m00'])
	except ZeroDivisionError:
		#how do i do nothing
		print("divide by zero in moments")
	width = len(imFiltered_gray[0])
	rospy.loginfo("Width: " + str(width))

	# cx = scaleXCoord(cx, len(thresh[0]))

	#perhaps try implementing primitive PID
	if cx <= width / 2:
		forward = 0.25
		turn = -1
		sendDriveCommand(forward, turn)
	elif cx > width / 2:
		forward = 0.25
		turn = 1
		sendDriveCommand(forward, turn)
	# rospy.loginfo("Forward value: " + str(forward)  + "Turn value: " + str(turn))

	color = (0,0,0)
	radius = 15
	thickness = 30
	processedPhoto = cv2.circle(thresh[-400:-1], (cx, cy), radius, color, thickness)
	publishPhoto(processedPhoto)
	rospy.loginfo(imHSV[-1,0])

def scaleXCoord(coord, imWidth):
	return coord - imWidth // 3


def turnRange(value, fromMin, fromMax, toMin, toMax):
	fromRange = fromMax - fromMin
	toRange = toMax - toMin

	mappingValue = toRange / fromRange
	output = value * mappingValue - 0.5 

#called from lineDrive, publishes driving instructions to cmd_vel
def sendDriveCommand(forwardAmount, turnAmount):
	pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
	move = Twist()
	move.linear.x = forwardAmount
	move.angular.z = turnAmount
	pub.publish(move)

#http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
def publishPhoto(processed_image):
	imagepub = rospy.Publisher('/R1/processed_image', Image, queue_size=1)
	bridge = CvBridge()
	image_message = bridge.cv2_to_imgmsg(processed_image, encoding="passthrough")
	imagepub.publish(image_message)

if __name__ == '__main__':
	setup()

#while not rospy.is_shutdown():
# pub.publish(move)
#	interpret()
#	rate = rospy.Rate(2)
#	rate.sleep()