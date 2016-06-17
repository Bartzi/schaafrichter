import argparse
import threading

from collections import namedtuple
from time import sleep


import cv2
import numpy as np

from pydub import AudioSegment
from pydub.playback import play

from camera import CameraHandler

do_play = False
abort = False

cv_rect = namedtuple("CVRect", ['x', 'y', 'width', 'height'])
cv_size = namedtuple("CVSize", ['width', 'height'])

def play_thread():
	global do_play
	while True:
		if do_play:
			play(song)
			do_play = False
		sleep(0.5)
		if abort:
			break

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='scene text recognition live demo')
	parser.add_argument('-c', '--camera', dest='camera', default=0, type=int, help='camera id to use')

	args = parser.parse_args()

	camera_handler = CameraHandler(args.camera)

	ORANGE_MIN = np.array([1, 50, 50], np.uint8)
	ORANGE_MAX = np.array([15, 255, 255], np.uint8)

	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 1
	thickness = 2

	audio_file = 'sheep.mp3'
	song = AudioSegment.from_mp3(audio_file)

	thread = threading.Thread(target=play_thread)
	thread.start()

	last = None

	with camera_handler as camera:
		while True:
			try:
				frame = camera.get_frame()

				frame = cv2.medianBlur(frame, 3)
				img = frame.copy()

				hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				threshold = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)

				eroded = cv2.erode(threshold, np.ones((3, 3), dtype=np.uint8), iterations=2)
				dilated = cv2.dilate(eroded, np.ones((3, 3), dtype=np.uint8), iterations=2)

				image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				if len(contours) == 0:
					areas = [0]
					best = 0
				else:
					areas = [cv2.contourArea(contour) for contour in contours]
					best = np.argmax(areas)
					bounding_rect = cv_rect._make(cv2.boundingRect(contours[best]))
					cv2.rectangle(img, (bounding_rect.x, bounding_rect.y), (bounding_rect.x + bounding_rect.width, bounding_rect.y + bounding_rect.height), (0, 255, 0))

				text_size = cv_size._make(cv2.getTextSize(str(areas[best]), font, font_scale, thickness)[0])
				left_corner = 0
				bottom_corner = text_size.height
				bottom_left = (left_corner, bottom_corner)
				cv2.putText(img, str(areas[best]), bottom_left, font, font_scale, (0, 255, 0), thickness)

				if areas[best] < 200:
					continue

				do_play = True
				

			finally:
				cv2.imshow('original', img)

				pressed_key = cv2.waitKey(1) & 0xff
				if pressed_key == ord('q'):
					break
	abort = True			
