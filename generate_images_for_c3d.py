import re
from moviepy.editor import *
import os
from scipy.misc import imsave, imresize
import math
import cv2 as cv
import time
import random
import numpy as np
from skimage.morphology import opening, disk, closing
from skimage.measure import label, regionprops
from random import shuffle
from copy import deepcopy
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
# Generate only for smoke using BgS - background substraction
class GenImg3DoneVideo:
	def __init__(self, size_of_images, duration, path_to_save, file_with_label, video_file,value_intersect, amount_move, onlyMove, info):
		# size of image tuple (height, width)
		self.image_size = size_of_images
		# duration of one frame sequence
		self.path_to_save = path_to_save
		# file with label
		# content of this file - 1000 smoke  fire [310.0, 253.0, 367.0, 156.99999999999997]
		# 1000 - frame number, after smoke and fire zero or more rectangles
		self.info = info
		self.amount_move = amount_move
		self.onlyMove = onlyMove
		self.value_intersect = value_intersect
		self.file = file_with_label
		self.video_name = video_file
		self.video = VideoFileClip(video_file)
		self.duration = int(duration * self.video.fps)
		self.frame = 0
		self.dict = self.get_info_from_file()
		self.step = self.get_step_of_label()
		self.amount_sequence_fire = 0
		self.amount_sequence_smoke = 0
		self.amount_sequence_none = 0
		self.mean_r = np.zeros((self.video.size[1], self.video.size[0], 5), dtype = np.uint8)
		self.mean_g = np.zeros((self.video.size[1], self.video.size[0], 5), dtype = np.uint8)
		self.mean_b = np.zeros((self.video.size[1], self.video.size[0], 5), dtype = np.uint8)
		self.frames = np.zeros((self.video.size[1], self.video.size[0], 3, self.duration), dtype= np.uint8)
		# self.one_frame = np.zeros((100, 100,3))


	def create_directories(self):
		name = ""
		if self.video_name.rfind('/') == -1:
			name = self.video_name[:self.video_name.rfind('.')]
		else:
			name = self.video_name[self.video_name.rfind("/") + 1:self.video_name.rfind('.')]

		self.path_to_smoke = self.path_to_save + '/' + name + '/smoke/'
		self.path_to_fire = self.path_to_save + '/' + name + '/fire/'
		self.path_to_none= self.path_to_save + '/' + name + '/none/'
		self.path_to_info = self.path_to_save + '/' + name + '/info/'
		if not os.path.exists(self.path_to_smoke):
			os.makedirs(self.path_to_smoke)
		else:
			# pass
			return None
		if not os.path.exists(self.path_to_fire):
			os.makedirs(self.path_to_fire)
		if not os.path.exists(self.path_to_none):
			os.makedirs(self.path_to_none)
		if not os.path.exists(self.path_to_info):
			os.makedirs(self.path_to_info)
		return 1

	def get_info_from_file(self):
		dict = {}
		with open(self.file) as f:
			str = f.readline()
			while str != '':
				str_split = self.split_string(str)
				info = self.get_info_from_one_line(str_split)
				if info != None:
					dict[int(str_split[0])] = info
				str = f.readline()
		return dict

	def split_string(self, str):
		pattern = re.compile('\[|\]|,')
		return re.sub(pattern, ' ', str).split()

	def get_info_from_one_line(self, str):
		if not "fire" in str:
			return None
		smoke_ind = str.index("smoke")
		fire_ind = str.index("fire")
		dict = {}
		dict["smoke"] = []
		dict["fire"] = []
		if (fire_ind - smoke_ind != 1):
			for i in range(0, (fire_ind - smoke_ind) / 4):
				dict["smoke"].append([int(float(k)) for k in str[smoke_ind + 1 + 4 * i: smoke_ind + 1 + 4 + 4 * i]])
		if (len(str) - fire_ind != 1):
			for i in range(0, (len(str) - fire_ind) / 4):
				dict["fire"].append([int(float(k)) for k in str[fire_ind + 1 + 4 * i: fire_ind + 1 + 4 + 4 * i]])
		return dict

	def get_step_of_label(self):
		with open(self.file) as f:
			str = f.readline()
			step = -1
			while str != '':
				str_split = self.split_string(str)
				info = self.get_info_from_one_line(str_split)
				if info != None:
					if step == -1:
						step = int(str_split[0])
					else:
						return int(str_split[0]) - step
				str = f.readline()
		return None

	#convert label coordinate in normal coordinate
	def label_coordinate_2_video_coordinate(self, rect):
		x0 = rect[0]
		x1 = rect[2]
		y0 = self.video.size[1] - rect[1]
		y1 = self.video.size[1] - rect[3]
		return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

	def generate_empty(self):
		amount_move = self.amount_move
		img_with_rect = self.video.get_frame((self.frame) / self.video.fps)
		for i in range(0, self.duration):
			self.frames[:,:,:,i] = self.video.get_frame((self.frame + i) / self.video.fps)
		if self.frame >= int(self.video.fps * 2) and self.frame <= int(self.video.duration * self.video.fps) - self.duration:

			for i in range (-2,3):
				temp_frame = self.video.get_frame((self.frame + i * 10) / self.video.fps)
				self.mean_r[:,:,i] = temp_frame[:,:,0]
				self.mean_g[:,:,i] = temp_frame[:,:,1]
				self.mean_b[:,:,i] = temp_frame[:,:,2]
			temp = np.dstack((np.median(self.mean_r, axis = 2), np.median(self.mean_g, axis = 2), np.median(self.mean_b, axis = 2)))

			temp_image = rgb2gray(abs(self.video.get_frame(self.frame / self.video.fps) - temp))
			# imsave(self.path_to_info + str(self.frame) + "_.jpg", temp_image)
			temp_image = (temp_image > 3) * 255
			if self.info:
				imsave(self.path_to_info + str(self.frame) + "__.jpg", temp_image)
			temp_image = opening(temp_image, disk(2))
			temp_image = closing(temp_image, disk(4))
			temp_image = opening(temp_image, disk(4))
			if self.info:
				imsave(self.path_to_info + str(self.frame) + "___.jpg", temp_image)
			x1 = label(temp_image)
			t = regionprops(x1)
			shuffle(t)
			for region in t:
				if region.area < 50:
					continue
				y, x, y_end, x_end = region.bbox
				x = (x + x_end) / 2 - 50
				y = (y + y_end) / 2 - 50
				x_end = x + 100
				y_end = y + 100
				if x_end > self.video.size[0] - 1:
					x_end = self.video.size[0] - 1
					x = x_end - 100
				if y_end > self.video.size[1] - 1:
					y_end = self.video.size[1] - 1
					y = y_end - 100
				if x < 0:
					x = 0
					x_end = x + 100
				if y < 0:
					y = 0
					y_end = y + 100
				if not self.check_intersection([x,y, x_end,y_end]) and amount_move != 0:
					self.cut_and_save([x,y,x_end, y_end], self.frame, "none")
					cv.rectangle(img_with_rect, (x, y),(x_end, y_end), (0,255, 0), 2)
					amount_move -= 1
				else:
					[IsIntersect, square, type] = self.check_intersection_with_event([x,y, x_end,y_end])
					# print square
					if IsIntersect and square > 0.7:
						self.cut_and_save([x,y,x_end, y_end], self.frame, type)
						if type == "fire":
							cv.rectangle(img_with_rect, (x, y),(x_end, y_end), (0,0, 255), 2)
						else:
							cv.rectangle(img_with_rect, (x, y), (x_end, y_end), (255, 0, 0), 2)

		for type in ["fire", "smoke"]:
			for rect1 in self.dict[self.frame][type]:
				if rect1 == [0, 0, 0, 0]:
					continue
				x0, y0, x1, y1 = self.label_coordinate_2_video_coordinate(rect1)
				if type == "fire":
					cv.rectangle(img_with_rect, (x0, y0),(x1, y1), (0,191, 255), 2)
				else:
					cv.rectangle(img_with_rect, (x0, y0), (x1, y1), (255, 69, 0), 2)
		if not self.onlyMove:
			for i in range(0, amount_move):
				amount_of_attempt = 10
				while (amount_of_attempt != 0):
					center_x = random.randint(1 + self.image_size[0] / 2, self.video.size[0]- 1 - self.image_size[0] / 2)
					center_y = random.randint(1 + self.image_size[1] / 2, self.video.size[1]- 1 - self.image_size[1] / 2)
					rect = [center_x - self.image_size[0]/ 2, center_y - self.image_size[1] / 2,
						            center_x + self.image_size[0] /2, center_y + self.image_size[1] / 2]
					if not self.check_intersection(rect):
						break
					else:
						amount_of_attempt -= 1
				if amount_of_attempt > 0:
					self.cut_and_save([center_x - self.image_size[0] / 2, center_y - self.image_size[0] / 2,
					                  center_x + self.image_size[0] / 2, center_y + self.image_size[0] / 2 ], self.frame, "none")
					cv.rectangle(img_with_rect, (center_x - self.image_size[0] / 2, center_y - self.image_size[0] / 2),
					             (center_x + self.image_size[0] / 2, center_y + self.image_size[0] / 2 ), (0,255, 0), 2)
		imsave(self.path_to_info + str(self.frame) + "____.jpg", img_with_rect)


	def check_intersection(self, rect):
		for type in ["fire", "smoke"]:
			for rect1 in self.dict[self.frame][type]:
				if rect1 == [0, 0, 0, 0]:
					continue
				x0, y0, x1, y1 = self.label_coordinate_2_video_coordinate(rect1)
				rect_temp = [x0, y0, x1, y1]
				[IsIntersect, square] = self.intersectionTwoRectangle(rect, rect_temp)
				if IsIntersect:
					return True
		return False

	def check_intersection_with_event(self, rect):
		for type in ["fire","smoke"]:
			for rect1 in self.dict[self.frame][type]:
				if rect1 == [0, 0, 0, 0]:
					continue
				x0, y0, x1, y1 = self.label_coordinate_2_video_coordinate(rect1)
				rect_temp = [x0, y0, x1, y1]
				[IsIntersect, square] = self.intersectionTwoRectangle(rect, rect_temp)
				if IsIntersect:
					return True, square, type
		return False, 0, ""

	def main_loop(self):
		print self.video_name, self.video.duration
		random.seed()
		self.frame = 2700
		while (self.dict.get(self.frame, -1) != -1):
			self.generate_empty()
			self.frame += self.step
			if self.frame % 500 == 0:
				print self.frame, self.video_name

	def cut_and_save(self, center, start_frame, type):
		if type == "fire":
			path = self.path_to_fire
			number = self.amount_sequence_fire
			self.amount_sequence_fire += 1
		elif type == "smoke":
			path = self.path_to_smoke
			number = self.amount_sequence_smoke
			self.amount_sequence_smoke += 1
		else:
			path = self.path_to_none
			number = self.amount_sequence_none
			self.amount_sequence_none += 1
		if self.video.duration * self.video.fps < self.frame + self.duration:
			return
		# for i in range(0, int(self.duration * self.video.fps)):
		# for i in [0,5,6,7,8,10,12,14,16, 15, 18,21,24]:
		for i in range(0, self.duration):
		# 	frame = self.video.get_frame((start_frame + i) / float(self.video.fps))
			#frame = self.frames[:,:,:,i]
			# frame1 = deepcopy(self.frames[center[1]:center[3], center[0]:center[2],:,i])
			self.one_frame = deepcopy(self.frames[center[1]:center[3], center[0]:center[2],:,i].astype(np.uint8))
			if center[3] - center[1] == 0 or center[2] - center[0] == 0:
				if type == "fire":
					self.amount_sequence_fire -= 1
				if type == "smoke":
					self.amount_sequence_smoke -= 1
				if type == "none":
					self.amount_sequence_none -= 1
				return
			#frame1 = imresize(frame1, (100, 100))
			# imsave(path + "%06d.jpg" % (number * int(self.duration * self.video.fps) + i + 1), frame1)
			imsave(path + "%06d.jpg" % (number * self.duration + i + 1), self.one_frame)

	def intersectionTwoRectangle(self, rect1, rect2):
		# rect2 - label
		isIntersect = not (rect1[2] < rect2[0] or rect1[3] < rect2[1] or rect2[2] < rect1[0] or rect2[3] < rect1[1])
		x0 = y0 =  x1 =  y1 = 0
		if isIntersect:
			x0 = max(rect1[0], rect2[0])
			y0 = max(rect1[1], rect2[1])
			x1 = min(rect1[2], rect2[2])
			y1 = min(rect1[3], rect2[3])
		return isIntersect, abs(((x1 - x0) * (y1 - y0) / float(((rect1[0] - rect1[2]) * (rect1[1] - rect1[3])))))

from multiprocessing import Pool
path_to_data = "/home/sasha/Desktop/data/"
#folders = ['freelance4']
folders = ['freelance3', 'freelance1', 'freelance2']
total_list_video = []
total_list_label = []
total_list = []
for folders in folders:
	for path, subdires, files in os.walk(path_to_data + folders):
		for one_file in files:
			if '.mp4' in one_file:
				temp = one_file[:one_file.rfind('.')] + '_label.txt'
				#print temp
				if os.path.exists(os.path.join(path, temp)):
					total_list_video.append(os.path.join(path, one_file))
					total_list_label.append(os.path.join(path, temp))
					total_list.append([os.path.join(path, one_file), os.path.join(path, temp)])

#path_to_save = "/home/sasha/Desktop/data_for_3d_test"
# path_to_save = '/media/sasha/11ef7cf5-cdd6-44ca-9e5f-dd60cdb781da/sasha/onlySmoke3d_200200-4'
#path_to_save = "/media/sasha/11ef7cf5-cdd6-44ca-9e5f-dd60cdb781da/sasha/data_for_3d_smoke"
#path_to_save = "/home/sasha/Desktop/data_c3d_back_smoke"
path_to_save = "/home/sasha/Desktop/temp_data"
def generate_images_for_one_video(info):
	start = time.time()
	temp = GenImg3DoneVideo([100, 100], 1, path_to_save, info[1], info[0], 0.7, 7, True, True)
	#temp.create_directories()
	res = temp.create_directories()
	# if res == None:
	# 	print info[0]
	# 	return
	temp.main_loop()
	print info[0],time.time() - start,temp.amount_sequence_fire,temp.amount_sequence_smoke,temp.amount_sequence_none

# number_smoke = 0
# for info in os.listdir(path_to_save):
# 	number_smoke += len(os.listdir(path_to_save + '/' + info + '/none')) / 25
# print number_smoke
for l in total_list:
	generate_images_for_one_video(l)
#generate_images_for_one_video(["/home/sasha/Desktop/data/freelance3/yard21.mp4", "/home/sasha/Desktop/data/freelance3/yard21_label.txt"])
# pool = Pool(9)
# pool.map(generate_images_for_one_video, total_list)
# pool.close()
# pool.join()