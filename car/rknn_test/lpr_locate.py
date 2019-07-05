import cv2
import numpy as np

SZ = 20          #训练图片长宽
MAX_WIDTH = 800 #原始图片最大宽度
Min_Area = 2000  #车牌区域允许最大面积
PROVINCE_START = 1000
#读取图片文件
def imreadex(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

def point_limit(point):
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0

#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []
	for i,x in enumerate(histogram):
		if is_peak and x < threshold:
			if i - up_point > 2:
				is_peak = False
				wave_peaks.append((up_point, i))
		elif not is_peak and x >= threshold:
			is_peak = True
			up_point = i
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))
	return wave_peaks

#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
	part_cards = []
	width = waves[0][1] - waves[0][0]
	for wave in waves:
		start = (wave[0] + wave[1] - width)/2
		start = int(start)
		end = start + width

		resize = cv2.resize(img[:, start:end], (28,34))
		constant = cv2.copyMakeBorder(resize, 3,3,2,2, cv2.BORDER_CONSTANT, 0)
		part_cards.append(constant)
	return part_cards

open = 1
blur = 3
morphologyr = 4
morphologyc = 19
col_num_limit = 10
row_num_limit = 21

#精确定位
def accurate_place(card_img_hsv, limit1, limit2, color):
	row_num, col_num = card_img_hsv.shape[:2]
	H, S, V = cv2.split(card_img_hsv)

	area = np.where((H >= limit1) & (H <= limit2) & (S >= 34) & (V >= 46), 1, 0)

	row_sum = np.sum(area, axis=0)
	col_sum = np.sum(area, axis=1)
	col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5#绿色有渐变
	row_num_limit = row_num - 21

	row_pos = np.where(row_sum>=row_num_limit)[0]
	col_pos = np.where(col_sum>=col_num_limit)[0]
	xl, xr, yl, yh = 0, 0, 0, 0
	if len(row_pos) != 0:
		xl = row_pos[0]
		xr = row_pos[-1]
	if len(col_pos) != 0:
		yl = col_pos[0]
		yh = col_pos[-1]

	return xl, xr, yh, yl

#图片缩放
def scalar(img):
	pic_hight, pic_width = img.shape[:2]

	if pic_width > MAX_WIDTH:
		resize_rate = MAX_WIDTH / pic_width
		img = cv2.resize(img, (MAX_WIDTH, int(pic_hight*resize_rate)), interpolation=cv2.INTER_AREA)

	return img

def locate(car_pic):
	if type(car_pic) == type(""):
		img = imreadex(car_pic)
	else:
		img = car_pic
	scalar(img)
	pic_hight, pic_width = img.shape[:2]

	#高斯去噪
	if blur > 0:
		img = cv2.GaussianBlur(img, (blur, blur), 0)#图片分辨率调整
	oldimg = img
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#去掉图像中不会是车牌的区域
	kernel = np.ones((20, 20), np.uint8)
	img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0);

	#找到图像边缘
	ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	img_edge = cv2.Canny(img_thresh, 100, 200)
	#使用开运算和闭运算让图像边缘成为一个整体
	kernel = np.ones((morphologyr, morphologyc), np.uint8)
	img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
	img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

	#查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
	image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
	print('len(contours)', len(contours))
	#一一排除不是车牌的矩形区域
	car_contours = []
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		area_width, area_height = rect[1]
		if area_width < area_height:
			area_width, area_height = area_height, area_width
		wh_ratio = area_width / area_height
		#print(wh_ratio)
		#要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
		if wh_ratio > 2 and wh_ratio < 5.5:
			car_contours.append(rect)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			#newimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)

			#print(rect)

	print(len(car_contours))

	print("精确定位")
	card_imgs = []
	#矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
	for rect in car_contours:
		if rect[2] > -1 and rect[2] < 1:#创造角度，使得左、高、右、低拿到正确的值
			angle = 1
		else:
			angle = rect[2]
		rect = (rect[0], (rect[1][0]+5, rect[1][1]+5), angle)#扩大范围，避免车牌边缘被排除

		box = cv2.boxPoints(rect)
		heigth_point = right_point = [0, 0]
		left_point = low_point = [pic_width, pic_hight]
		for point in box:
			if left_point[0] > point[0]:
				left_point = point
			if low_point[1] > point[1]:
				low_point = point
			if heigth_point[1] < point[1]:
				heigth_point = point
			if right_point[0] < point[0]:
				right_point = point

		if left_point[1] <= right_point[1]:#正角度
			new_right_point = [right_point[0], heigth_point[1]]
			pts2 = np.float32([left_point, heigth_point, new_right_point])#字符只是高度需要改变
			pts1 = np.float32([left_point, heigth_point, right_point])
			M = cv2.getAffineTransform(pts1, pts2)
			dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
			point_limit(new_right_point)
			point_limit(heigth_point)
			point_limit(left_point)
			card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
			card_imgs.append(card_img)
			#cv2.imshow("card", card_img)
			#cv2.waitKey(0)

		elif left_point[1] > right_point[1]:#负角度
			new_left_point = [left_point[0], heigth_point[1]]
			pts2 = np.float32([new_left_point, heigth_point, right_point])#字符只是高度需要改变
			pts1 = np.float32([left_point, heigth_point, right_point])
			M = cv2.getAffineTransform(pts1, pts2)
			dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
			point_limit(right_point)
			point_limit(heigth_point)
			point_limit(new_left_point)
			card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
			card_imgs.append(card_img)
			#cv2.imshow("card", card_img)
			#cv2.waitKey(0)

	#开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
	colors = []
	for card_index,card_img in enumerate(card_imgs):
		if not card_img.size:
			continue

		green = yellow = blue = black = white = 0
		card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
		#有转换失败的可能，原因来自于上面矫正矩形出错
		if card_img_hsv is None:
			continue

		row_num, col_num= card_img_hsv.shape[:2]
		card_img_count = row_num * col_num

		H, S, V = cv2.split(card_img_hsv)
		#筛选对应颜色的图像数据，并进行二值化
		yellow_area = np.where((H > 11) & (H <= 34) & (S > 34), 1, 0)
		green_area = np.where((H > 35) & (H <= 99) & (S > 34), 1, 0)
		blue_area = np.where((H > 99) & (H <= 124) & (S > 34), 1, 0)
		black_area = np.where((H > 0) & (H <= 180) & (S > 0) & (S < 255) & (V > 0) & (V < 46), 1, 0)
		white_area = np.where((H > 0) & (H <= 180) & (S > 0) & (S < 43) & (V > 221) & (V < 225), 1, 0)

		#对二值化的数据进行求和，得到点的个数
		yellow = np.sum(yellow_area)
		green = np.sum(green_area)
		blue = np.sum(blue_area)
		black = np.sum(black_area)
		white = np.sum(white_area)

		limit1 = limit2 = 0
		color = ''
		if yellow*2 >= card_img_count:
			color = "yellow"
			limit1 = 11
			limit2 = 34#有的图片有色偏偏绿
		elif green*2 >= card_img_count:
			color = "green"
			limit1 = 35
			limit2 = 99
		elif blue*2 >= card_img_count:
			color = "blue"
			limit1 = 100
			limit2 = 124#有的图片有色偏偏紫
		elif black + white >= card_img_count*0.7:#TODO
			color = "bw"
		print(color)
		colors.append(color)
		if limit1 == 0:
			continue

		#以上为确定车牌颜色
		#以下为根据车牌颜色再定位，缩小边缘非车牌边界
		xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
		if yl == yh and xl == xr:
			continue
		need_accurate = False
		if yl >= yh:
			yl = 0
			yh = row_num
			need_accurate = True
		if xl >= xr:
			xl = 0
			xr = col_num
			need_accurate = True
		card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh-yl)//4 else card_img[yl-(yh-yl)//4:yh, xl:xr]
		if need_accurate:#可能x或y方向未缩小，需要再试一次
			card_img = card_imgs[card_index]
			card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
			xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
			if yl == yh and xl == xr:
				continue
			if yl >= yh:
				yl = 0
				yh = row_num
			if xl >= xr:
				xl = 0
				xr = col_num
		card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh-yl)//4 else card_img[yl-(yh-yl)//4:yh, xl:xr]
		roi = card_imgs[card_index]
	#以上为车牌定位
	#cv2.imshow("roi", roi)
	#cv2.waitKey(0)

	#以下为识别车牌中的字符
	predict_result = []
	roi = None
	card_color = None
	for i, color in enumerate(colors):
		if color in ("blue", "yellow", "green"):
			card_img = card_imgs[i]
			gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
			#cv2.imshow('gray_img', gray_img)
			#cv2.waitKey(0)
			#黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
			if color == "green" or color == "yellow":
				gray_img = cv2.bitwise_not(gray_img)
			ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			#查找水平直方图波峰
			x_histogram  = np.sum(gray_img, axis=1)
			x_min = np.min(x_histogram)
			x_average = np.sum(x_histogram)/x_histogram.shape[0]
			x_threshold = (x_min + x_average)/2
			wave_peaks = find_waves(x_threshold, x_histogram)
			if len(wave_peaks) == 0:
				print("peak less 0:")
				continue
			#认为水平方向，最大的波峰为车牌区域
			wave = max(wave_peaks, key=lambda x:x[1]-x[0])
			gray_img = gray_img[wave[0]:wave[1]]
			#查找垂直直方图波峰
			row_num, col_num= gray_img.shape[:2]
			#去掉车牌上下边缘1个像素，避免白边影响阈值判断
			gray_img = gray_img[1:row_num-1]
			y_histogram = np.sum(gray_img, axis=0)
			y_min = np.min(y_histogram)
			y_average = np.sum(y_histogram)/y_histogram.shape[0]
			y_threshold = (y_min + y_average)/5#U和0要求阈值偏小，否则U和0会被分成两半

			wave_peaks = find_waves(y_threshold, y_histogram)

			#for wave in wave_peaks:
			#	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
			#车牌字符数应大于6
			if len(wave_peaks) <= 6:
				print("peak less 1:", len(wave_peaks))
				continue

			wave = max(wave_peaks, key=lambda x:x[1]-x[0])
			max_wave_dis = wave[1] - wave[0]
			#判断是否是左侧车牌边缘
			if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
				wave_peaks.pop(0)

			#组合分离汉字
			cur_dis = 0
			for i,wave in enumerate(wave_peaks):
				if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
					break
				else:
					cur_dis += wave[1] - wave[0]
			if i > 0:
				wave = (wave_peaks[0][0], wave_peaks[i][1])
				wave_peaks = wave_peaks[i+1:]
				wave_peaks.insert(0, wave)

			#去除车牌上的分隔点
			point = wave_peaks[2]
			if point[1] - point[0] < max_wave_dis/3:
				point_img = gray_img[:,point[0]:point[1]]
				if np.mean(point_img) < 255/5:
					wave_peaks.pop(2)

			if len(wave_peaks) <= 6:
				print("peak less 2:", len(wave_peaks))
				continue
			part_cards = seperate_card(gray_img, wave_peaks)
			return card_img, part_cards

if __name__ == '__main__':
	locate("test/闽GLC185.jpg")

