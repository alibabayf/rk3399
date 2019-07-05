import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
from rknn.api	import RKNN
import lpr_locate

PROVINCES = ("京","闽","粤","苏","沪","浙")
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z")

def trans_digit(data):
	digits = ""
	for i in range(len(data)):
		digits += LETTERS_DIGITS[data[i]]
	return digits

def load_model(modle_path):
	# Create RKNN object
	rknn = RKNN()

	print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
	print('-->loading model')
	rknn.load_rknn(modle_path)
	print('loading model done')

	# init runtime environment
	print('--> Init runtime environment')
	ret = rknn.init_runtime()
	if ret != 0:
		print('Init runtime environment failed')
		exit(ret)
	print('done')
	print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
	return rknn

def predict(rknn, parts):
	predict = []

	# 遍历字符并给出预测结果
	for img in parts:
		# 转换成二值化图像
		_, img_binary = cv.threshold(img, 150, 1, cv.THRESH_BINARY_INV)
		# Reshape节点的输入需要是(1, 32, 40, 1)形状的
		data = img_binary.reshape(1, 32, 40, 1)
		_prob = rknn.inference(inputs=[data])
		_prob = _prob[0][0].tolist()
		max_index = _prob.index(max(_prob))	# 返回概率最大处的索引号
		predict.append(max_index)			# 添加预测结果到队列

	return predict

def predict_lpr(rknn_province, rknn_digits, parts):
	lpr = ""
	# 第一个字符为省份，使用省份的模型进行预测
	province = predict(rknn_province, parts[:1])
	lpr += PROVINCES[province[0]]

	# 后续的字符为数字和字母，使用数字字母的模型进行预测
	digits = predict(rknn_digits, parts[1:])
	lpr += trans_digit(digits)

	return lpr

def show_card(img, card, parts):
	plt.figure(figsize = (10, 5))
	#绘制原图
	plt.subplot(1, 2, 1)
	pil_card = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	plt.imshow(pil_card), plt.axis('off')

	#绘制车牌
	plt.subplot(2, 2, 2)
	pil_card = cv.cvtColor(card, cv.COLOR_BGR2RGB)
	plt.imshow(pil_card), plt.axis('off')

	#绘制分割后的字符
	for i, part in enumerate(parts):
		plt.subplot(2, 2*len(parts), i + 1 + 3*len(parts))
		pil_part = Image.fromarray(np.uint8(part))
		plt.imshow(pil_part), plt.axis('off')

	plt.savefig('tmp.png', format = 'png', bbox_inches = 'tight')
	test_img = cv.imread('tmp.png')
	cv.imshow("LPR predict",test_img)
	cv.waitKey(0)

if __name__ =='__main__':
	rknn_province = load_model('./lpr_province_predict.rknn')
	rknn_digits = load_model('./lpr_digits_predict.rknn')

	img = cv.imread('京Q73Y60.jpg')
	card, parts = lpr_locate.locate(img)

	show_card(img, card, parts)
	lpr = predict_lpr(rknn_province, rknn_digits, parts)
	print(lpr)
	cv.destroyAllWindows()
	rknn_province.release()
	rknn_digits.release()

