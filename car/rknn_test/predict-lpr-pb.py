import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import lpr_locate

SIZE = 1280

PROVINCE_SAVER_DIR = "../train-saver/province/"
DIGITS_SAVER_DIR = "../train-saver/digits/"

PROVINCES = ("京","闽","粤","苏","沪","浙")
LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z")

def trans_digit(data):
	digits = ""
	for i in range(len(data)):
		digits += LETTERS_DIGITS[data[i]]
	return digits

def load_model(path):
	with tf.gfile.GFile('{}/lpr_predict.pb'.format(path), "rb") as f:  #读取模型数据
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read()) #得到模型中的计算图和数据
	with tf.Graph().as_default() as graph:  # 这里的Graph()要有括号，不然会报TypeError
		tf.import_graph_def(graph_def, name="")  # 导入模型中的图到现在这个新的计算图中
	return graph

def predict(modle_path, parts):
	graph = load_model(modle_path)
	reshape = graph.get_tensor_by_name("Reshape:0")
	prob = graph.get_tensor_by_name("probability:0")

	predict = []
	with tf.Session(graph=graph) as sess:
		# 遍历字符并给出预测结果
		for img in parts:
			# 转换成二值化图像
			_, img_binary = cv.threshold(img, 150, 1, cv.THRESH_BINARY_INV)
			# Reshape节点的输入需要是(1, 32, 40, 1)形状的
			data = img_binary.reshape(1, 32, 40, 1)
			_prob = sess.run([prob],feed_dict={reshape:data})
			_prob = _prob[0][0].tolist()
			max_index = _prob.index(max(_prob))	# 返回概率最大处的索引号
			predict.append(max_index)			# 添加预测结果到队列

	return predict

def predict_lpr(parts):
	lpr = ""
	# 第一个字符为省份，使用省份的模型进行预测
	province = predict(PROVINCE_SAVER_DIR, parts[:1])
	lpr += PROVINCES[province[0]]

	# 后续的字符为数字和字母，使用数字字母的模型进行预测
	digits = predict(DIGITS_SAVER_DIR, parts[1:])
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
	img = cv.imread('京Q73Y60.jpg')
	card, parts = lpr_locate.locate(img)

	show_card(img, card, parts)
	lpr = predict_lpr(parts)
	print(lpr)

	cv.destroyAllWindows()
