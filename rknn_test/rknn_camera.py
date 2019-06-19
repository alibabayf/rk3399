import numpy as np
from PIL import	Image
from rknn.api import RKNN
import cv2 as cv

def get_predict(probability):
	data = probability[0][0]
	l_data = data.tolist()
	max_prob = max(l_data)
	return l_data.index(max_prob), max_prob;

def load_model():
	# Create RKNN object
	rknn = RKNN()

	print('-->loading model')
	rknn.load_rknn('./digital_gesture.rknn')
	print('loading model done')

	# init runtime environment
	print('--> Init runtime	environment')
	ret = rknn.init_runtime(host='rk3399pro')
	if ret != 0:
		print('Init runtime environment	failed')
		exit(ret)
	print('done')

	return rknn

def predict(rknn):
	font = cv.FONT_HERSHEY_SIMPLEX;
	capture	= cv.VideoCapture(0)
	while (True):
		ret,frame = capture.read()
		if ret == True:
			image = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
			image =	cv.resize(image, (64, 64), interpolation = cv.INTER_AREA)

			# Inference
			outputs	= rknn.inference(inputs=[image])
			pred, prob = get_predict(outputs)

			cv.putText(frame, 'pred:{}'.format(pred), (50, 50), font, 1.2, (0, 0, 255), 2)
			cv.putText(frame, 'prob:{}'.format(prob), (50, 100), font, 1.2, (0, 0, 255), 2)
			cv.imshow("camera", frame)
			c = cv.waitKey(10) & 0xff
			if c ==	27:
				capture.release()
				rknn.release()
				break

if __name__=="__main__":
	rknn = load_model()
	predict(rknn)
