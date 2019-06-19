import numpy as np
from PIL import	Image
from rknn.api import RKNN

def get_predict(probability):
	data = probability[0][0]
	data = data.tolist()
	max_prob = max(data)
	return data.index(max_prob), max_prob;

def load_model():
	# Create RKNN object
	rknn = RKNN()

	print('-->loading model')
	rknn.load_rknn('./digital_gesture.rknn')
	print('loading model done')

	# init runtime environment
	print('--> Init	runtime	environment')
	ret = rknn.init_runtime(host='rk3399pro')
	if ret != 0:
		print('Init runtime environment	failed')
		exit(ret)
	print('done')

	return rknn

def predict(rknn):
	im = Image.open("../picture/6_7.jpg")
	im = im.resize((64, 64),Image.ANTIALIAS)
	mat = np.asarray(im.convert('RGB'))
	outputs	= rknn.inference(inputs=[mat])
	pred, prob = get_predict(outputs)

	print(prob)
	print(pred)

if __name__=="__main__":
	# Create RKNN	object
	rknn = load_model()

	predict(rknn)
