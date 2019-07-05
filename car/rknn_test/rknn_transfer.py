from rknn.api import RKNN

INPUT_WIDTH = 32
INPUT_HEIGHT = 40

def transfer(pb_path, rknn_name):
  # Create RKNN object
  rknn = RKNN()

  # init runtime environment
  print('--> Init runtime environment')
  ret = rknn.init_runtime()
  if ret != 0:
      print('Init runtime environment failed')
      exit(ret)
  print('done')

  # Config for Model Input PreProcess
  rknn.config()

  # Load TensorFlow Model
  print('--> Loading model')
  rknn.load_tensorflow(tf_pb=pb_path,
                       inputs=['Reshape'],
                       outputs=['probability'],
                       input_size_list=[[INPUT_WIDTH, INPUT_HEIGHT, 1]])
  print('done')

  # Build Model
  print('--> Building model')
  rknn.build(do_quantization=False)
  print('done')

  # Export RKNN Model
  rknn.export_rknn(rknn_name)

  # Release RKNN Context
  rknn.release()

if __name__ == '__main__':
	transfer('../train-saver/digits/lpr_predict.pb', 'lpr_digits_predict.rknn')
	transfer('../train-saver/province/lpr_predict.pb', 'lpr_province_predict.rknn')
