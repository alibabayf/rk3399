from rknn.api import RKNN

INPUT_SIZE = 64

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(host='rk3399pro')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Config for Model Input PreProcess
    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2')
    #rknn.config(channel_mean_value='0 0 0 255', reorder_channel='2 1 0')

    # Load TensorFlow Model
    print('--> Loading model')
    rknn.load_tensorflow(tf_pb='../digital_gesture_recognition/model_2500/digital_gesture.pb',
                         inputs=['input_x'],
                         outputs=['probability'],
                         input_size_list=[[INPUT_SIZE, INPUT_SIZE, 3]])
    print('done')

    # Build Model
    print('--> Building model')
    rknn.build(do_quantization=False, dataset='./dataset.txt')
    print('done')

    # Export RKNN Model
    rknn.export_rknn('./digital_gesture.rknn')

    # Release RKNN Context
    rknn.release()
