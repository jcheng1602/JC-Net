#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os,collections
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'###使用gpu:0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import scipy.misc as smi
import numpy as np
import tensorflow  as tf
import time
import cv2
import matplotlib.pyplot as plt

sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that tf is on the python path:
tf_root = '/home/ubuntu/DL/tf-segnet-cudnn5-master/'
sys.path.insert(0, tf_root + 'python')
import tf
def load_graph(frozen_graph_filename, prefix='', return_elements=None):
    # We load the protobuf file from the disk and parse it to retrieve the 
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        tf.import_graph_def(graph_def, name=prefix)
    
    return graph

def generate_image(image_dir):

    for path, dirs, files in os.walk(image_dir):
        for file in files:
            im_file = os.path.join(path,file)
            if ".png" in im_file:
                image = smi.imread(im_file)
                # image = image[280:792,:,:]
                image = smi.imresize(image, (384,960))
                # image = smi.imresize(image, (512,1280))

                image = np.expand_dims(image, 0)

                yield image, im_file

def copyDir(sourcePath,targetPath):###新加的，，，处理多个文件夹图片
    # 传入原目录,和需要复制后的目标目录
    # 判断需要复制的目录是否存在,如果不存在就返回
    if not os.path.isdir(sourcePath):
        return '源目录不存在'
    # 创建两个栈,一个用来存放原目录路径,另一个用来存放需要复制的目标目录
    sourceStack = collections.deque()
    sourceStack.append(sourcePath)

    targetStack = collections.deque()
    targetStack.append(targetPath)
    # 创建一个循环当栈里面位空时结束循环
    while True:
        if len(sourceStack) == 0:
            break
        # 将路径从栈的上部取出
        sourcePath = sourceStack.pop()  #sourcePath = sourceStack.popleft()
        # 遍历出该目录下的所有文件和目录
        listName = os.listdir(sourcePath)

        # 将目录路径取出来
        targetPath = targetStack.pop()  #targetPath = targetStack.popleft()
        # 判断该目标目录是否存在,如果不存在就创建
        if not os.path.isdir(targetPath):
            os.makedirs(targetPath)
        # 遍历目录下所有文件组成的列表,判断是文件,还是目录
        for name in listName:
            # 拼接新的路径
            sourceAbs = os.path.join(sourcePath, name)
            targetAbs = os.path.join(targetPath, name)
            # 判断是否时目录
            if os.path.isdir(sourceAbs):
                # 判断目标路径是否存在,如果不存在就创建一个
                if not os.path.exists(targetAbs):
                    os.makedirs(targetAbs)
                # 将新的目录添加到栈的顶部
                sourceStack.append(sourceAbs)
                targetStack.append(targetAbs)
# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
args = parser.parse_args()

net = tf.Net(args.model,
                args.weights,
                tf.TEST)

tf.set_mode_gpu()

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape

label_colours = cv2.imread(args.colours).astype(np.uint8)

cv2.namedWindow("Input")
cv2.namedWindow("JCNet")

cap = cv2.VideoCapture(0) # 使用相机
# cap = cv2.VideoCapture("/home/ubuntu/DL/SegNet-Tutorial-master/Scripts/epoch01_front.mkv")#传入视频
rval = True

if __name__ == '__main__':


    former_path = r'../temp_input'
    process_path = r'../temp_output'
    copyDir(former_path,process_path)

    ori_image = smi.imread(cap)
    ori_image = ori_image[100:919,:,:]
    ori_image = smi.imresize(ori_image, (512,1280))

    r,g,b = np.split(ori_image, 3,axis=2)
    # print(r.shape)

    new_image = np.concatenate((b,g,r),axis=2)
    # print(new_image.shape)

    ori_image = cv2.imread(im_file)
    ori_image = ori_image[100:919,:,:]
    ori_image = cv2.resize(ori_image, (512,1280))

    image = np.expand_dims(ori_image, 0)

    image = np.concatenate((image,image),axis=0)
    # print(image.shape)

    frozen_graph_filename = "/home/kevin/catkin_ws/src/gridmap/gridmap/src/model_mobilenet/frozen_model.pb"
    frozen_graph_filename = '/home/kevin/Codes/DeepNet/log/20180418_131625/frozen_model.pb'
    frozen_graph_filename ='/home/ubuntu/DL/deepnetV3/deeplabv3_cityscapes_train/frozen_model.pb'
    graph = load_graph(frozen_graph_filename, prefix='')
    
    for op in graph.get_operations():
        print(op.name)

    inputs = graph.get_tensor_by_name('ImagePlaceholder:0')
    output = graph.get_tensor_by_name('SemanticPredictions:0')

    print inputs.shape,output.shape

    image_dir = '../temp_input'
    image_gen = generate_image(image_dir)

    with tf.Session(graph=graph,config=config) as sess:
        pred = sess.run(output, feed_dict={inputs:cap})

        cv2.imshow('train_im',cap[0,:,:,:])
        k = cv2.waitKey(0)
        if k==27:
            cv2.destroyWindow('train_im')
        # road_id = pred[0,:,:,1].reshape((512,1280)) > 0.5
        road_id = pred[0,:,:,1].reshape((384,960)) > 0.5
        road_id = (pred == 0)[0,:,:]
        road_id = pred[0,:,:,1]
        
        road_id = cv2.resize(road_id,(1280,720)) > 0.6
        out_img = road_id.astype(np.uint8)*255
        out_img = cv2.resize(bi_im,(1280,720))
        print("out_img",out_img.shape)
        print(road_id)
        image = image[0,:,:,:]
        #cv2.imshow(image)
        out_img = cv2.cvtColor(out_img,cv2.COLOR_BGR2GRAY)
        end = time.time()
        print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'

        start = time.time()
        frame = cv2.resize(frame, (input_shape[3],input_shape[2]))
        input_image = frame.transpose((2,0,1))
        input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
        input_image = cv2.bitwise_not(input_image,input_image)
		#caculate S
		im_ = input_image.flatten()#区别： ravel()：如果没有必要，不会产生源数据的副本 
          #flatten()：返回源数据的副本 squeeze()：只能对维数为1的维度降维
		a = 0
		for i in im_:
			if a== 255:
				a++
		s = a/(input_image[0]*input_image[1])
		if a < 0.35:
			print("warning")
        end = time.time()
        print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

        start = time.time()
        out = net.forward_all(data=input_image)
        end = time.time()
        print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

        cv2.imshow("Input", frame)
        cv2.imshow("SegNet", out_img)
        
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break
        cap.release()
        cv2.destroyAllWindows()


