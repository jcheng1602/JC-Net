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

######config
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.90
config.gpu_options.allow_growth = True # 允许自增长
# config.log_device_placement=True # 允许打印gpu使用日志
# config.gpu_options.per_process_gpu_memory_fraction=0.333
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def load_graph(frozen_graph_filename, prefix='', return_elements=None):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
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
            #判断是否是文件
            # if os.path.isfile(sourceAbs):
            #     # 1.如果目标子级文件不存在 或者目标子级文件存在但是该文件与原子级文件大小不一致 则需要复制
            #     if (not os.path.exists(targetAbs)) or (os.path.exists(targetAbs) and os.path.getsize(targetAbs) != os.path.getsize(targetAbs)):
            #         rf = open(sourceAbs, mode='rb')
            #         wf = open(targetAbs, mode='wb')
            #         while True:
            #             # 一点一点读取,防止当文件较大时候内存吃不消
            #             content = rf.read(1024*1024*10)
            #             if len(content) == 0:
            #                 break
            #             wf.write(content)
            #             # 写入缓冲区时候手动刷新一下,可能会加快写入
            #             wf.flush()
            #         # 读写完成关闭文件
            #         rf.close()
            #         wf.close()


# def generate_image(image_dir):

#     for path, dirs, files in os.walk(image_dir):
#         for file in files:
#             im_file = os.path.join(path,file)
#             if ".png" in im_file:
#                 image = smi.imread(im_file)
#                 # image = cv2.imread(im_file)
#                 # plt.imread(im)

#                 # r,b,g = np.split(image, 3, axis=-1)
#                 # new_image = np.concatenate([b,r,g],axis=2)
#                 # image = image[280:792,:,:]
#                 # image = smi.imresize(image, (384,960))
#                 # image = smi.imresize(image, (512,1280))

#                 image = np.expand_dims(image, 0)

#                 yield image, im_file

if __name__ == '__main__':


    former_path = r'../temp_input'
    process_path = r'../temp_output'
    copyDir(former_path,process_path)

    # im_file = '/home/kevin/Codes/DeepNet/dataset/cityscapes/leftImg8bit/train/stuttgart/stuttgart_000007_000019_leftImg8bit.png'
    # ori_image = smi.imread(im_file)
    # ori_image = ori_image[100:919,:,:]
    # ori_image = smi.imresize(ori_image, (512,1280))

    # # print(ori_image.__class__)
    # # r,g,b = np.split(ori_image, 3,axis=2)
    # # print(r.shape)

    # # new_image = np.concatenate((b,g,r),axis=2)
    # # print(new_image.shape)

    # # ori_image = cv2.imread(im_file)
    # # ori_image = ori_image[100:919,:,:]
    # # ori_image = cv2.resize(ori_image, (512,1280))

    # image = np.expand_dims(ori_image, 0)
    # print(image.shape)

    # image = np.concatenate((image,image),axis=0)
    # print(image.shape)

    # frozen_graph_filename = "/home/kevin/catkin_ws/src/gridmap/gridmap/src/model_mobilenet/frozen_model.pb"
    # frozen_graph_filename = '/home/kevin/Codes/DeepNet/log/20180418_131625/frozen_model.pb'
    frozen_graph_filename ='/home/ubuntu/DL/deepnetV3/deeplabv3_cityscapes_train/frozen_inference_graph.pb'
    # frozen_graph_filename ='/home/ubuntu/DL/deepnetV3/deeplabv3_cityscapes_train/frozen_model.pb'
    graph = load_graph(frozen_graph_filename, prefix='')
    
    # for op in graph.get_operations():
    #     print(op.name)
    # inputs = graph.get_tensor_by_name('ImagePlaceholder:0')#my_
    inputs = graph.get_tensor_by_name('ImageTensor:0')###deeplabv3+:ImageTensor

    output = graph.get_tensor_by_name('SemanticPredictions:0')

    print inputs.shape,output.shape

    image_dir = '../temp_input'
    image_gen = generate_image(image_dir)



    
    with tf.Session(graph=graph,config=config) as sess:

        for image, im_file in image_gen:
            
            pro_path = im_file.replace("../temp_input/","")
            print pro_path

            # cv2.imshow('train_im',image[0,:,:,:])
            # k = cv2.waitKey(0)
            # if k==27:
            #     cv2.destroyWindow('train_im')

            # plt.figure(1)
            # plt.imshow(image[0,:,:,:])
            # plt.show()
            # print("enter")
            # print("image_gen",image.shape)
            time1 = time.time()
            pred = sess.run(output, feed_dict={inputs:image})
            time2 = time.time()
            
            #print("cost time", time2-time1)
            # print(im_file)
            # print('pred',pred.shape)
            # print('image',image.shape)

            
            #road_id = pred[0,:,:,1].reshape((512,1280)) > 0.5
            # road_id = pred[0,:,:,1].reshape((384,960)) > 0.5
            # road_id = (pred == 0)[0,:,:]
            # road_id = pred[0,:,:,1]#my_
            road_id = pred[:,:,:].reshape((384,960))> 0.6
            # print "ok" 
            # print( road_id.shape,type(road_id))
            
            # road_id = cv2.resize(road_id,(1280,720)) > 0.6#my_
            # road_id = cv2.resize(road_id,(3384,2710)) > 0.6#(w,h)
            # road_id = cv2.resize(road_id,(1280,720)) > 0.6
            out_img = road_id.astype(np.uint8)*255
            print out_img.shape
            #out_img = cv2.resize(bi_im,(1280,720))
            #print("out_img",out_img.shape)
            # print(road_id)
            # # print(road_id.shape)
            # image = image[0,:,:,:]
            # #cv2.imshow(image)
            #out_img = cv2.cvtColor(out_img,cv2.COLOR_BGR2GRAY)
            # print("img.shape",image.shape)
            # image[road_id,1] = 255
             # = cv2.resize(img,(dstWidth,dstHeight))


            # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # image = image[road_id,1]= 255
            #save_name = './temp_output/' + os.path.basename(im_file)
            #print("im_file:",im_file)

            save_name = '../temp_output/' + pro_path
            #print("save_name:",save_name)
            smi.imsave(save_name, out_img)

            # break

