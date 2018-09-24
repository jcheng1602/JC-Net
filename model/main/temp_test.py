import tensorflow as tf 

#from tensorflow.contrib import tensorrt as trt


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

def get_graph_def(pb_file):
    with tf.gfile.FastGFile(pb_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


if __name__ == "__main":

    sess = tf.InteractiveSession()

    c = tf.placeholder(dtype=tf.int32, shape=(1,28,28,3))
    a = tf.placeholder(dtype=tf.int32, shape=())

    b = a + 1

    print(sess.run(b, feed_dict={a:1}))
    print(tf.__version__)


    graph_file = '/home/yu/File/deepnetV3/deeplabv3_cityscapes_train/frozen_inference_graph.pb'
    input_graph = get_graph_def(graph_file)
    outputs = ['SemanticPredictions:0']

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)

    # trt_graph = trt.create_inference_graph(input_graph, outputs,
    #                                 max_workspace_size_bytes=1<<25)

    with tf.gfile.FastGFile('/home/kevin/deepnet_FP32.pb','wb') as f:
        f.write(trt_graph.SerializeToString())


    