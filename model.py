import tensorflow as tf
import numpy as np

REGULARIZER_COF = 0.01


def _norm(x,name="BN",isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    s = tf.get_variable(name+"s", c,
                        initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    o = tf.get_variable(name+"o", c,
                        initializer=tf.constant_initializer(0.0))
    mean, var = tf.nn.moments(x, axes=[1,2], keep_dims=True)
    eps = 10e-10
    normalized = (x - mean) / (tf.sqrt(var) + eps)
    return s * normalized + o

def _fc_variable(weight_shape,name):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)
            regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)

            # define variables
            weight = tf.get_variable("w", weight_shape     ,
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer =regularizer)
            bias   = tf.get_variable("b", [weight_shape[1]],
                                    initializer=tf.constant_initializer(0.0))
        return weight, bias

def _conv_variable(weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _deconv_variable(weight_shape,name="deconv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels  = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape    ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [input_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def _deconv2d(x, W, output_shape, stride=1):
    # x           : [nBatch, height, width, in_channels]
    # output_shape: [nBatch, height, width, out_channels]
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

def MinibatchstateConcat(input, averaging='all'):

    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean((x - tf.reduce_mean(x, **kwargs)) **2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print ("nothing")
    vals = tf.tile(vals, multiples=[tf.shape(input)[0], 4, 4, 1])
    return tf.concat([input, vals], axis=3)

def _deconv(x,input_layer, output_layer, stride=2, filter_size=3, name="deconv", isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    deconv_w, deconv_b = _deconv_variable([filter_size,filter_size,input_layer,output_layer],name="deconv"+name )
    h = _deconv2d(x,deconv_w, output_shape=[bs,h*stride,w*stride,output_layer], stride=stride) + deconv_b
    return h

def _conv(x, input_layer, output_layer, stride, filter_size=5, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    return h

def resBlock_g(x,input_layer, output_layer,filter_size=3, name="deconv", isTraining=True):
    h = _norm(x,name="BN1_"+name,isTraining=isTraining)
    h = tf.nn.leaky_relu(h)
    h = _deconv(h,input_layer,output_layer,stride=2,filter_size=filter_size,name=name+"_1")
    h = _norm(h,name="BN2_"+name,isTraining=isTraining)
    h = tf.nn.leaky_relu(h)
    h = _deconv(h,output_layer,output_layer,stride=1,filter_size=filter_size,name=name+"_2")

    x = _deconv(x,input_layer,output_layer,stride=2,filter_size=filter_size,name=name+"_skip")
    return h+x

def resBlock_d(x,input_layer, output_layer,filter_size=3, name="conv", isTraining=True):
    #h = tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="BN1_"+name)
    h = tf.nn.leaky_relu(x)
    h = _conv(h,input_layer,output_layer,stride=2,filter_size=filter_size,name=name+"_1")
    #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="BN2_"+name)
    h = tf.nn.leaky_relu(h)
    h = _conv(h,output_layer,output_layer,stride=1,filter_size=filter_size,name=name+"_2")

    x = _conv(x,input_layer,output_layer,stride=2,filter_size=filter_size,name=name+"_skip")
    return h+x

def buildGenerator(z, z_dim,reuse=False, isTraining=True):
    with tf.variable_scope("Generator") as scope:
        if reuse: scope.reuse_variables()
        h = z
        # fc1
        g_fc1_w, g_fc1_b = _fc_variable([z_dim,1024*4*4],name="fc1")
        h = tf.matmul(h, g_fc1_w) + g_fc1_b
        h = tf.nn.relu(h)
        #
        h = tf.reshape(h,(-1,4,4,1024))

        h = resBlock_g(h,1024,512,name="g5")
        h = resBlock_g(h,512,256,name="g4")
        h = resBlock_g(h,256,128,name="g3")
        h = resBlock_g(h,128,64,name="g2")
        h = resBlock_g(h,64,64,name="g1")

        h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="BN_out")
        h = tf.nn.leaky_relu(h)
        g_deconv1_w, g_deconv1_b = _conv_variable([1,1,64,3],name="deconv1")
        h = _conv2d(h,g_deconv1_w, stride=1) + g_deconv1_b
        y = tf.tanh(h)

    return y

def buildDiscriminator(y, nBatch, reuse=False, isTraining=True):
    with tf.variable_scope("Discriminator") as scope:
        if reuse: scope.reuse_variables()
        h = y
        # conv1
        h = resBlock_d(h,3,64,name="d1")
        # conv2
        h = resBlock_d(h,64,128,name="d2")
        h = resBlock_d(h,128,256,name="d3")
        h = resBlock_d(h,256,512,name="d4")
        h = resBlock_d(h,512,512,name="d5")
        h = MinibatchstateConcat(h)
        print(h)
        # fc1
        n_b, n_h, n_w, n_f = [int(x) for x in h.get_shape()]
        h = tf.reshape(h,[nBatch,n_h*n_w*n_f])
        #print(h)
        d_fc1_w, d_fc1_b = _fc_variable([n_h*n_w*n_f,1],name="fc1")
        h = tf.matmul(h, d_fc1_w) + d_fc1_b

        ### summary
    return h
