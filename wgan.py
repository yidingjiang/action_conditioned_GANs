import tensorflow as tf
import numpy as np
import os
import pickle
import argparse
print('Finished imports')

class WGAN:

    def __init__(self, input_frame, label, real_frames, phase, input_dim=64, frame_count=32):
        self.input_width, self.input_height = input_dim, input_dim
        self.frame_count = frame_count
        self.robot_state, self.robot_action = 5, 5
        self.G_weight = []
        self.D_weight = []
        self.build_graph(input_frame, label, real_frames, phase)


    def build_graph(self, input_frame, label, real_frames, phase):
        """
        Build the computational graph
        """
        print('[*] starting graph construction')
        with tf.name_scope('data'):
            self.x = tf.placeholder('float32', (None, self.input_height, self.input_width, 3), name='input_frame')
            self.y = tf.placeholder('float32', (None, 1), name='label')
            self.true_frames = tf.placeholder('float32', (None, self.frame_count, self.input_height, self.input_width, 3), name='true_frames')
            self.phase = tf.placeholder(tf.bool, name='phase')

        self.G_train = self.naive_generator(self.x, reuse=False)
        self.G_sample = self.naive_generator(self.x, reuse=True)
        self.D_real = self.discriminator(self.true_frames, self.y, reuse=False)
        self.D_sample = self.discriminator(self.G_train, self.y, reuse=True)
        print("[*] all graphs compiled")

        trainable_vars = tf.trainable_variables()
        self.D_weight = [var for var in trainable_vars if 'd_' in var.name]
        self.G_weight = [var for var in trainable_vars if 'g_' in var.name]
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.D_weight]

        self.D_loss = tf.reduce_mean(self.D_real) - tf.reduce_mean(self.D_sample)
        self.G_loss = -tf.reduce_mean(self.D_sample)
        self.D_solver = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(-self.D_loss, var_list=self.D_weight))

        self.G_solver = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(self.G_loss, var_list=self.G_weight))
        print("[*] all solvers compiled")


    def train(self, epoch_num=2000, D_per_G=5):
        """
        Traning the network for specific number of epoch.
        Train the discriminator *D_per_G* times for every one training of generator

        param
            epoch_num:  maximum number of iteration for training
            D_per_G:    number of times discriminator is trained for every single time
                        generator is trained
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print('[*] finish initialize all variables...')
        for i in range(epoch_num):
            for _ in range(D_per_G):
                real = self.load_single_batch('test_data.npy')
                first_frame = real[:,0,:,:,:]
                _, D_loss_curr, _ = sess.run(
                        [self.D_solver, self.D_loss, self.clip_D],
                        feed_dict={self.true_frames: real, self.x: first_frame}
                        )
            sample_initial_frame = self.load_single_batch('test_data.npy')[:,0,:,:,:]
            _, G_loss_curr = sess.run(
                        [self.G_solver, self.G_loss],
                        feed_dict={self.x: sample_initial_frame}
                        )

            if i % 100 == 0:
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
                      .format(i, D_loss_curr, G_loss_curr))

                if i % 100 == 0:
                    initial_frame = self.load_single_batch('test_data.npy')[:,0,:,:,:]
                    samples = sess.run(self.G_sample, feed_dict={self.x: initial_frame})
                    np.save('./output/sample'+str(i), np.array(samples))


    def load_single_batch(self, fname):
        data = np.load("./data/" + fname)
        return (data / 255.)

    def naive_generator(self, input_frame, reuse):
        """
        Build a non-conditioned generator

        param
            input_frame:    a tensorflow placeholder for the input_frame (32*32*3)
            reuse:          specify whether the previous variables are shared
        """
        with tf.variable_scope('generator') as scope:
            train = not reuse
            if reuse:
                scope.reuse_variables()
                train = False
            network = self.conv2d(input_frame, [4,4,3,64], [1,2,2,1], 'g_cv2_1', group=self.G_weight) #32*32
            network = tf.nn.relu(self.batchnorm(network, train, 'g_bn_1', group=self.G_weight))
            network = self.conv2d(network, [4,4,64,128], [1,2,2,1], 'g_cv2_2', group=self.G_weight) #16*16
            network = tf.nn.relu(self.batchnorm(network, train, 'g_bn_2', group=self.G_weight))
            network = self.conv2d(network, [4,4,128,256], [1,2,2,1], 'g_cv2_3', group=self.G_weight) #8*8
            network = tf.nn.relu(self.batchnorm(network, train, 'g_bn_3', group=self.G_weight))
            network = self.conv2d(network, [4,4,256,512], [1,2,2,1], 'g_cv2_4', group=self.G_weight) #4*4
            network = tf.nn.relu(self.batchnorm(network, train, 'g_bn_4', group=self.G_weight))
            network = tf.reshape(network, (5,1,4,4,512))
            network = self.deconv3d(network, [1,4,4,512,512], [5,2,4,4,512], [1,2,1,1,1], 'g_dcv3_1', group=self.G_weight)
            network = tf.nn.relu(self.batchnorm(network, train, 'g_bn_5', group=self.G_weight))
            network = self.deconv3d(network, [2,4,4,256,512], [5,4,8,8,256], [1,2,2,2,1], 'g_dcv3_2', group=self.G_weight)
            network = tf.nn.relu(self.batchnorm(network, train, 'g_bn_6', group=self.G_weight))
            network = self.deconv3d(network, [4,4,4,128,256], [5,8,16,16,128], [1,2,2,2,1], 'g_dcv3_3', group=self.G_weight)
            network = tf.nn.relu(self.batchnorm(network, train, 'g_bn_7', group=self.G_weight))
            network = self.deconv3d(network, [4,4,4,64,128], [5,16,32,32,64], [1,2,2,2,1], 'g_dcv3_4', group=self.G_weight)
            network = tf.nn.relu(self.batchnorm(network, train, 'g_bn_8', group=self.G_weight))
            depth_step_size = int(float(self.frame_count)/32*2) #use a *full* convolution to adjust the output size
            network = self.deconv3d(network, [4,4,4,3,64], [5,self.frame_count,64,64,3], [1,depth_step_size,2,2,1], 'g_dcv3_5', group=self.G_weight)
        return network


    def discriminator(self, frames, label, reuse=False):
        """
        Build a discriminator generator

        param:
            frames:     a tensorflow placeholder for a searies of frames (num_frames*32*32*3)
            reuse:      specify whether the previous variables are shared
        """
        with tf.variable_scope('discriminator') as scope:
            train = not reuse
            if reuse:
                scope.reuse_variables()
                train = False
            network = self.conv3d(frames, [4, 4, 4, 3, 64], [1, 2, 2, 2, 1], 'd_cv3_1', group=self.D_weight) #32*32*16
            network = tf.nn.relu(self.batchnorm(network, train, 'd_bn_8', group=self.D_weight))
            network = self.conv3d(network, [4, 4, 4, 64, 128], [1, 2, 2, 2, 1], 'd_cv3_2', group=self.D_weight) #16*16*8
            network = tf.nn.relu(self.batchnorm(network, train, 'd_bn_9', group=self.D_weight))
            network = self.conv3d(network, [4, 4, 4, 128, 256], [1, 2, 2, 2, 1], 'd_cv3_3', group=self.D_weight) #8*8*4
            network = tf.nn.relu(self.batchnorm(network, train, 'd_bn_10', group=self.D_weight))
            network = self.conv3d(network, [4, 4, 4, 256, 512], [1, 2, 2, 2, 1], 'd_cv3_4', group=self.D_weight) #4*4*2
            network = tf.nn.relu(self.batchnorm(network, train, 'd_bn_11', group=self.D_weight))
            network = self.conv3d(network, [2, 4, 4, 512, 1], [1, 2, 4, 4, 1], 'd_cv3_5', group=self.D_weight) #1*1*1
        return network

    def conv2d(self, incoming_layer, filter_shape,
            strides, name, group, act=tf.identity,
            padding='SAME'):
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=filter_shape,
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable(name='b', shape=(filter_shape[-1]),
                initializer=tf.constant_initializer(value=0.0))
        return act(tf.nn.conv2d(incoming_layer, W, strides=strides, padding=padding) + b)


    def deconv3d(self, incoming_layer, filter_shape, output_shape,
            strides, name,  group, act=tf.identity,
            padding='SAME'):
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=filter_shape,
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable(name='b', shape=(filter_shape[-2]),
                initializer=tf.constant_initializer(value=0.0))
        rtn = tf.nn.conv3d_transpose(incoming_layer, W, output_shape=output_shape, strides=strides, padding=padding)
        rtn.set_shape([None] + output_shape[1:])
        return act(tf.nn.bias_add(rtn, b))


    def conv3d(self, incoming_layer, filter_shape,
            strides, name, group, act=tf.identity,
            padding='SAME'):
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=filter_shape,
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            b = tf.get_variable(name='b', shape=(filter_shape[-1]),
                initializer=tf.constant_initializer(value=0.0))
        return act(tf.nn.conv3d(incoming_layer, W, strides=strides, padding=padding, name=None) + b)


    def batchnorm(self, incoming_layer, phase, name, group):
        dummy = self.dummyLayer(incoming_layer)
        network = BatchNormLayer(layer = dummy,
                                is_train = phase,
                                name = name)
        return network.outputs


    class dummyLayer:
        #a class designed to fool tensorlayer...
        def __init__(self, output):
            self.outputs = output
            self.all_layers = []
            self.all_params = []
            self.all_drop = []


class BatchNormLayer():
    def __init__(
        self,
        layer = None,
        decay = 0.9,
        epsilon = 0.00001,
        act = tf.identity,
        is_train = False,
        beta_init = tf.constant_initializer(value=0.0),
        gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
        name ='g_bn_layer',
    ):
        self.inputs = layer.outputs
        # print("  [TL] BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" %
        #                     (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            # if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
            #     beta_init = beta_init()
            beta = tf.get_variable('beta', shape=params_shape,
                               initializer=beta_init,
                               trainable=is_train)#, restore=restore)

            gamma = tf.get_variable('gamma', shape=params_shape,
                                initializer=gamma_init, trainable=is_train,
                                )#restore=restore)

            ## 2.
            moving_mean_init = tf.constant_initializer(0.0)
            moving_mean = tf.get_variable('moving_mean',
                                      params_shape,
                                      initializer=moving_mean_init,
                                      trainable=False,)#   restore=restore)
            moving_variance = tf.get_variable('moving_variance',
                                          params_shape,
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False,)#   restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:    # TF12
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay, zero_debias=False)     # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay, zero_debias=False) # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay)
                # print("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
                self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon) )
            else:
                self.outputs = act( tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon) )

            variables = [beta, gamma, moving_mean, moving_variance]

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( variables )


if __name__ == "__main__":
    '''
    python wgan.py 10000
    '''
    if not os.path.exists('./output/'):
        os.makedirs('output/')
    parser = argparse.ArgumentParser()
    parser.add_argument('num_epochs', type=int)
    args = parser.parse_args()
    test = WGAN(None, None, None, None, frame_count=16)
    test.train(epoch_num=args.num_epochs)

