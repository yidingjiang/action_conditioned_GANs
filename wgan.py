import tensorflow as tf
import numpy as np
import os
import pickle
import argparse
from utils import *
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

        self.D_loss = wasserstein_discriminator_loss(self.D_real, self.D_sample)
        self.G_loss = wasserstein_generator_loss(self.D_sample)
        self.D_solver = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(self.D_loss, var_list=self.D_weight))

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
            network = conv2d(input_frame, [4,4,3,64], [1,2,2,1], 'g_cv2_1', group=self.G_weight) #32*32
            network = tf.nn.relu(batchnorm(network, train, 'g_bn_1', group=self.G_weight))
            network = conv2d(network, [4,4,64,128], [1,2,2,1], 'g_cv2_2', group=self.G_weight) #16*16
            network = tf.nn.relu(batchnorm(network, train, 'g_bn_2', group=self.G_weight))
            network = conv2d(network, [4,4,128,256], [1,2,2,1], 'g_cv2_3', group=self.G_weight) #8*8
            network = tf.nn.relu(batchnorm(network, train, 'g_bn_3', group=self.G_weight))
            network = conv2d(network, [4,4,256,512], [1,2,2,1], 'g_cv2_4', group=self.G_weight) #4*4
            network = tf.nn.relu(batchnorm(network, train, 'g_bn_4', group=self.G_weight))
            network = tf.reshape(network, (5,1,4,4,512))
            network = deconv3d(network, [1,4,4,512,512], [5,2,4,4,512], [1,2,1,1,1], 'g_dcv3_1', group=self.G_weight)
            network = tf.nn.relu(batchnorm(network, train, 'g_bn_5', group=self.G_weight))
            network = deconv3d(network, [2,4,4,256,512], [5,4,8,8,256], [1,2,2,2,1], 'g_dcv3_2', group=self.G_weight)
            network = tf.nn.relu(batchnorm(network, train, 'g_bn_6', group=self.G_weight))
            network = deconv3d(network, [4,4,4,128,256], [5,8,16,16,128], [1,2,2,2,1], 'g_dcv3_3', group=self.G_weight)
            network = tf.nn.relu(batchnorm(network, train, 'g_bn_7', group=self.G_weight))
            network = deconv3d(network, [4,4,4,64,128], [5,16,32,32,64], [1,2,2,2,1], 'g_dcv3_4', group=self.G_weight)
            network = tf.nn.relu(batchnorm(network, train, 'g_bn_8', group=self.G_weight))
            depth_step_size = int(float(self.frame_count)/32*2) #use a *full* convolution to adjust the output size
            network = deconv3d(network, [4,4,4,3,64], [5,self.frame_count,64,64,3], [1,depth_step_size,2,2,1], 'g_dcv3_5', group=self.G_weight)
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
            network = conv3d(frames, [4, 4, 4, 3, 64], [1, 2, 2, 2, 1], 'd_cv3_1', group=self.D_weight) #32*32*16
            network = tf.nn.relu(batchnorm(network, train, 'd_bn_8', group=self.D_weight))
            network = conv3d(network, [4, 4, 4, 64, 128], [1, 2, 2, 2, 1], 'd_cv3_2', group=self.D_weight) #16*16*8
            network = tf.nn.relu(batchnorm(network, train, 'd_bn_9', group=self.D_weight))
            network = conv3d(network, [4, 4, 4, 128, 256], [1, 2, 2, 2, 1], 'd_cv3_3', group=self.D_weight) #8*8*4
            network = tf.nn.relu(batchnorm(network, train, 'd_bn_10', group=self.D_weight))
            network = conv3d(network, [4, 4, 4, 256, 512], [1, 2, 2, 2, 1], 'd_cv3_4', group=self.D_weight) #4*4*2
            network = tf.nn.relu(batchnorm(network, train, 'd_bn_11', group=self.D_weight))
            network = conv3d(network, [2, 4, 4, 512, 1], [1, 2, 4, 4, 1], 'd_cv3_5', group=self.D_weight) #1*1*1
        return network

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

