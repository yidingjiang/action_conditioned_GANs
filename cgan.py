import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import pickle
import argparse
from utils import *
print('Finished imports')

np.random.seed(7)

class CGAN:

    def __init__(self, input_dim=64, gen_frame_num=2):
        self.input_width, self.input_height = input_dim, input_dim
        self.frame_count = gen_frame_num
        self.robot_state, self.robot_action = 5, 5
        self.batch_size = 16
        self.G_weight = []
        self.D_weight = []
        self.build_graph()

    def build_graph(self):
        """
        Build the computational graph
        """
        print('[*] starting graph construction')
        with tf.name_scope('data'):
            self.x = tf.placeholder('float32', (None, self.input_height, self.input_width, 3), name='input_frame')
            self.true_frames = tf.placeholder('float32', (None, self.input_height, self.input_width, 3 * self.frame_count), name='true_frames')

        self.G_train = self.naive_generator(self.x)
        self.D_real = self.discriminator2d(self.true_frames, reuse=False)
        concat_frames = tf.concat([self.x, self.G_train], 3)
        self.D_sample = self.discriminator2d(concat_frames, reuse=True)
        print("[*] all graphs compiled")

        trainable_vars = tf.trainable_variables()
        self.D_weight = [var for var in trainable_vars if 'd_' in var.name]
        self.G_weight = [var for var in trainable_vars if 'g_' in var.name]
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.D_weight]
        difference = concat_frames - self.true_frames
        l2_difference_norm = tf.norm(difference, ord=2, axis=None, keep_dims=False, name='l2_difference')

        self.D_loss = wasserstein_discriminator_loss(self.D_real, self.D_sample)
        self.G_loss = wasserstein_generator_loss(self.D_sample) + 0.5 * l2_difference_norm
        self.D_solver = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(self.D_loss, var_list=self.D_weight))
        self.G_solver = (tf.train.RMSPropOptimizer(learning_rate=5e-5)
                    .minimize(self.G_loss, var_list=self.G_weight))
        print("[*] all solvers compiled")

    def naive_generator(self, input_frame):
        with tf.variable_scope('generator') as scope:
            bn = tf.identity
            hidden1 = slim.layers.conv2d(
                input_frame, 64, [5, 5], stride=2, scope='g_cv2_1', normalizer_fn=slim.batch_norm)
            hidden2 = slim.layers.conv2d(
                hidden1, 128, [4, 4], stride=2, scope='g_cv2_2', normalizer_fn=slim.batch_norm)
            hidden3 = slim.layers.conv2d(
                hidden2, 256, [4, 4], stride=2, scope='g_cv3_3', normalizer_fn=slim.batch_norm)
            hidden4 = slim.layers.conv2d(
                hidden3, 512, [4, 4], stride=2, scope='g_cv2_4', normalizer_fn=slim.batch_norm)
            #condition here?
            hidden5 = slim.layers.conv2d_transpose(
                hidden4, 256, kernel_size=[4,4], stride=2, scope='g_cvt2_1', normalizer_fn=slim.batch_norm)
            # print(hidden5.get_shape())
            hidden6 = slim.layers.conv2d_transpose(
                hidden5, 128, kernel_size=[4,4], stride=2, scope='g_cvt2_2', normalizer_fn=slim.batch_norm)
            # print(hidden6.get_shape())
            hidden7 = slim.layers.conv2d_transpose(
                hidden6, 64, kernel_size=[4,4], stride=2, scope='g_cvt2_3', normalizer_fn=slim.batch_norm)
            # print(hidden7.get_shape())
            hidden8 = slim.layers.conv2d_transpose(
                hidden7, 3, kernel_size=[4,4], stride=2, scope='g_cvt2_4', activation_fn=tf.nn.sigmoid)
            return hidden8

    def discriminator2d(self, frames, reuse):
        with tf.variable_scope('discriminator') as scope:
            is_train = True
            bn = tf.identity
            act = lrelu
            if reuse:
                scope.reuse_variables()
                is_train = False
            hidden1 = slim.layers.conv2d(
                frames, 64, [4, 4], stride=2, scope='d_cv2_1', reuse=reuse, normalizer_fn=slim.batch_norm, activation_fn=act)
            hidden2 = slim.layers.conv2d(
                hidden1, 128, [4, 4], stride=2, scope='d_cv2_2', reuse=reuse, normalizer_fn=slim.batch_norm, activation_fn=act) #4*4 condition here
            hidden3 = slim.layers.conv2d(
                hidden2, 256, [4, 4], stride=2, scope='d_cv2_3', reuse=reuse, normalizer_fn=slim.batch_norm, activation_fn=act) #8*8
            hidden4 = slim.layers.conv2d(
                hidden3, 512, [4, 4], stride=2, scope='d_cv2_4', reuse=reuse, normalizer_fn=slim.batch_norm, activation_fn=act)
            hidden5 = slim.layers.conv2d(
                hidden4, 1, [4, 4], stride=1, scope='d_cv2_5', reuse=reuse, activation_fn=None)
            return hidden5

    def train(self, input_path, output_path, epoch_num=50000, D_per_G=5, batch_size=64):
        sample_dir = os.path.join(output_path, 'output')
        model_dir = os.path.join(output_path, 'models')
        log_dir = os.path.join(output_path, 'logs')

        real_data = np.load(os.path.join(input_path, 'real.npy'))/255.
        real_data_size = real_data.shape[0]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(
                log_dir, sess.graph)
            saver = tf.train.Saver()
            for i in range(epoch_num):
                for _ in range(D_per_G):
                    real = real_data[np.random.randint(real_data_size, size=batch_size)]
                    first_frame_start = np.random.randint(7)
                    first_frame = real[:, first_frame_start,:,:,:]
                    next_frame = real[:, first_frame_start+1,:,:,:]
                    combined_frame = np.concatenate((first_frame, next_frame), axis=3)

                    _, D_loss_curr, _ = sess.run(
                            [self.D_solver, self.D_loss, self.clip_D],
                            feed_dict={self.x: first_frame, self.true_frames: combined_frame}
                            )
                sample_frame_start = np.random.randint(7)
                sample_frame = real_data[np.random.randint(real_data_size, size=self.batch_size)]
                sample_start = sample_frame[:,sample_frame_start,:,:,:]
                sample_next = sample_frame[:,sample_frame_start+1,:,:,:]
                _, G_loss_curr = sess.run(
                            [self.G_solver, self.G_loss],
                            feed_dict={self.x: sample_start, 
                                self.true_frames: np.concatenate((sample_start, sample_next), axis=3)
                                }
                            )

                if i % 100 == 0:
                    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
                          .format(i, D_loss_curr, G_loss_curr))

                if i %  100 == 0:
                    batch = real_data[np.random.randint(real_data_size, size=8)]
                    initial_frame = batch[:,4,:,:,:]
                    gt = batch[:,5,:,:,:]
                    samples = sess.run(self.G_train, feed_dict={self.x: initial_frame})
                    save_samples(sample_dir, initial_frame, samples, gt, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--model_dir', type=str, default='./model')
    args = parser.parse_args()
    output_path = os.path.join(args.output_path, 'output')
    log_dir = os.path.join(args.output_path, 'logs')
    model_dir = os.path.join(args.output_path, 'models')
    os.makedirs(args.output_path)
    os.makedirs(model_dir)
    g = CGAN()
    g.train(args.input_path, args.output_path)







