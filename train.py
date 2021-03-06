import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse
import random

from util import *
from ops import *
from models import *

np.random.seed(7)

HISTORY_LENGTH = 1
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 64
ADAM_LR = 1e-3
NORM_ORDER = 1
L2_WEIGHT = 0.05

PRETRAIN_ITER = 20
TRAIN_ITER = 60000

class Trainer():
    def __init__(self, sess, arg_adv, arg_loss, arg_opt, arg_transform):
        self.sess = sess

        self.img_ph = tf.placeholder(
            tf.float32,
            [BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3 * HISTORY_LENGTH],
            name='current_frame')
        self.next_frame_ph = tf.placeholder(
            tf.float32,
            [BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3],
            name='next_frame')
        self.action_ph = tf.placeholder(
            tf.float32,
            [BATCH_SIZE, 10],
            name='action')
        self.next_state = tf.placeholder(
            tf.float32,
            [BATCH_SIZE,5],
            name='next_state')

        reshaped_actions_raw = tf.reshape(self.action_ph, [tf.shape(self.action_ph)[0], 1, 1, 10])
        reshaped_actions = tf.tile(reshaped_actions_raw, [1, 4, 4, 1])
        reshaped_actions_d = tf.tile(reshaped_actions_raw, [1, 4, 4, 1])

        if arg_transform:
            self.g_out, self.g_state_out = build_generator_transform(self.img_ph, reshaped_actions, 
                                                                     batch_size=BATCH_SIZE, ksize=6)
            self.g_next_frame = self.g_out
            gt_output = self.next_frame_ph
            gt_stateout = self.next_state
        else:
            self.g_out = build_generator(self.img_ph, reshaped_actions)
            self.g_next_frame = self.g_out
            gt_output = self.next_frame_ph

        self.d_out_gen = build_discriminator(
            tf.concat(values=[self.img_ph, self.g_next_frame], axis=3),
            reshaped_actions_d,
            reuse=False)
        self.d_out_real = build_discriminator(
            tf.concat(values=[self.img_ph, self.next_frame_ph], axis=3),
            reshaped_actions_d,
            reuse=True)

        g_psnr = build_psnr(self.next_frame_ph, self.g_next_frame)
        g_l2_loss = tf.norm(self.g_out-gt_output, ord=NORM_ORDER, axis=None, keep_dims=False, name='difference')/BATCH_SIZE

        if arg_transform:
            g_l2_loss *= L2_WEIGHT
            g_state_loss = tf.norm(self.g_state_out - gt_stateout, ord=2, axis=None, keep_dims=False)/BATCH_SIZE
            g_l2_loss += g_state_loss
        if arg_adv:
            g_adv_loss = build_g_adv_loss(self.d_out_gen, arg_loss)
            self.g_loss = g_l2_loss + g_adv_loss + build_gdl(self.next_frame_ph, self.g_next_frame)
        else:
            self.g_loss = g_l2_loss

        self.d_loss = build_d_loss(self.d_out_real, self.d_out_gen, arg_loss)

        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
        self.clip_d = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]

        if arg_opt == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer
            lr = 5e-5
        elif arg_opt = 'adam':
            optimizer = tf.train.AdamOptimizer
            lr = ADAM_LR
        else:
            raise ValueError('unexpected opt argument')

        self.g_opt_op = optimizer(lr, name='g_opt').minimize(self.g_loss, var_list=self.g_vars)
        self.g_pretrain_opt_op = optimizer(lr, name='g_pretrain_opt').minimize(g_l2_loss, var_list=self.g_vars)
        self.d_opt_op = optimizer(lr, name='d_opt').minimize(self.d_loss, var_list=self.d_vars)

        d_loss_summary = tf.summary.scalar('discriminator_loss', self.d_loss)
        g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        g_l2_loss_summary = tf.summary.scalar('g_l2_loss', g_l2_loss)
#        g_state_loss_summary = tf.summary.scalar('g_state_loss', g_state_loss)

        if arg_adv:
            g_adv_loss_summary = tf.summary.scalar('g_adv_loss', g_adv_loss)
        psnr_summary = tf.summary.scalar('g_psnr', g_psnr)
        self.merged_summaries = tf.summary.merge_all()

    def pretrain_g(self, input_images, next_frame, actions, state):
        _, g_res = self.sess.run([self.g_pretrain_opt_op, self.g_loss], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame,
            self.action_ph: actions,
            self.next_state: state
        })
        return g_res

    def train_g(self, input_images, next_frame, actions, state):
        _, gen_next_frames = self.sess.run([self.g_opt_op, self.g_next_frame], feed_dict={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame,
            self.action_ph: actions,
            self.next_state: state
        })
        return gen_next_frames

    def train_d(self, input_images, next_frame, actions, summarize=False):
        fd={
            self.img_ph: input_images,
            self.next_frame_ph: next_frame,
            self.action_ph: actions,
            self.next_state: np.zeros((BATCH_SIZE, 5))
        }
        if summarize:
            _, summ, _ = self.sess.run([self.d_opt_op, self.merged_summaries, self.clip_d], feed_dict=fd)
            return summ
        else:
            self.sess.run([self.d_opt_op, self.clip_d], feed_dict=fd)
            return None

    def test(self, input_images, next_frame, actions):
        tensors = [self.g_next_frame, self.g_state_out, self.merged_summaries]
        fd = {
              self.img_ph: input_images,
              self.next_frame_ph: next_frame,
              self.action_ph: actions,
              self.next_state: np.zeros((BATCH_SIZE, 5))
        }
        gen_next_frames, gen_next_state, summ = self.sess.run(tensors, feed_dict=fd)
        return gen_next_frames, gen_next_state, summ

    def test_sequence(self, input_images, test_next_frame, test_actions):

        predicted = []
        current_frame = input_images[:,0,:,:,:]
        current_state = test_actions[:,0,5:]
        for j in range(0, 6):
            acs = np.concatenate((np.squeeze(test_actions[:,j*2,:5]), current_state), axis=1)
            test_output, test_state, test_summ = self.test(current_frame,
                                                           test_next_frame[:,j*2,:,:,:],
                                                           acs)
            if j==0:
                recorded_summ = test_summ
            predicted.append(test_output)
            current_frame = test_output
            current_state = test_state

        # test_output, test_summ = trainer.test(test_input, test_next_frame, test_actions)
        predicted = np.array(predicted)
        predicted = np.transpose(predicted, (1,0,2,3,4))
        return predicted, current_frame[1:7]


def train(input_path, 
          output_path, 
          test_output_path, 
          log_dir, 
          model_dir, 
          arg_adv, 
          arg_loss, 
          arg_opt, 
          arg_transform):

    img_data_train, action_data_train = build_tfrecord_input(
        BATCH_SIZE,
        input_path,
        1 + HISTORY_LENGTH, .9, True)
    img_data_test, action_data_test = build_tfrecord_input(
        BATCH_SIZE,
        input_path,
        1 + HISTORY_LENGTH, .9, True, training=False)

    next_state_test = tf.squeeze(action_data_test[:,:,5:])
    next_state_train = tf.squeeze(action_data_train[:,:,5:])
    # action_data_train = tf.squeeze(action_data_train[:,0,:])
    # action_data_test = tf.squeeze(action_data_test[:,0,:])
    boolean_mask = build_all_mask(img_data_train.shape[1])
    # print(len(boolean_mask))

    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        trainer = Trainer(sess, arg_adv, arg_loss, arg_opt, arg_transform)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(
            os.path.join(log_dir, 'test'))
        saver = tf.train.Saver()

        if arg_loss == 'wass':
            D_per_G = 5
        else:
            D_per_G = 1

        for i in range(TRAIN_ITER):
            if i < PRETRAIN_ITER:
                input_batch, next_frame_batch, action_batch, state_batch = get_batch(
                    sess,
                    img_data_train,
                    action_data_train,
                    BATCH_SIZE,
                    next_state_train)

                start_mask = boolean_mask[np.random.randint(0,len(boolean_mask),size=BATCH_SIZE)]
                end_mask = np.roll(start_mask, 1, axis=1)

                gen_next_frames = trainer.pretrain_g(input_batch[start_mask],
                                                        next_frame_batch[end_mask],
                                                        action_batch[start_mask],
                                                        state_batch[end_mask])
                print('pre-train iter: '+str(i))
                continue

            for j in range(D_per_G):
                input_batch, next_frame_batch, action_batch, state_batch = get_batch(
                    sess,
                    img_data_train,
                    action_data_train,
                    BATCH_SIZE,
                    next_state_train)

                start_mask = boolean_mask[np.random.randint(0,len(boolean_mask),size=BATCH_SIZE)]
                end_mask = np.roll(start_mask, 1, axis=1)

                make_summ = (i % 100 == 0) and (j==D_per_G-1)
                summ = trainer.train_d(input_batch[start_mask],
                                        next_frame_batch[end_mask],
                                        action_batch[start_mask],
                                        summarize=make_summ)

            start_mask = boolean_mask[np.random.randint(0,len(boolean_mask),size=BATCH_SIZE)]
            end_mask = np.roll(start_mask, 1, axis=1)
            gen_next_frames = trainer.train_g(input_batch[start_mask],
                                                next_frame_batch[end_mask],
                                                action_batch[start_mask],
                                                state_batch[end_mask])

            if i % 100 == 0:
                print('Iteration {:d}'.format(i))
           #     start_mask = boolean_mask[np.random.randint(0,len(boolean_mask),size=BATCH_SIZE)]
           #     end_mask = np.roll(start_mask, 1, axis=1)
                save_samples(output_path,
                    np.expand_dims(input_batch[start_mask][:32], axis=1),
                    np.expand_dims(gen_next_frames[:32], axis=1),
                    np.expand_dims(next_frame_batch[end_mask][:32], axis=1),
                    i)
                saver.save(sess, os.path.join(model_dir, 'model{:d}').format(i))
                writer.add_summary(summ, i)
                writer.flush()

            if i % 500 == 0:
                test_input, test_next_frame, test_actions, state_batch = get_batch(
                    sess,
                    img_data_test,
                    action_data_test,
                    BATCH_SIZE,
                    next_state_train)

                predicted = []
                current_frame = test_input[:,0,:,:,:]
                current_state = state_batch[:,0]

                for j in range(0, test_input.shape[1]-1):
                    acs = np.concatenate((np.squeeze(test_actions[:,j,:5]), current_state), axis=1)
                    test_output, test_state, test_summ = trainer.test(current_frame,
                                                                      test_next_frame[:,j,:,:,:],
                                                                      acs)
                    if j==0:
                        recorded_summ = test_summ
                    predicted.append(test_output)
                    current_frame = test_output
                    current_state = test_state

                # test_output, test_summ = trainer.test(test_input, test_next_frame, test_actions)
                predicted = np.array(predicted)
                predicted = np.transpose(predicted, (1,0,2,3,4))
                save_samples(test_output_path,
                             test_input[:16],
                             predicted[:16],
                             test_next_frame[:16],
                             i)
                test_writer.add_summary(test_summ, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--dna', action='store_true')
    args = parser.parse_args()
    output_path = os.path.join(args.output_path, 'train_output')
    test_output_path = os.path.join(args.output_path, 'test_output')
    model_dir = os.path.join(args.output_path, 'models')
    log_dir = os.path.join(args.output_path, 'logs')
    os.makedirs(args.output_path)
    os.makedirs(model_dir)
    train(args.input_path,
          output_path,
          test_output_path,
          log_dir,
          model_dir,
          args.adv,
          args.loss,
          args.opt,
          args.dna)