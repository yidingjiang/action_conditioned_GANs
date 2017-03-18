import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import vid_processing


def lrelu(x):
    return tf.maximum(x, 0.1 * x)


def build_generator(inputs):
    with tf.variable_scope('g'):
        out = slim.conv2d(
            inputs,
            64, [5, 5],
            normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            stride=2,
            scope='conv1',
            padding='SAME')
        out = slim.conv2d(
            out,
            128, [5, 5],
            normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            stride=2,
            scope='conv2',
            padding='SAME')
        out = slim.conv2d_transpose(
            out,
            128, [5, 5],
            stride=2,
            normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            scope='tconv1')
        out = slim.conv2d_transpose(
            out,
            128, [5, 5],
            normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            scope='tconv2')
        out = slim.conv2d_transpose(
            out,
            64, [5, 5],
            normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            stride=2,
            scope='tconv3')
        out = slim.conv2d_transpose(
            out,
            64, [5, 5],
            normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            scope='tconv4')
        out = slim.conv2d_transpose(
            out,
            3, [5, 5],
            normalizer_fn=slim.batch_norm,
            activation_fn=tf.tanh,
            scope='tconv5')
    return out


def build_discriminator(inputs, reuse=False):
    with tf.variable_scope('d', reuse=reuse):
        out = slim.conv2d(
            inputs,
            64, [5, 5],
            #normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            scope='conv1',
            stride=2,
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d(
            out,
            128, [5, 5],
            #normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            scope='conv2',
            stride=2,
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d(
            out,
            256, [5, 5],
            #normalizer_fn=slim.batch_norm,
            activation_fn=lrelu,
            scope='conv3',
            stride=2,
            padding='SAME',
            reuse=reuse)
        out = slim.conv2d(
            out,
            512, [5, 5],
            activation_fn=lrelu,
            scope='conv4',
            stride=2,
            padding='SAME',
            reuse=reuse)
        out = slim.flatten(out)
        out = slim.fully_connected(
            out, 1,
            #normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.sigmoid,
            scope='fc1',
            reuse=reuse)
        out = tf.squeeze(out)
    return out


def main():
    img_ph = tf.placeholder(tf.float32, [None, 64, 64, 3], name='current_frame')
    next_frame_ph = tf.placeholder(tf.float32, [None, 64, 64, 3], name='next_frame')
    next_frame_label_ph = tf.placeholder(tf.float32, [None], name='next_frame_label')
    g_out = build_generator(img_ph)
    d_out_gen = build_discriminator(g_out, reuse=True)
    d_out_direct = build_discriminator(next_frame_ph, reuse=True)
    desired_d_output = tf.placeholder(tf.float32, [None], name='desired_d_output')
    d_cost = tf.losses.mean_squared_error(d_out_direct, next_frame_label_ph)
    g_cost = tf.losses.mean_squared_error(d_out_gen, desired_d_output)
    g_cost_pretrain = tf.losses.mean_squared_error(g_out, next_frame_ph)
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
    g_pretrain_op = tf.train.RMSPropOptimizer(0.001, name='g_pretrain').minimize(g_cost_pretrain, var_list=g_vars)
    g_opt_op = tf.train.RMSPropOptimizer(0.001, name='g_opt').minimize(g_cost, var_list=g_vars)
    d_opt_op = tf.train.RMSPropOptimizer(0.001, name='d_opt').minimize(d_cost, var_list=d_vars)
    # writer = tf.summary.FileWriter('/tmp/gan_logs', graph=tf.get_default_graph())
    data = (vid_processing.read_data('./data/1000vids.npy') / 127.) - 1.
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(10000):
            print('Iteration {:d}'.format(i))
            #idx = np.random.randint(0, 8)
            idx = 6
            batch_size = 20
            batch_indices = np.random.choice(data.shape[0], batch_size, replace=True)
            batch = data[batch_indices]
            frames = batch[:,idx,:,:,:]
            next_frames = batch[:,idx+1,:,:,:]
            delta_frames = next_frames - frames
            if i < 1000:
                _, pretrain_cost = sess.run([g_pretrain_op, g_cost_pretrain], {
                    img_ph: frames,
                    next_frame_ph: delta_frames
                })
                print('Pretrain cost', pretrain_cost)
                continue

            generated_next_frames = sess.run(g_out, feed_dict={
                img_ph: frames
            })

            if i % 100 == 0:
                vid_processing.save_samples(frames, frames + generated_next_frames, i)

            if i % 10 == 0:
                _, d_res2 = sess.run([d_opt_op, d_cost], feed_dict={
                    next_frame_ph: generated_next_frames,
                    next_frame_label_ph: np.random.uniform(0., 0.1, batch_size)
                })
                _, d_res1 = sess.run([d_opt_op, d_cost], feed_dict={
                    next_frame_ph: delta_frames,
                    next_frame_label_ph: np.random.uniform(0.9, 1.0, batch_size)
                })
                print('Discriminator results:', d_res1, d_res2)

            if i % 1 == 0:
                _, g_res = sess.run([g_opt_op, g_cost], feed_dict={
                    img_ph: frames,
                    desired_d_output: [1.] * batch_size
                })
                print('Generator results:', g_res)


if __name__ == '__main__':
    main()
