import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import gfile
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL
import imageio

def build_all_mask(num_frame):
    masks = []
    for i in range(num_frame-1):
        m = [0]*num_frame
        m[i]=1
        masks.append(m)
    return np.array(masks).astype(bool)

def save_samples(output_path, input_sample, generated_sample, gt, sample_number, gif=False):
    input_sample = (255. / 2) * (input_sample + 1.)
    input_sample = input_sample.astype(np.uint8)
    generated_sample = (255. / 2) * (generated_sample + 1.)
    generated_sample = generated_sample.astype(np.uint8)
    gt = (255. / 2) * (gt + 1.)
    gt = gt.astype(np.uint8)
    save_folder =  os.path.join(
        output_path,
        'sample{:d}'.format(sample_number))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i in range(input_sample.shape[0]):
        vid_folder = os.path.join(save_folder, 'vid{:d}'.format(i))
        if not os.path.exists(vid_folder):
            os.makedirs(vid_folder)
        vid = input_sample[i]
        if gif:
            imageio.mimsave(os.path.join(vid_folder, 'gt.gif'), vid, duration=.25)
        else:
            for j in range(int(vid.shape[0])):
                save_path = os.path.join(vid_folder, 'frame{:d}.png'.format(j))
                frame = vid[j]
                plt.imsave(save_path, frame)
        vid = generated_sample[i]
        if gif:
            imageio.mimsave(os.path.join(vid_folder, 'generated.gif'), vid, duration=.25)
        else:
            for j in range(int(vid.shape[0])):
                save_path = os.path.join(vid_folder, 'generated{:d}.png'.format(j))
                frame = vid[j]
                plt.imsave(save_path, frame)
        if not gif:
            vid = gt[i]
            for j in range(int(vid.shape[0])):
                save_path = os.path.join(vid_folder, 'ground_truth{:d}.png'.format(j))
                frame = vid[j]
                plt.imsave(save_path, frame)