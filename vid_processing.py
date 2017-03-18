import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt


def get_frames(fname, num_frames=16, frame_freq=5):
    '''
    Returns num_frames frames, taking every frame_freq frame.

    Usage:
    >>> frame_data = get_frames('./videos/example_video.mp4')
    '''
    cap = cv2.VideoCapture(fname)
    frames = []
    i = 0
    j = 0
    while cap.isOpened() and i < num_frames:
        if j % frame_freq == 0:
            ret, frame = cap.read()
            if np.allclose(frame, 0.):
                print("ERR")
            resized_frame = cv2.resize(frame, (64, 64))
            frames.append(resized_frame)
            i += 1
        j += 1
    cap.release()
    return np.array(frames)


def read_data(fname):
    '''
    Reads in saved data.
    '''
    return np.load(fname)


def save_samples(input_sample, generated_sample, sample_number):
    for i in range(input_sample.shape[0]):
        save_folder =  './output/sample{:d}/'.format(
            sample_number)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path_input = save_folder + 'batch{:d}-0.png'.format(i)
        plt.imsave(save_path_input, input_sample[i][:,:,::-1])
        save_path_generated = save_folder + 'batch{:d}-1.png'.format(i)
        plt.imsave(save_path_generated, generated_sample[i][:,:,::-1])


if __name__ == '__main__':
    '''
    Usage from terminal: python ./vid_processing.py ./videos ./output_data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    data = []
    for fname in os.listdir(args.videos_path):
        if fname.split('.')[-1] == 'mp4':
            data.append(get_frames(os.path.join(
                args.videos_path, fname)))
    data = np.array(data)

    np.save(args.output_path, data, allow_pickle=False)
