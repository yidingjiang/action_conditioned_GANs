import numpy as np
import cv2
import argparse
import os
import h5py
import glob


IMG_WIDTH = 100
IMG_HEIGHT = 100
NUM_FRAMES = 51


def get_frames(fname):
    cap = cv2.VideoCapture(fname)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frames.append(frame)
    cap.release()
    return np.concatenate(frames, axis=2)


def convert(input_folder, save_path):
    video_fnames = glob.glob(os.path.join(input_folder, '*.mp4'))
    num_videos = len(video_fnames)
    f = h5py.File(save_path, 'w')
    video_dset_shape = (num_videos, IMG_WIDTH, IMG_HEIGHT, 3 * NUM_FRAMES)
    action_dset_shape = (num_videos, 50, 1, 2)
    vid_dset = f.create_dataset('videos', video_dset_shape, dtype=np.uint8)
    action_dset = f.create_dataset('actions', action_dset_shape, dtype=np.float32)
    for i, fname in enumerate(video_fnames):
        if i % 10 == 0:
            print('Iterations {:d}'.format(i))
        fname = os.path.join(fname)
        actions_fname = os.path.join(input_folder, '{:d}actions.npy'.format(i))
        vid_data = get_frames(fname)
        vid_dset[i] = vid_data
        action_dset[i] = np.load(actions_fname)


if __name__ == '__main__':
    '''
    Usage from terminal: python ./vid_processing.py ./videos ./output_data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    convert(args.videos_path, args.output_path)
