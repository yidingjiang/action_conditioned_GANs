import numpy as np
import cv2
import argparse
import os
import pickle


def get_frames(fname, num_frames=10, frame_freq=10):
    '''
    Returns num_frames frames, taking every frame_freq frame.

    Usage:
    >>> frame_data = get_frames('./videos/example_video.mp4')
    '''
    cap = cv2.VideoCapture(fname)
    frames = []
    i = 0
    while cap.isOpened() and i < num_frames and i % num_frames == 0:
        ret, frame = cap.read()
        frames.append(frame)
        i += 1
    cap.release()
    return np.array(frames)


def read_data(fname):
    '''
    Reads in pickled data.
    '''
    return pickle.load(open(fname, 'rb'))


if __name__ == '__main__':
    '''
    Usage from terminal: ./vid_processing.py ./videos ./output_data
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    result = {}
    for fname in os.listdir(args.videos_path):
        if fname.split('.')[-1] == 'mp4':
            result[fname] = get_frames(os.path.join(
                args.videos_path, fname))

    with open(args.output_path, 'wb') as f:
        pickle.dump(result, f)
