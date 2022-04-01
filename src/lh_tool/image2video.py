import os
import cv2
import glob
import tqdm
import argparse
from lh_tool.Iterator import SingleProcess, MultiProcess
import lh_tool.imageio as iio


def images2video(image_path, video_file, postfix, fourcc, fps, frameSize=None):
    image_file_list = glob.glob(os.path.join(image_path, f'*.{postfix}'))
    if len(image_file_list) == 0:
        return
    video_file = os.path.abspath(image_path) + '.mp4' if video_file is None else video_file

    if frameSize is None:
        image = iio.imread(image_file_list[0])
        frameSize = (image.shape[1], image.shape[0])
    frameSize = tuple(frameSize)
    videoWriter = cv2.VideoWriter(video_file, fourcc, fps, frameSize)
    assert videoWriter.isOpened(), f'Failed to create file: {video_file}'

    for image_file in tqdm.tqdm(image_file_list, desc=video_file):
        image = iio.imread(image_file)
        if frameSize != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, tuple(frameSize))
        videoWriter.write(image)
    videoWriter.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='.', help='path of image files')
    parser.add_argument('-o', '--output', type=str, help='output video file')
    parser.add_argument('-p', '--postfix', type=str, default='png', help='postfix of image filename')
    parser.add_argument('-f', '--fps', type=float, default=29.97, help='desired fps for video')
    parser.add_argument('-s', '--size', type=int, nargs=2, help='desired frame size for video')
    parser.add_argument('-r', '--recursive', action='store_true', help='convert video to images recursively')
    parser.add_argument('-n', '--nprocs', type=int, default=1, help='number of process')
    opts = parser.parse_args()
    print(opts)

    try:
        image_path = opts.input
        video_file = opts.output
        postfix = opts.postfix
        fps = opts.fps
        size = opts.size
        recursive = opts.recursive
        nprocs = opts.nprocs
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        if recursive:
            image_path_list = glob.glob(os.path.join(opts.input, '*/'))
            if nprocs == 1:
                iterator = SingleProcess(images2video)
            else:
                iterator = MultiProcess(images2video, nprocs=nprocs)
            iterator.run(image_path_list, None, postfix, fourcc, fps, size)
        else:
            images2video(image_path, video_file, postfix, fourcc, fps, size)

    except AssertionError as e:
        print(e)


if __name__ == '__main__':
    main()
