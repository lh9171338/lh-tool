import os
import cv2
import glob
import tqdm
import argparse


def images2video(image_file_list, video_file, fourcc, fps, frameSize=None):
    assert len(image_file_list), 'There is no image'
    if frameSize is None:
        image = cv2.imread(image_file_list[0])
        frameSize = (image.shape[1], image.shape[0])
    frameSize = tuple(frameSize)
    videoWriter = cv2.VideoWriter(video_file, fourcc, fps, frameSize)
    assert videoWriter.isOpened(), f'Failed to create file: {video_file}'

    for image_file in tqdm.tqdm(image_file_list, desc=video_file):
        image = cv2.imread(image_file)
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
    opts = parser.parse_args()
    print(opts)

    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    fps = opts.fps
    size = opts.size
    image_file_list = sorted(glob.glob(os.path.join(opts.input, f'*.{opts.postfix}')))
    if len(image_file_list) > 0:
        video_file = os.path.abspath(opts.input) + '.mp4' if opts.output is None else opts.output
        images2video(image_file_list, video_file, fourcc, fps, size)
    else:
        image_path_list = sorted(glob.glob(os.path.join(opts.input, '*/')))
        for image_path in image_path_list:
            video_file = os.path.abspath(image_path) + '.mp4'
            image_file_list = sorted(glob.glob(os.path.join(image_path, f'*.{opts.postfix}')))
            images2video(image_file_list, video_file, fourcc, fps, size)


if __name__ == '__main__':
    main()
