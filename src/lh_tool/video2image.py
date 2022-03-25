import os
import cv2
import glob
import tqdm
import argparse


def video2images(video_file, image_path, postfix, frameSize=None):
    os.makedirs(image_path, exist_ok=True)
    videoCapture = cv2.VideoCapture(video_file)
    assert videoCapture.isOpened(), f'Failed to open file: {video_file}'

    frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in tqdm.trange(frame_count, desc=video_file):
        status, image = videoCapture.read()
        if not status:
            break
        if frameSize is not None:
            image = cv2.resize(image, tuple(frameSize))
        image_file = os.path.join(image_path, f'{index + 1:06d}.{postfix}')
        cv2.imwrite(image_file, image)

    videoCapture.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='.', help='video file or path of video files')
    parser.add_argument('-o', '--output', type=str, help='path of image files')
    parser.add_argument('-p', '--postfix', type=str, default='png', help='postfix of image filename')
    parser.add_argument('-s', '--size', type=int, nargs=2, help='desired frame size for image')
    opts = parser.parse_args()
    print(opts)

    postfix = opts.postfix
    size = opts.size
    if os.path.isdir(opts.input):
        video_file_list = sorted(glob.glob(os.path.join(opts.input, '*.mp4')))
        for video_file in video_file_list:
            image_path = os.path.splitext(video_file)[0]
            video2images(video_file, image_path, postfix, size)

    elif os.path.isfile(opts.input):
        video_file = opts.input
        image_path = os.path.splitext(video_file)[0] if opts.output is None else opts.output
        video2images(video_file, image_path, postfix, size)


if __name__ == '__main__':
    main()
