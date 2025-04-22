import os
import cv2
import glob
import tqdm
import argparse
from lh_tool.iterator import SingleProcess, MultiProcess


def images2video(image_path, video_file, postfix, fourcc, fps, frameSize=None, speed=1):
    image_file_list = sorted(glob.glob(os.path.join(image_path, f"*.{postfix}")))
    if len(image_file_list) == 0:
        return
    video_file = os.path.abspath(image_path) + ".mp4" if video_file is None else video_file

    if frameSize is None:
        image = cv2.imread(image_file_list[0])
        image = image[:, :, ::-1]  # RGB2BGR
        frameSize = (image.shape[1], image.shape[0])
    videoWriter = cv2.VideoWriter(video_file, fourcc, fps, frameSize)
    assert videoWriter.isOpened(), f"Failed to create file: {video_file}"

    image_file_list = [image_file_list[i] for i in range(0, len(image_file_list), speed)]
    for image_file in tqdm.tqdm(image_file_list, desc=video_file):
        image = cv2.imread(image_file)
        if frameSize != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, tuple(frameSize))
        videoWriter.write(image)
    videoWriter.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=".", help="path of image files")
    parser.add_argument("-o", "--output", type=str, help="output video file")
    parser.add_argument(
        "-p",
        "--postfix",
        type=str,
        default="png",
        help="postfix of image filename",
    )
    parser.add_argument("-f", "--fps", type=float, default=29.97, help="desired fps for video")
    parser.add_argument("-s", "--size", type=int, nargs=2, help="desired frame size for video")
    parser.add_argument("-a", "--speed", type=int, default=1, help="speed for video")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="convert video to images recursively",
    )
    parser.add_argument("-n", "--nprocs", type=int, default=1, help="number of process")
    opts = parser.parse_args()
    print(opts)

    try:
        image_path = opts.input
        video_file = opts.output
        postfix = opts.postfix
        fps = opts.fps
        size = None if opts.size is None else tuple(opts.size)
        speed = opts.speed
        recursive = opts.recursive
        nprocs = opts.nprocs
        fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
        if recursive:
            image_path_list = glob.glob(os.path.join(opts.input, "*/"))
            if nprocs == 1:
                iterator = SingleProcess(images2video)
            else:
                iterator = MultiProcess(images2video, nprocs=nprocs)
            iterator.run(
                image_path_list,
                video_file=None,
                postfix=postfix,
                fourcc=fourcc,
                fps=fps,
                frameSize=size,
                speed=speed,
            )
        else:
            images2video(image_path, video_file, postfix, fourcc, fps, size, speed)

    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    main()
