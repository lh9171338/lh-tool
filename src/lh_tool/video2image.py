import os
import cv2
import glob
import tqdm
import argparse
from lh_tool.iterator import SingleProcess, MultiProcess


def video2images(video_file, image_path, postfix, frameSize=None):
    assert os.path.isfile(video_file), f"'{video_file}' is not a file"
    image_path = os.path.splitext(video_file)[0] if image_path is None else image_path
    os.makedirs(image_path, exist_ok=True)

    videoCapture = cv2.VideoCapture(video_file)
    assert videoCapture.isOpened(), f"Failed to open file: '{video_file}'"

    frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in tqdm.trange(frame_count, desc=video_file):
        status, image = videoCapture.read()
        if not status:
            break
        if frameSize is not None:
            image = cv2.resize(image, tuple(frameSize))
        image = image[:, :, ::-1]  # BGR2RGB
        image_file = os.path.join(image_path, f"{index + 1:06d}.{postfix}")
        cv2.imwrite(image_file, image)

    videoCapture.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=".",
        help="input video file or input path",
    )
    parser.add_argument("-o", "--output", type=str, help="output path")
    parser.add_argument(
        "-p",
        "--postfix",
        type=str,
        default="png",
        help="postfix of image filename",
    )
    parser.add_argument("-s", "--size", type=int, nargs=2, help="desired frame size for image")
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
        video_file = opts.input
        image_path = opts.output
        postfix = opts.postfix
        size = None if opts.size is None else tuple(opts.size)
        recursive = opts.recursive
        nprocs = opts.nprocs
        if recursive:
            video_file_list = glob.glob(os.path.join(opts.input, "*.mp4"))
            if nprocs == 1:
                iterator = SingleProcess(video2images)
            else:
                iterator = MultiProcess(video2images, nprocs=nprocs)
            iterator.run(
                video_file_list,
                image_path=None,
                postfix=postfix,
                frameSize=size,
            )
        else:
            video2images(video_file, image_path, postfix, size)

    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    main()
