import os
import cv2
import glob
import tqdm
import argparse
import time


def play_image(image_file_list, fps):
    period = 1.0 / fps
    for image_file in tqdm.tqdm(image_file_list):
        start_time = time.time()
        image = cv2.imread(image_file)
        cv2.namedWindow("image", 0)
        cv2.imshow("image", image)
        current_time = time.time()
        delta_time = current_time - start_time
        sleep_time = max(1, int((period - delta_time) * 1000))
        cv2.waitKey(sleep_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=".", help="path of image files")
    parser.add_argument(
        "-p",
        "--postfix",
        type=str,
        default="png",
        help="postfix of image filename",
    )
    parser.add_argument("-f", "--fps", type=float, default=29.97, help="desired fps for video")
    opts = parser.parse_args()
    print(opts)

    fps = opts.fps
    image_file_list = sorted(glob.glob(os.path.join(opts.input, f"*.{opts.postfix}")))
    play_image(image_file_list, fps)


if __name__ == "__main__":
    main()
