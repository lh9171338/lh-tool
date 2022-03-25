import os
import glob
import imageio
import argparse


def image2gif(image_file_list, gif_file, duration):
    assert len(image_file_list), 'There is no image'
    images = [imageio.imread(image_file) for image_file in image_file_list]
    duration /= len(image_file_list) - 1
    imageio.mimsave(gif_file, images, 'GIF', duration=duration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='.', help='path of image files')
    parser.add_argument('-o', '--output', type=str, help='output gif file')
    parser.add_argument('-p', '--postfix', type=str, default='png', help='postfix of image filename')
    parser.add_argument('-d', '--duration', type=float, default=1, help='duration')
    opts = parser.parse_args()

    opts.input = os.path.abspath(opts.input)
    opts.output = opts.input + '.gif' if opts.output is None else opts.output
    print(opts)

    image_path = opts.input
    gif_file = opts.output
    postfix = opts.postfix
    duration = opts.duration
    image_file_list = sorted(glob.glob(os.path.join(image_path, '*.' + postfix)))
    image2gif(image_file_list, gif_file, duration)


if __name__ == '__main__':
    main()
