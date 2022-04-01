import os
import glob
import imageio
import argparse
from lh_tool.Iterator import SingleProcess, MultiProcess


def image2gif(image_path, gif_file, postfix, duration):
    image_file_list = glob.glob(os.path.join(image_path, f'*.{postfix}'))
    if len(image_file_list) == 0:
        return
    gif_file = os.path.abspath(image_path) + '.gif' if gif_file is None else gif_file

    images = [imageio.imread(image_file) for image_file in image_file_list]
    duration /= len(image_file_list) - 1
    imageio.mimsave(gif_file, images, 'GIF', duration=duration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='.', help='path of image files')
    parser.add_argument('-o', '--output', type=str, help='output gif file')
    parser.add_argument('-p', '--postfix', type=str, default='png', help='postfix of image filename')
    parser.add_argument('-d', '--duration', type=float, default=1, help='duration')
    parser.add_argument('-r', '--recursive', action='store_true', help='convert images to gif recursively')
    parser.add_argument('-n', '--nprocs', type=int, default=1, help='number of process')
    opts = parser.parse_args()
    print(opts)

    try:
        image_path = opts.input
        gif_file = opts.output
        postfix = opts.postfix
        duration = opts.duration
        recursive = opts.recursive
        nprocs = opts.nprocs
        if recursive:
            image_path_list = glob.glob(os.path.join(opts.input, '*/'))
            if nprocs == 1:
                iterator = SingleProcess(image2gif)
            else:
                iterator = MultiProcess(image2gif, nprocs=nprocs)
            iterator.run(image_path_list, None, postfix, duration)
        else:
            image2gif(image_path, gif_file, postfix, duration)

    except AssertionError as e:
        print(e)


if __name__ == '__main__':
    main()
