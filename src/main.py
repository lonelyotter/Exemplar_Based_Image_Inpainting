import argparse
from skimage.io import imread, imsave
from inpainter import Inpainter


def main():
    args = parse_args()

    image = imread(args.input_image)
    mask = imread(args.mask)

    # remove channels of the mask
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    output_image = Inpainter(image,
                             mask,
                             patch_size=args.patch_size,
                             plot_progress=args.plot_progress).inpaint()
    imsave(args.output, output_image)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image',
                        help='the image containing objects to be removed')
    parser.add_argument('mask',
                        help='the mask image of the region to be removed')
    parser.add_argument('-ps',
                        '--patch-size',
                        help='the size of the patches',
                        type=int,
                        default=9)
    parser.add_argument('-o',
                        '--output',
                        help='the file path to save the output image',
                        default='output.jpg')
    parser.add_argument('--plot-progress',
                        help='plot each generated image',
                        action='store_true',
                        default=False)
    return parser.parse_args()


if __name__ == '__main__':
    main()
