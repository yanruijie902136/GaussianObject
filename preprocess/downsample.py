import os
import argparse
from PIL import Image

from progress_bar import *


class Downsample:
    def __init__(self, wizard=None, progress_bar=None):
        self.wizard = wizard
        self.progress_bar = progress_bar

    def main(self, source_path):
        images_path = os.path.join(source_path, 'images')

        image_names = sorted(os.listdir(images_path))
        factors = (2, 4, 8)

        if self.progress_bar is not None:
            set_progress_bar(self.wizard, self.progress_bar, len(factors) * len(image_names))

        for factor in factors:
            images_path_resize = f'{images_path}_{factor}'
            if not os.path.exists(images_path_resize):
                os.mkdir(images_path_resize)

            for image_name in image_names:
                image = Image.open(os.path.join(images_path, image_name))
                orig_w, orig_h = image.size[0], image.size[1]
                resolution = round(orig_w / factor), round(orig_h / factor)
                image = image.resize(resolution)
                image.save(os.path.join(images_path_resize, image_name))

                progress_bar_step()

        clear_progress_bar()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source-path', type=str, default='data/realcap/rabbit')

    args = parser.parse_args()
    Downsample().main(args.source_path)
