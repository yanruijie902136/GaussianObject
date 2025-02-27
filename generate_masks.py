import argparse
import os

import cv2
import numpy as np

from segment_anything import SamPredictor, sam_model_registry

from progress_bar import *


def main(argv=None, conn=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", type=str, required=True)
    parser.add_argument("--sparse_num", type=int, required=True)
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    source_path = args.source_path
    sparse_num = args.sparse_num
    image_path = os.path.join(source_path, 'images')
    sparse_ids = np.loadtxt(os.path.join(source_path, f'sparse_{sparse_num}.txt'), dtype=np.int32)
    image_names = [name for idx, name in enumerate(sorted(os.listdir(image_path))) if idx in sparse_ids]
    images = [cv2.cvtColor(cv2.imread(os.path.join(image_path, image_name)), cv2.COLOR_BGR2RGB)
              for image_name in image_names]

    input_points = [
        [[600, 600]],
        [[600, 600]],
        [[400, 600]],
        [[550, 500]]
    ]

    sam = sam_model_registry["default"](checkpoint="models/sam_vit_h_4b8939.pth")
    sam = sam.cuda()
    predictor = SamPredictor(sam)

    if conn is not None:
        create_progress_bar(conn, len(images))

    multimasks = []
    for image, input_point_list in zip(images, input_points):
        predictor.set_image(image)
        input_point = np.array(input_point_list)
        input_label = np.array([1] * len(input_point_list))
        multimask, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        multimasks.append(multimask)
        progress_bar_step(1)

    chosed_mask_ids = [2, 2, 2, 2]

    masks = [multimasks[i][chosed_mask_ids[i]] for i in range(sparse_num)]
    mask_path = os.path.join(source_path, 'masks')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    for image_name, mask in zip(image_names, masks):
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_path, image_name), mask)

    remove_progress_bar()


if __name__ == "__main__":
    main()
