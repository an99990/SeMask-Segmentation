# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from pathlib import Path
# from config import settings

def main():
    # parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--palette',
    #     default='cityscapes',
    #     help='Color palette used for segmentation map')
    # parser.add_argument(
    #     '--opacity',
    #     type=float,
    #     default=0.5,
    #     help='Opacity of painted segmentation map. In (0, 1] range.')
    # args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    root_dir = Path(__file__).parents[1]
    model_dir = Path(__file__).parents[4]
    img_dir = Path(__file__).parents[5]
    args = [
        f"{root_dir}/configs/semask_swin/coco_stuff10k/semfpn_semask_swin_tiny_patch4_window7_512x512_80k_coco10k.py",
        f"{img_dir}/images/person_bike.jpg",
           f"{model_dir}/models_weights/semask_tiny_fpn_coco10k.pth",]
           
    model = init_segmentor(args[0], args[-1], device='cuda:0')
    # test a single image
    result = inference_segmentor(model, args[1])
    # show the results
    show_result_pyplot(
        model,
        args[1],
        result,
        get_palette('cityscapes'),
        opacity=0.5)

if __name__ == '__main__':
    main()
