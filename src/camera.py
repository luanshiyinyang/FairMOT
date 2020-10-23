import os

import _init_paths
from opts import opts
from tracking_utils.utils import mkdir_if_missing
import datasets.dataset.jde as datasets
from track import eval_seq


def recogniton():
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)
    print("start tracking")
    dataloader = datasets.LoadVideo(0, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else os.path.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=True, frame_rate=frame_rate)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    recogniton()
