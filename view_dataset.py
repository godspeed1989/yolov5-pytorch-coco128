import torch
import torch.nn as nn
import yaml
import argparse
import numpy as np
import cv2

from utils.general import check_img_size, plot_images
from utils.datasets import create_dataloader

from train import hyp

def view_data(opt):
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    gs = 32
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    rank = opt.local_rank

    dataloader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, gs, opt, hyp=hyp, augment=True,
                                            cache=opt.cache_images, rect=opt.rect, local_rank=rank,
                                            world_size=opt.world_size)
    nd = len(dataset)
    nb = len(dataloader)  # number of batches
    print('TRAIN Num of images {}, num of batches {}'.format(nd, nb))
    #
    testloader, testset = create_dataloader(test_path, imgsz_test, opt.total_batch_size, gs, opt, hyp=hyp, augment=False,
                                    cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size)
    nd = len(testset)
    nb = len(testloader)  # number of batches
    print('TEST Num of images {}, num of batches {}'.format(nd, nb))
    print('Using %g dataloader workers' % dataloader.num_workers)

    for name, loader in {'train': dataloader, 'test': testloader}.items():
        for i, (imgs, targets, paths, _) in enumerate(loader):
            imgs = imgs.float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            mosaic = plot_images(images=imgs, targets=targets, paths=paths, fname=None, max_subplots=1)
            cv2.imshow('{}'.format(name), mosaic)
            cv2.waitKey(0)
            if i == 3: break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--batch-size', type=int, default=2, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[512, 512], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    cfg = parser.parse_args()
    cfg.total_batch_size = cfg.batch_size
    cfg.world_size = 1

    view_data(cfg)
