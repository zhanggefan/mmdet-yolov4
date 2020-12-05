from multiprocessing import Pool
import cv2
import glob
import os.path as osp
import os
import numpy as np
from shutil import copyfile
from skimage import io


class TencentImageSplitTool(object):
    def __init__(self,
                 in_root,
                 out_root,
                 tile_overlap,
                 tile_shape,
                 num_process=8,
                 ):
        self.in_images_dir = osp.join(in_root, 'img/')
        self.in_labels_dir = osp.join(in_root, 'label/')
        self.out_images_dir = osp.join(out_root, 'img/')
        self.out_labels_dir = osp.join(out_root, 'label/')
        assert isinstance(tile_shape, tuple), f'argument "tile_shape" must be tuple but got {type(tile_shape)} instead!'
        assert isinstance(tile_overlap,
                          tuple), f'argument "tile_overlap" must be tuple but got {type(tile_overlap)} instead!'
        self.tile_overlap = tile_overlap
        self.tile_shape = tile_shape
        images = glob.glob(self.in_images_dir + '*.jpg')
        labels = glob.glob(self.in_labels_dir + '*.circle')
        image_ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], images)]
        label_ids = [*map(lambda x: osp.splitext(osp.split(x)[-1])[0], labels)]
        assert set(image_ids) == set(label_ids)
        self.image_ids = image_ids
        if not osp.isdir(out_root):
            os.mkdir(out_root)
        if not osp.isdir(self.out_images_dir):
            os.mkdir(self.out_images_dir)
        if not osp.isdir(self.out_labels_dir):
            os.mkdir(self.out_labels_dir)
        self.num_process = num_process

    def _parse_annotation_single(self, image_id):
        label_dir = osp.join(self.in_labels_dir, image_id + '.circle')
        with open(label_dir, 'r') as f:
            s = f.readlines()
        objects = []
        for si in s:
            bbox_info = si.split(',')
            assert len(bbox_info) == 8
            bbox = [*map(lambda x: int(float(x)), bbox_info[4:])]
            center = bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]
            objects.append({'bbox': bbox,
                            'labeltxt': bbox_info[1:4],
                            'center': center})
        return objects

    def _split_single(self, image_id):
        objs = self._parse_annotation_single(image_id)
        image_dir = osp.join(self.in_images_dir, image_id + '.jpg')
        try:
            io.imread(image_dir)
        except Exception as e:
            print(e)
            return
        img = cv2.imread(image_dir)
        assert img is not None, image_dir
        h, w, _ = img.shape
        w_ovr, h_ovr = self.tile_overlap
        w_s, h_s = self.tile_shape
        for h_off in range(0, max(1, h - h_ovr), h_s - h_ovr):
            if h_off > 0:
                h_off = min(h - h_s, h_off)  # h_off + hs <= h if h_off > 0
            for w_off in range(0, max(1, w - w_ovr), w_s - w_ovr):
                if w_off > 0:
                    w_off = min(w - w_s, w_off)  # w_off + ws <= w if w_off > 0
                objs_tile = []
                for obj in objs:
                    if w_off <= obj['center'][0] <= w_off + w_s - 1:
                        if h_off <= obj['center'][1] <= h_off + h_s - 1:
                            objs_tile.append(obj)
                if len(objs_tile) > 0:
                    img_tile = img[h_off:h_off + h_s, w_off:w_off + w_s, :]
                    save_image_dir = osp.join(self.out_images_dir, f'{image_id}_{w_off}_{h_off}.jpg')
                    save_label_dir = osp.join(self.out_labels_dir, f'{image_id}_{w_off}_{h_off}.circle')
                    cv2.imwrite(save_image_dir, img_tile)
                    label_tile = []
                    for row, obj in enumerate(objs_tile):
                        obj_tile = obj["bbox"][:]
                        obj_tile[0] -= w_off
                        obj_tile[1] -= h_off
                        bbox_str = ','.join(map(lambda x: str(x), obj_tile))
                        obj_s = f'{row},{",".join(obj["labeltxt"])},{bbox_str}\n'
                        label_tile.append(obj_s)
                    with open(save_label_dir, 'w') as f:
                        f.writelines(label_tile)

    def split(self):
        if self.num_process == 0:
            for i in self.image_ids:
                self._split_single(i)
        else:
            with Pool(self.num_process) as p:
                p.map(self._split_single, self.image_ids)


if __name__ == '__main__':
    ratio = 0.9
    ratio_dgb = 0.001

    origset = 'data/tencent/det/origindata'
    trainset = 'data/tencent/det/train'
    valset = 'data/tencent/det/val'
    trainsetsplit = 'data/tencent/det/trainsplit'
    valsetsplit = 'data/tencent/det/valsplit'

    trainset_dbg = 'data/tencent/det/train_dbg'
    valset_dbg = 'data/tencent/det/val_dbg'
    trainsetsplit_dbg = 'data/tencent/det/trainsplit_dbg'
    valsetsplit_dbg = 'data/tencent/det/valsplit_dbg'


    def makeval(_trainset, _valset, totalratio=None):
        if not (osp.isdir(_trainset) or osp.isdir(_valset)):
            os.mkdir(_trainset)
            os.mkdir(_trainset + '/img')
            os.mkdir(_trainset + '/label')
            os.mkdir(_valset)
            os.mkdir(_valset + '/img')
            os.mkdir(_valset + '/label')

            imgs = glob.glob(origset + '/img/*.jpg')
            imgs = [*map(lambda x: osp.split(x)[-1][:-4], imgs)]
            np.random.shuffle(imgs)

            if totalratio is not None:
                start = int(len(imgs) * ratio * totalratio)
                end = int(len(imgs) * totalratio)
            else:
                start = int(len(imgs) * ratio)
                end = int(len(imgs))

            for i, img in enumerate(imgs):
                if i >= end:
                    return
                dest = _trainset if i < start else _valset

                img_src = f'{origset}/img/{img}.jpg'
                img_des = f'{dest}/img/{img}.jpg'
                ann_src = f'{origset}/label/{img}.circle'
                ann_des = f'{dest}/label/{img}.circle'

                copyfile(img_src, img_des)
                copyfile(ann_src, ann_des)


    makeval(trainset, valset)
    makeval(trainset_dbg, valset_dbg, ratio_dgb)

    trainsplit = TencentImageSplitTool(trainset,
                                       trainsetsplit,
                                       tile_overlap=(160, 160),
                                       tile_shape=(640, 640),
                                       # num_process=0
                                       )
    trainsplit.split()
    valsplit = TencentImageSplitTool(valset,
                                     valsetsplit,
                                     tile_overlap=(160, 160),
                                     tile_shape=(640, 640),
                                     # num_process=0
                                     )
    valsplit.split()

    # debug ------------------
    trainsplit_dbg = TencentImageSplitTool(trainset_dbg,
                                           trainsetsplit_dbg,
                                           tile_overlap=(160, 160),
                                           tile_shape=(640, 640),
                                           # num_process=0
                                           )
    trainsplit_dbg.split()
    valsplit_dbg = TencentImageSplitTool(valset_dbg,
                                         valsetsplit_dbg,
                                         tile_overlap=(160, 160),
                                         tile_shape=(640, 640),
                                         # num_process=0
                                         )
    valsplit_dbg.split()
