import os.path as osp
import tempfile

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RescueNetDataset(CustomDataset):

    CLASSES = (
        'unlabeled',
        'water',
        'building-no-damage',
        'building-medium-damage',
        'building-major-damage',
        'building-total-destruction',
        'vehicle',
        'road-clear',
        'road-blocked',
        'tree',
        'pool'
    )

    PALETTE = [
        [0, 0, 0],           # 0: unlabeled / background
        [61, 230, 250],      # 1: water
        [180, 120, 120],     # 2: building-no-damage
        [235, 255, 7],       # 3: building-medium-damage
        [255, 184, 6],       # 4: building-major-damage
        [255, 0, 0],         # 5: building-total-destruction
        [255, 0, 245],       # 6: vehicle
        [140, 140, 140],     # 7: road-clear
        [160, 150, 20],      # 8: road-blocked
        [4, 250, 7],         # 9: tree
        [255, 235, 0],       # 10: pool
    ]

    def __init__(self, **kwargs):
        super(RescueNetDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id):
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            png_filename = osp.join(imgfile_prefix, f'{basename}.png')
            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()
        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset length: '
            f'{len(results)} != {len(self)}')
        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)
        return result_files, tmp_dir
