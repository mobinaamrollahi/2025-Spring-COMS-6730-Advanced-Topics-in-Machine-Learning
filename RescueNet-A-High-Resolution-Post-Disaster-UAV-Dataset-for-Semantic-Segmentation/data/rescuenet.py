import numpy as np
from pathlib import Path

from data.base import BaseMMSeg
from data import utils
from config import dataset_dir

RESCUENET_CONFIG_PATH = Path(__file__).parent / "config" / "rescuenet.py"
RESCUENET_CATS_PATH = Path(__file__).parent / "config" / "rescuenet.yml"

class RescueNetDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(image_size, crop_size, split, RESCUENET_CONFIG_PATH, **kwargs)
        self.names, self.colors = utils.dataset_cat_description(RESCUENET_CATS_PATH)
        self.n_cls = 11  # 0 to 10 inclusive
        self.ignore_label = []  # No ignore class based on paper
        self.reduce_zero_label = False  # 0 is background, so we do NOT reduce

    def update_default_config(self, config):
        root_dir = dataset_dir()
        path = Path(root_dir) / "rescuenet"  # Adjust if your dataset folder is differently named
        config.data_root = path
        config.data[self.split]["data_root"] = path
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels
