# Segmenter: Transformer for Semantic Segmentation (with RescueNet)

[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)

---

## RescueNet Support

This fork supports training and evaluation on the [RescueNet dataset](https://www.nature.com/articles/s41597-023-02799-4), a high-resolution UAV-based semantic segmentation dataset for disaster response.

Folder structure:
```
data/rescuenet/
├── annotations/
│   ├── training/
│   ├── validation/
│   └── test/
├── images/
│   ├── training/
│   ├── validation/
│   └── test/
```

---

## Train and Test with RescueNet

### Training example:
```bash
python main.py --dataset rescuenet --backbone vit_tiny_patch16_384 --decoder mask_transformer --singlescale --save-images --wandb
```

### Testing example:
```bash
python test.py --dataset rescuenet --model_path <log_folder> --singlescale --save-images --combine
```

**Notes:**
- `--model_path` refers to the subfolder under `logs/`, e.g., `rescunet_20250511@013000`
- `--save-images` stores predicted overlays
- `--combine` optionally merges image, ground truth, and prediction into a single image

---

## Setup Instructions for Custom Dataset (e.g., RescueNet)

1. Place your dataset in `data/rescuenet/` with the folder structure shown above.
2. In `data/config/`, create:
   - `rescunet.yml` — contains class labels and color palette
   - `rescunet.py` — defines `dataset_type`, `data_root`, and preprocessing pipelines
3. In `data/`, create `rescunet.py` which defines your `RescueNetDataset` class.
   - Set `img_suffix='.jpg'` and `seg_map_suffix='_lab.jpg'` (or based on your mask filenames).
4. Modify `data/__init__.py` to import `RescueNetDataset`
5. Modify `data/factory.py` to support `"rescunet"` as a valid dataset.
6. In `data/mmseg_config/`, create `rescunet.py` with your `CLASSES`, `PALETTE`, and pipeline config.
7. Add a new `rescunet` entry in `config/config.py` or `config.yml`.

---

## Logs and Weights

Training and evaluation logs will be stored in:

```
logs/rescunet_<timestamp>/
```

This includes:
- `checkpoint.pth` — latest model checkpoint
- `best.pth` — model with best validation performance
- `config.yml` — training configuration
- `log.txt` — per-epoch logs

---

## Weights & Biases

To enable experiment logging:
1. Install wandb:
   ```bash
   pip install wandb
   ```
2. Log in:
   ```bash
   wandb login
   ```
3. Use `--wandb` flag during training.

---

## Citation (Original Segmenter Paper)
```
@article{strudel2021,
  title={Segmenter: Transformer for Semantic Segmentation},
  author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2105.05633},
  year={2021}
}
```

---

## Acknowledgements

The Vision Transformer backbone is adapted from [timm](https://github.com/rwightman/pytorch-image-models), and the training/evaluation framework builds on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

This project is based on the work of [quanghuy0497](https://github.com/quanghuy0497), with modifications to support the RescueNet dataset.
