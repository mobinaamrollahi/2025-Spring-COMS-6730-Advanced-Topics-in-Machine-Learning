import utils.torch as ptu

from data import ImagenetDataset
from data import ADE20KSegmentation
from data import PascalContextDataset
from data import CityscapesDataset
from data import StreetHazardsDataset
from data import RescueNetDataset
from data import Loader


def create_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")
    
    # load dataset_name
    if dataset_name == "imagenet":
        dataset_kwargs.pop("patch_size")
        dataset = ImagenetDataset(split=split, **dataset_kwargs)
    elif dataset_name == "ade20k":
        dataset = ADE20KSegmentation(split=split, **dataset_kwargs)
    elif dataset_name == "pascal_context":
        dataset = PascalContextDataset(split=split, **dataset_kwargs)
    elif dataset_name == "cityscapes":
        dataset = CityscapesDataset(split=split, **dataset_kwargs)
    elif dataset_name == "streethazards":
        dataset = StreetHazardsDataset(split=split, **dataset_kwargs)
    elif dataset_name == "rescuenet":
        dataset = RescueNetDataset(split=split, **dataset_kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")
    
    
    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=ptu.distributed,
        split=split,
    )
    return dataset
