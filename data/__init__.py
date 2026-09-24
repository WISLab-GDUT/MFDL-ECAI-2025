import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from .datasets import dataset_folder


def get_dataset(opt):
    # ✅ 只传这一层（最关键）
    dataset = dataset_folder(opt, opt.dataroot)

    # 🔍 debug（强烈建议保留）
    try:
        print("classes:", dataset.classes)
        print("class_to_idx:", dataset.class_to_idx)
    except:
        pass

    return dataset


def get_bal_sampler(dataset):
    targets = dataset.targets  # ✅ ImageFolder 自带

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt):
    dataset = get_dataset(opt)

    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(opt.num_threads)
    )

    return data_loader