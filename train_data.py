from collections import defaultdict

from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets

from augmentation import get_transforms

USE_TRAIN_SUBSET_ONLY=True
SAMPLES_PER_CLASS = 120

def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,

):
    assert USE_TRAIN_SUBSET_ONLY, "USE_TRAIN_SUBSET_ONLY must be True"
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=USE_TRAIN_SUBSET_ONLY, # True
        download=True,
        transform=get_transforms(train=True),
    )
    class_to_indices = defaultdict(list)
    for idx, target in enumerate(train_dataset.targets):
        if len(class_to_indices[target]) < SAMPLES_PER_CLASS:
            class_to_indices[target].append(idx)

    balanced_indices = [
        idx
        for target in sorted(class_to_indices)
        for idx in class_to_indices[target]
    ]
    train_dataset = Subset(train_dataset, balanced_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train
    )

    return train_dataset, train_loader
