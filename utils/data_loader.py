"""
Data loading utilities for MNIST, cardiac, and OASIS datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


class CardiacDataset(Dataset):
    """Dataset for LGE cardiac scar mapping images (128x128, 5 classes)."""

    CLASS_NAMES = [
        'healthy',
        'ischemic_subendo',
        'ischemic_transmural',
        'dcm_mid_wall_line',
        'myocarditis_epicardial'
    ]

    def __init__(self, root: str, transform=None):
        """
        Args:
            root: Root directory containing data/cardiac/ with per-class NPZ files.
            transform: Optional transform to apply.
        """
        self.transform = transform

        data_dir = Path(root) / 'cardiac'

        all_images = []
        all_labels = []

        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            npz_path = data_dir / f'{class_name}.npz'
            if not npz_path.exists():
                raise FileNotFoundError(f"Data file not found: {npz_path}")

            data = np.load(npz_path)
            images = data['images']  # Shape: (N, 128, 128)

            all_images.append(images)
            all_labels.extend([class_idx] * len(images))

        # Concatenate all data
        self.images = np.concatenate(all_images, axis=0)  # (N_total, 128, 128)
        self.labels = np.array(all_labels)

        # Normalize to [0, 1]
        self.images = self.images.astype(np.float32)
        if self.images.max() > 1.0:
            self.images = self.images / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (128, 128)
        label = self.labels[idx]

        # Add channel dimension: (128, 128) -> (1, 128, 128)
        image = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, label


class OASISVentricleDataset(Dataset):
    """Dataset for OASIS 2D brain ventricle slices (128x128)."""

    def __init__(self, root: str, views: List[str] = ['coronal'],
                 exclude_subjects: Optional[List[str]] = None,
                 normalization: str = 'none',
                 label_type: str = 'none'):
        """
        Args:
            root: Root directory containing data/oasis/ with per-view NPZ files.
            views: Which views to include (['axial'], ['coronal'], ['sagittal'], or combination).
            exclude_subjects: Subject IDs to exclude (e.g., ['OAS30001_MR_d0129']).
            normalization: 'none' (default), 'minmax', or 'zscore'.
            label_type: 'none' (unsupervised, returns -1) or 'view' (returns view ID).
        """
        self.normalization = normalization
        self.label_type = label_type
        self.view_names_list = ['axial', 'coronal', 'sagittal']  # Reference order

        # Validate views
        for view in views:
            if view not in self.view_names_list:
                raise ValueError(f"Invalid view: {view}. Must be one of {self.view_names_list}")

        # Load data from view-specific NPZ files
        all_images = []
        all_subject_ids = []
        all_slice_indices = []
        all_view_labels = []

        data_dir = Path(root) / 'oasis'

        for view in views:
            npz_path = data_dir / f'{view}.npz'
            if not npz_path.exists():
                raise FileNotFoundError(f"Data file not found: {npz_path}")

            data = np.load(npz_path, allow_pickle=True)
            images = data['images']  # Shape: (N, 128, 128)
            subject_ids = data['subject_ids']  # Shape: (N,)
            slice_indices = data['slice_indices']  # Shape: (N,)

            # Assign view label
            view_label = self.view_names_list.index(view)

            all_images.append(images)
            all_subject_ids.extend(subject_ids)
            all_slice_indices.extend(slice_indices)
            all_view_labels.extend([view_label] * len(images))

        # Concatenate all data
        self.images = np.concatenate(all_images, axis=0)  # (N_total, 128, 128)
        self.subject_ids = np.array(all_subject_ids, dtype=object)
        self.slice_indices = np.array(all_slice_indices, dtype=np.int32)
        self.view_labels = np.array(all_view_labels, dtype=np.int32)

        # Apply exclusions
        if exclude_subjects is not None and len(exclude_subjects) > 0:
            mask = ~np.isin(self.subject_ids, exclude_subjects)
            self.images = self.images[mask]
            self.subject_ids = self.subject_ids[mask]
            self.slice_indices = self.slice_indices[mask]
            self.view_labels = self.view_labels[mask]

        # Ensure float32 and [0, 1] range
        self.images = self.images.astype(np.float32)
        if self.images.max() > 1.0:
            self.images = self.images / self.images.max()

        print(f"  Loaded OASIS dataset: {len(self.images)} slices from {len(np.unique(self.subject_ids))} subjects")
        print(f"  Views: {views}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image = self.images[idx].copy()  # (128, 128)

        # Convert to torch tensor
        image = torch.from_numpy(image).unsqueeze(0).float()  # (1, 128, 128)

        # Apply normalization
        if self.normalization == 'minmax':
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min + 1e-8)
            image = torch.clamp(image, 0.0, 1.0)
        elif self.normalization == 'zscore':
            img_mean = image.mean()
            img_std = image.std()
            if img_std > 0:
                image = (image - img_mean) / (img_std + 1e-8)
        else:  # 'none': keep as is
            image = torch.clamp(image, 0.0, 1.0)

        # Generate label
        if self.label_type == 'view':
            label = int(self.view_labels[idx])
        elif self.label_type == 'none':
            label = -1  # Unsupervised
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")

        return image, label


def get_mnist_loaders(root: str, batch_size: int, num_workers: int = 0,
                      shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset.

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader


def get_cardiac_loaders(root: str, batch_size: int,
                        num_workers: int = 0, shuffle: bool = True,
                        test_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Load cardiac dataset.

    Args:
        root: Root directory containing data/cardiac/.
        batch_size: Batch size.
        num_workers: Number of workers for data loading.
        shuffle: Whether to shuffle training data.
        test_split: Fraction of data to use for testing.

    Returns:
        train_loader, test_loader
    """
    dataset = CardiacDataset(root=root)

    # Split into train/test
    n_total = len(dataset)
    n_test = int(n_total * test_split)
    n_train = n_total - n_test

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader


def get_oasis_loaders(root: str, views: List[str], exclude_subjects: List[str],
                      normalization: str, label_type: str,
                      batch_size: int, num_workers: int = 0, shuffle: bool = True,
                      test_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Load OASIS ventricle dataset.

    Args:
        root: Root directory containing data/oasis/.
        views: List of views to include (['axial'], ['coronal'], ['sagittal'], or combination).
        exclude_subjects: Subject IDs to exclude.
        normalization: Normalization method.
        label_type: 'none' or 'view'.
        batch_size: Batch size.
        num_workers: Number of workers for data loading.
        shuffle: Whether to shuffle training data.
        test_split: Fraction of subjects to use for testing.

    Returns:
        train_loader, test_loader
    """
    dataset = OASISVentricleDataset(
        root=root, views=views, exclude_subjects=exclude_subjects,
        normalization=normalization, label_type=label_type
    )

    # Subject-level split to prevent data leakage
    unique_subjects = np.unique(dataset.subject_ids)

    # Split subjects
    n_subjects = len(unique_subjects)
    n_test = int(n_subjects * test_split)
    n_train = n_subjects - n_test

    # Shuffle subjects for random split
    np.random.seed(42)
    shuffled_subjects = np.random.permutation(unique_subjects)
    train_subjects = set(shuffled_subjects[:n_train])
    test_subjects = set(shuffled_subjects[n_train:])

    # Create train/test indices
    train_indices = [i for i in range(len(dataset))
                     if dataset.subject_ids[i] in train_subjects]
    test_indices = [i for i in range(len(dataset))
                    if dataset.subject_ids[i] in test_subjects]

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"  Train: {len(train_dataset)} slices from {len(train_subjects)} subjects")
    print(f"  Test: {len(test_dataset)} slices from {len(test_subjects)} subjects")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader


def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Main entry point for loading datasets.

    Supports three datasets:
        - 'mnist': MNIST handwritten digits (28x28, 10 classes)
        - 'cardiac': LGE cardiac scar mapping (128x128, 5 classes)
        - 'oasis': OASIS brain ventricle slices (128x128, unsupervised)

    Args:
        config: Configuration dictionary.

    Returns:
        train_loader, test_loader, dataset_info
    """
    dataset_name = config['dataset']['name']
    root = config['dataset']['root']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    shuffle = config['dataset']['shuffle']

    if dataset_name == 'mnist':
        train_loader, test_loader = get_mnist_loaders(
            root, batch_size, num_workers, shuffle
        )
        dataset_info = {
            'name': 'mnist',
            'num_classes': 10,
            'image_size': (1, 28, 28),
            'class_names': [str(i) for i in range(10)]
        }

    elif dataset_name == 'cardiac':
        train_loader, test_loader = get_cardiac_loaders(
            root, batch_size, num_workers, shuffle
        )
        dataset_info = {
            'name': 'cardiac',
            'num_classes': 5,
            'image_size': (1, 128, 128),
            'class_names': [
                'Healthy',
                'Ischemic (Subendo)',
                'Ischemic (Transmural)',
                'DCM (Mid-wall Line)',
                'Myocarditis (Epicardial)'
            ]
        }

    elif dataset_name == 'oasis':
        views = config['dataset'].get('views', ['coronal'])
        exclude_subjects = config['dataset'].get('exclude_subjects', [])
        normalization = config['dataset'].get('normalization', 'none')
        label_type = config['dataset'].get('label_type', 'none')

        train_loader, test_loader = get_oasis_loaders(
            root, views, exclude_subjects, normalization, label_type,
            batch_size, num_workers, shuffle
        )

        # Determine num_classes
        if label_type == 'view':
            num_classes = len(views)
            class_names = views
        else:  # 'none'
            num_classes = config['model']['num_clusters']
            class_names = [f'Cluster {i}' for i in range(num_classes)]

        dataset_info = {
            'name': 'oasis',
            'num_classes': num_classes,
            'image_size': (1, 128, 128),
            'class_names': class_names
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of: 'mnist', 'cardiac', 'oasis'")

    print(f"Loaded {dataset_info['name']} dataset:")
    print(f"  - Training samples: {len(train_loader.dataset)}")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    print(f"  - Num classes: {dataset_info['num_classes']}")
    print(f"  - Image size: {dataset_info['image_size']}")
    print(f"  - Batch size: {batch_size}")

    return train_loader, test_loader, dataset_info
