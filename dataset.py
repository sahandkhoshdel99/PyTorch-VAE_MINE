import os
import torch
import numpy as np
import h5py
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable, Tuple
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


class ThreeDShapesDataset(Dataset):
    """
    3DShapes dataset loader
    
    Dataset: https://storage.googleapis.com/3d-shapes/3dshapes.h5
    Place in: Data/3dshapes/3dshapes.h5
    
    Factors:
    - floor hue (10 values)
    - wall hue (10 values) 
    - object hue (10 values)
    - object size (8 values)
    - object shape (4 values)
    - camera elevation (15 values)
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 train_ratio: float = 0.8):
        
        self.data_path = Path(data_path) / "3dshapes" / "3dshapes.h5"
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"3DShapes dataset not found at {self.data_path}")
        
        # Load data
        with h5py.File(self.data_path, 'r') as f:
            self.images = f['images'][:]  # (480000, 64, 64, 3)
            self.labels = f['labels'][:]  # (480000, 6)
        
        # Convert to torch tensors
        self.images = torch.from_numpy(self.images).float() / 255.0  # Normalize to [0, 1]
        self.images = self.images.permute(0, 3, 1, 2)  # (N, C, H, W)
        self.labels = torch.from_numpy(self.labels).float()
        
        # Split data
        num_samples = len(self.images)
        train_size = int(num_samples * self.train_ratio)
        
        if split == 'train':
            self.images = self.images[:train_size]
            self.labels = self.labels[:train_size]
        else:  # val/test
            self.images = self.images[train_size:]
            self.labels = self.labels[train_size:]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DSpritesDataset(Dataset):
    """
    dSprites dataset loader
    
    Dataset: https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true
    Place in: Data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
    
    Factors:
    - color (1 value)
    - shape (3 values)
    - scale (6 values)
    - orientation (40 values)
    - position X (32 values)
    - position Y (32 values)
    """
    
    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 train_ratio: float = 0.8):
        
        self.data_path = Path(data_path) / "dsprites" / "dsprites.npz"
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"dSprites dataset not found at {self.data_path}")
        
        # Load data
        data = np.load(self.data_path)
        self.images = data['imgs']  # (737280, 64, 64)
        self.latents_values = data['latents_values']  # (737280, 6)
        self.latents_classes = data['latents_classes']  # (737280, 6)
        
        # Convert to torch tensors
        self.images = torch.from_numpy(self.images).float() / 255.0  # Normalize to [0, 1]
        self.images = self.images.unsqueeze(1)  # Add channel dimension (N, 1, H, W)
        self.labels = torch.from_numpy(self.latents_values).float()
        
        # Split data
        num_samples = len(self.images)
        train_size = int(num_samples * self.train_ratio)
        
        if split == 'train':
            self.images = self.images[:train_size]
            self.labels = self.labels[:train_size]
        else:  # val/test
            self.images = self.images[train_size:]
            self.labels = self.labels[train_size:]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MPI3DDataset(Dataset):
    """
    MPI3D dataset loader
    
    Dataset: https://www.dropbox.com/s/6q1z9v2v3q2q3g7/mpi3d_realistic.zip?dl=1
    Place in: Data/mpi3d/mpi3d_realistic/
    
    Factors:
    - object color (4 values)
    - object shape (4 values)
    - object size (2 values)
    - camera height (3 values)
    - background color (3 values)
    - horizontal axis (40 values)
    - vertical axis (40 values)
    """
    
    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 train_ratio: float = 0.8):
        
        self.data_path = Path(data_path) / "mpi3d" / "mpi3d_realistic"
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"MPI3D dataset not found at {self.data_path}")
        
        # Load data - MPI3D is stored as multiple .npz files
        image_files = sorted(list(self.data_path.glob("*.npz")))
        
        if not image_files:
            raise FileNotFoundError(f"No .npz files found in {self.data_path}")
        
        # Load all data
        all_images = []
        all_labels = []
        
        for file_path in image_files:
            data = np.load(file_path)
            if 'images' in data and 'labels' in data:
                all_images.append(data['images'])
                all_labels.append(data['labels'])
        
        if not all_images:
            raise ValueError("No valid data found in MPI3D files")
        
        # Concatenate all data
        self.images = np.concatenate(all_images, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        
        # Convert to torch tensors
        self.images = torch.from_numpy(self.images).float() / 255.0  # Normalize to [0, 1]
        if len(self.images.shape) == 3:
            self.images = self.images.unsqueeze(1)  # Add channel dimension if grayscale
        self.labels = torch.from_numpy(self.labels).float()
        
        # Split data
        num_samples = len(self.images)
        train_size = int(num_samples * self.train_ratio)
        
        if split == 'train':
            self.images = self.images[:train_size]
            self.labels = self.labels[:train_size]
        else:  # val/test
            self.images = self.images[train_size:]
            self.labels = self.labels[train_size:]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    Place in: Data/celeba/img_align_celeba/
    """
    
    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy data to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module supporting multiple datasets
    
    Supported datasets:
    - 3DShapes
    - dSprites  
    - MPI3D
    - CelebA
    - OxfordPets

    Args:
        data_path: root directory of your dataset.
        dataset_name: name of the dataset to use ('3dshapes', 'dsprites', 'mpi3d', 'celeba', 'oxford_pets')
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str = '3dshapes',
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (64, 64),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.dataset_name = dataset_name.lower()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        
        if self.dataset_name == '3dshapes':
            self._setup_3dshapes()
        elif self.dataset_name == 'dsprites':
            self._setup_dsprites()
        elif self.dataset_name == 'mpi3d':
            self._setup_mpi3d()
        elif self.dataset_name == 'celeba':
            self._setup_celeba()
        elif self.dataset_name == 'oxford_pets':
            self._setup_oxford_pets()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _setup_3dshapes(self):
        """Setup 3DShapes dataset"""
        transform = transforms.Compose([
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ])
        
        self.train_dataset = ThreeDShapesDataset(
            self.data_path,
            split='train',
            transform=transform,
        )
        
        self.val_dataset = ThreeDShapesDataset(
            self.data_path,
            split='test',
            transform=transform,
        )
    
    def _setup_dsprites(self):
        """Setup dSprites dataset"""
        # dSprites already returns tensors, so we only need to resize
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float()),  # Ensure float type
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),  # Add channel dim if needed
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert to RGB if grayscale
        ])
        
        self.train_dataset = DSpritesDataset(
            self.data_path,
            split='train',
            transform=transform,
        )
        
        self.val_dataset = DSpritesDataset(
            self.data_path,
            split='test',
            transform=transform,
        )
    
    def _setup_mpi3d(self):
        """Setup MPI3D dataset"""
        transform = transforms.Compose([
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ])
        
        self.train_dataset = MPI3DDataset(
            self.data_path,
            split='train',
            transform=transform,
        )
        
        self.val_dataset = MPI3DDataset(
            self.data_path,
            split='test',
            transform=transform,
        )
    
    def _setup_celeba(self):
        """Setup CelebA dataset"""
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ])
        
        val_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ])
        
        self.train_dataset = MyCelebA(
            self.data_path,
            split='train',
            transform=train_transforms,
            download=False,
        )
        
        self.val_dataset = MyCelebA(
            self.data_path,
            split='test',
            transform=val_transforms,
            download=False,
        )
    
    def _setup_oxford_pets(self):
        """Setup OxfordPets dataset"""
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        val_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.train_dataset = OxfordPets(
            self.data_path,
            split='train',
            transform=train_transforms,
        )
        
        self.val_dataset = OxfordPets(
            self.data_path,
            split='val',
            transform=val_transforms,
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     