#!/usr/bin/env python3
"""
Setup script for MINE Disentangle VAE datasets

This script helps download and organize all required datasets:
- 3DShapes
- dSprites
- MPI3D
- CelebA

Usage:
    python setup_datasets.py [--dataset DATASET_NAME] [--data_dir DATA_DIR]
"""

import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import urllib.request


def download_file(url, filepath, description="Downloading"):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def setup_3dshapes(data_dir):
    """Setup 3DShapes dataset"""
    print("\n=== Setting up 3DShapes dataset ===")
    
    # Create directory
    dataset_dir = Path(data_dir) / "3dshapes"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = dataset_dir / "3dshapes.h5"
    
    if filepath.exists():
        print(f"3DShapes dataset already exists at {filepath}")
        return True
    
    # Download URL
    url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
    
    print(f"Downloading 3DShapes dataset from {url}")
    print(f"Saving to {filepath}")
    
    success = download_file(url, filepath, "Downloading 3DShapes")
    
    if success:
        print(f"✅ 3DShapes dataset downloaded successfully!")
        print(f"   File size: {filepath.stat().st_size / (1024**3):.2f} GB")
        return True
    else:
        print("❌ Failed to download 3DShapes dataset")
        return False


def setup_dsprites(data_dir):
    """Setup dSprites dataset"""
    print("\n=== Setting up dSprites dataset ===")
    
    # Create directory
    dataset_dir = Path(data_dir) / "dsprites"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = dataset_dir / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    
    if filepath.exists():
        print(f"dSprites dataset already exists at {filepath}")
        return True
    
    # Download URL
    url = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"
    
    print(f"Downloading dSprites dataset from {url}")
    print(f"Saving to {filepath}")
    
    success = download_file(url, filepath, "Downloading dSprites")
    
    if success:
        print(f"✅ dSprites dataset downloaded successfully!")
        print(f"   File size: {filepath.stat().st_size / (1024**2):.2f} MB")
        return True
    else:
        print("❌ Failed to download dSprites dataset")
        return False


def setup_mpi3d(data_dir):
    """Setup MPI3D dataset"""
    print("\n=== Setting up MPI3D dataset ===")
    
    # Create directory
    dataset_dir = Path(data_dir) / "mpi3d"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    extract_dir = dataset_dir / "mpi3d_realistic"
    
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"MPI3D dataset already exists at {extract_dir}")
        return True
    
    # Download URL
    url = "https://www.dropbox.com/s/6q1z9v2v3q2q3g7/mpi3d_realistic.zip?dl=1"
    zip_path = dataset_dir / "mpi3d_realistic.zip"
    
    print(f"Downloading MPI3D dataset from {url}")
    print(f"Saving to {zip_path}")
    
    success = download_file(url, zip_path, "Downloading MPI3D")
    
    if not success:
        print("❌ Failed to download MPI3D dataset")
        return False
    
    # Extract the zip file
    print("Extracting MPI3D dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        print(f"✅ MPI3D dataset extracted successfully!")
        print(f"   Extracted to: {extract_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to extract MPI3D dataset: {e}")
        return False


def setup_celeba(data_dir):
    """Setup CelebA dataset"""
    print("\n=== Setting up CelebA dataset ===")
    
    # Create directory
    dataset_dir = Path(data_dir) / "celeba"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    img_dir = dataset_dir / "img_align_celeba"
    
    if img_dir.exists() and any(img_dir.iterdir()):
        print(f"CelebA dataset already exists at {img_dir}")
        return True
    
    # Download URL (Google Drive link)
    url = "https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing"
    zip_path = dataset_dir / "img_align_celeba.zip"
    
    print(f"Downloading CelebA dataset from {url}")
    print(f"Saving to {zip_path}")
    print("Note: This is a large file (~1.4 GB) and may take a while to download...")
    
    # For Google Drive, we need to modify the URL
    file_id = "1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ"
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    success = download_file(download_url, zip_path, "Downloading CelebA")
    
    if not success:
        print("❌ Failed to download CelebA dataset")
        print("You may need to download it manually from the Google Drive link")
        return False
    
    # Extract the zip file
    print("Extracting CelebA dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        print(f"✅ CelebA dataset extracted successfully!")
        print(f"   Extracted to: {img_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to extract CelebA dataset: {e}")
        return False


def verify_dataset(data_dir, dataset_name):
    """Verify that a dataset is properly set up"""
    print(f"\n=== Verifying {dataset_name} dataset ===")
    
    if dataset_name == "3dshapes":
        filepath = Path(data_dir) / "3dshapes" / "3dshapes.h5"
        if filepath.exists():
            size = filepath.stat().st_size / (1024**3)
            print(f"✅ 3DShapes dataset found: {size:.2f} GB")
            return True
        else:
            print("❌ 3DShapes dataset not found")
            return False
    
    elif dataset_name == "dsprites":
        filepath = Path(data_dir) / "dsprites" / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        if filepath.exists():
            size = filepath.stat().st_size / (1024**2)
            print(f"✅ dSprites dataset found: {size:.2f} MB")
            return True
        else:
            print("❌ dSprites dataset not found")
            return False
    
    elif dataset_name == "mpi3d":
        dirpath = Path(data_dir) / "mpi3d" / "mpi3d_realistic"
        if dirpath.exists() and any(dirpath.iterdir()):
            num_files = len(list(dirpath.glob("*.npz")))
            print(f"✅ MPI3D dataset found: {num_files} .npz files")
            return True
        else:
            print("❌ MPI3D dataset not found")
            return False
    
    elif dataset_name == "celeba":
        dirpath = Path(data_dir) / "celeba" / "img_align_celeba"
        if dirpath.exists() and any(dirpath.iterdir()):
            num_files = len(list(dirpath.glob("*.jpg")))
            print(f"✅ CelebA dataset found: {num_files} images")
            return True
        else:
            print("❌ CelebA dataset not found")
            return False
    
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup datasets for MINE Disentangle VAE")
    parser.add_argument("--dataset", choices=["3dshapes", "dsprites", "mpi3d", "celeba", "all"], 
                       default="all", help="Dataset to setup")
    parser.add_argument("--data_dir", default="Data", help="Data directory path")
    parser.add_argument("--verify", action="store_true", help="Verify existing datasets")
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    print(f"Setting up datasets in: {data_dir.absolute()}")
    
    if args.verify:
        if args.dataset == "all":
            datasets = ["3dshapes", "dsprites", "mpi3d", "celeba"]
        else:
            datasets = [args.dataset]
        
        for dataset in datasets:
            verify_dataset(data_dir, dataset)
        return
    
    # Setup datasets
    if args.dataset == "all":
        datasets = ["3dshapes", "dsprites", "mpi3d", "celeba"]
    else:
        datasets = [args.dataset]
    
    results = {}
    
    for dataset in datasets:
        if dataset == "3dshapes":
            results[dataset] = setup_3dshapes(data_dir)
        elif dataset == "dsprites":
            results[dataset] = setup_dsprites(data_dir)
        elif dataset == "mpi3d":
            results[dataset] = setup_mpi3d(data_dir)
        elif dataset == "celeba":
            results[dataset] = setup_celeba(data_dir)
    
    # Summary
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset:12} : {status}")
    
    print("\nNext steps:")
    print("1. Update your config file to use the desired dataset")
    print("2. Run training with: python run_enhanced.py --config configs/mine_disentangle_vae.yaml")
    print("3. For 3DShapes (recommended): set dataset_name: '3dshapes' in config")
    
    print("\nExpected directory structure:")
    print("Data/")
    print("├── 3dshapes/")
    print("│   └── 3dshapes.h5")
    print("├── dsprites/")
    print("│   └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    print("├── mpi3d/")
    print("│   └── mpi3d_realistic/")
    print("│       └── *.npz files")
    print("└── celeba/")
    print("    └── img_align_celeba/")
    print("        └── *.jpg files")


if __name__ == "__main__":
    main() 