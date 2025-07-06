#!/usr/bin/env python3
"""
Fixed setup script for MINE Disentangle VAE datasets
Improved download handling for MPI3D and CelebA
"""

import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import urllib.request
import subprocess


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


def download_with_curl(url, filepath, description="Downloading"):
    """Download using curl (more reliable for some URLs)"""
    try:
        print(f"{description} using curl...")
        result = subprocess.run([
            'curl', '-L', '-o', str(filepath), url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Downloaded successfully to {filepath}")
            return True
        else:
            print(f"❌ Curl failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Curl error: {e}")
        return False


def setup_mpi3d_fixed(data_dir):
    """Setup MPI3D dataset with improved download"""
    print("\n=== Setting up MPI3D dataset (Fixed) ===")
    
    # Create directory
    dataset_dir = Path(data_dir) / "mpi3d"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    extract_dir = dataset_dir / "mpi3d_realistic"
    
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"MPI3D dataset already exists at {extract_dir}")
        return True
    
    # Try multiple download URLs for MPI3D
    urls = [
        "https://www.dropbox.com/s/6q1z9v2v3q2q3g7/mpi3d_realistic.zip?dl=1",
        "https://storage.googleapis.com/disentanglement_dataset/data/mpi3d_realistic.zip",
        "https://github.com/rr-learning/disentanglement_dataset/raw/master/data/mpi3d_realistic.zip"
    ]
    
    zip_path = dataset_dir / "mpi3d_realistic.zip"
    
    success = False
    for i, url in enumerate(urls):
        print(f"Trying MPI3D download URL {i+1}/{len(urls)}: {url}")
        
        # Try curl first (more reliable for Dropbox)
        if download_with_curl(url, zip_path, f"Downloading MPI3D (attempt {i+1})"):
            success = True
            break
        
        # Fallback to requests
        if download_file(url, zip_path, f"Downloading MPI3D (attempt {i+1})"):
            success = True
            break
    
    if not success:
        print("❌ All MPI3D download attempts failed")
        print("\nManual download instructions:")
        print("1. Visit: https://www.dropbox.com/s/6q1z9v2v3q2q3g7/mpi3d_realistic.zip?dl=1")
        print("2. Download the file manually")
        print(f"3. Place it in: {zip_path}")
        print("4. Run this script again")
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


def setup_celeba_fixed(data_dir):
    """Setup CelebA dataset with improved download"""
    print("\n=== Setting up CelebA dataset (Fixed) ===")
    
    # Create directory
    dataset_dir = Path(data_dir) / "celeba"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    img_dir = dataset_dir / "img_align_celeba"
    
    if img_dir.exists() and any(img_dir.iterdir()):
        print(f"CelebA dataset already exists at {img_dir}")
        return True
    
    # Try multiple download methods for CelebA
    zip_path = dataset_dir / "img_align_celeba.zip"
    
    print("CelebA dataset requires manual download due to Google Drive restrictions.")
    print("\nManual download instructions:")
    print("1. Visit: https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing")
    print("2. Click 'Download' (you may need to sign in to Google)")
    print(f"3. Place the downloaded file in: {zip_path}")
    print("4. Run this script again to extract")
    
    # Check if user has already downloaded the file
    if zip_path.exists():
        print(f"\nFound existing zip file: {zip_path}")
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
    
    return False


def create_dummy_mpi3d_data(data_dir):
    """Create dummy MPI3D data for testing"""
    print("\n=== Creating dummy MPI3D data for testing ===")
    
    dataset_dir = Path(data_dir) / "mpi3d"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    extract_dir = dataset_dir / "mpi3d_realistic"
    extract_dir.mkdir(exist_ok=True)
    
    # Create dummy .npz files
    import numpy as np
    
    for i in range(3):  # Create 3 dummy files
        filepath = extract_dir / f"dummy_mpi3d_{i}.npz"
        
        # Create dummy data
        images = np.random.randint(0, 256, (100, 64, 64, 3), dtype=np.uint8)
        labels = np.random.rand(100, 7).astype(np.float32)
        
        np.savez(filepath, images=images, labels=labels)
    
    print(f"✅ Created dummy MPI3D data: {len(list(extract_dir.glob('*.npz')))} files")
    return True


def create_dummy_celeba_data(data_dir):
    """Create dummy CelebA data for testing"""
    print("\n=== Creating dummy CelebA data for testing ===")
    
    dataset_dir = Path(data_dir) / "celeba"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    img_dir = dataset_dir / "img_align_celeba"
    img_dir.mkdir(exist_ok=True)
    
    # Create dummy images using PIL
    try:
        from PIL import Image
        import numpy as np
        
        for i in range(100):  # Create 100 dummy images
            # Create random image
            img_array = np.random.randint(0, 256, (218, 178, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Save with proper filename format
            filename = f"{i+1:06d}.jpg"
            filepath = img_dir / filename
            img.save(filepath, 'JPEG')
        
        print(f"✅ Created dummy CelebA data: {len(list(img_dir.glob('*.jpg')))} images")
        return True
        
    except ImportError:
        print("❌ PIL not available. Install with: pip install Pillow")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fixed setup for MINE Disentangle VAE datasets")
    parser.add_argument("--dataset", choices=["3dshapes", "dsprites", "mpi3d", "celeba", "all"], 
                       default="all", help="Dataset to setup")
    parser.add_argument("--data_dir", default="Data", help="Data directory path")
    parser.add_argument("--verify", action="store_true", help="Verify existing datasets")
    parser.add_argument("--create_dummy", action="store_true", help="Create dummy data for testing")
    
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
    
    if args.create_dummy:
        if args.dataset == "all" or args.dataset == "mpi3d":
            create_dummy_mpi3d_data(data_dir)
        if args.dataset == "all" or args.dataset == "celeba":
            create_dummy_celeba_data(data_dir)
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
            results[dataset] = setup_mpi3d_fixed(data_dir)
        elif dataset == "celeba":
            results[dataset] = setup_celeba_fixed(data_dir)
    
    # Summary
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset:12} : {status}")
    
    print("\nNext steps:")
    print("1. For failed datasets, follow the manual download instructions above")
    print("2. Or create dummy data for testing: python setup_datasets_fixed.py --create_dummy")
    print("3. Update your config file to use the desired dataset")
    print("4. Run training with: python run_enhanced.py --config configs/mine_disentangle_vae.yaml")
    print("5. For 3DShapes (recommended): set dataset_name: '3dshapes' in config")


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


if __name__ == "__main__":
    main() 