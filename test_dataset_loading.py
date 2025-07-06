#!/usr/bin/env python3
"""
Test script to verify dataset loading functionality
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import h5py


def create_dummy_3dshapes_data(filepath, num_samples=1000):
    """Create dummy 3DShapes data for testing"""
    print(f"Creating dummy 3DShapes data at {filepath}")
    
    # Create dummy images and labels
    images = np.random.randint(0, 256, (num_samples, 64, 64, 3), dtype=np.uint8)
    labels = np.random.rand(num_samples, 6).astype(np.float32)
    
    # Save to HDF5 file
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('labels', data=labels)
    
    print(f"Created dummy 3DShapes data: {images.shape}, labels: {labels.shape}")


def create_dummy_dsprites_data(filepath, num_samples=1000):
    """Create dummy dSprites data for testing"""
    print(f"Creating dummy dSprites data at {filepath}")
    
    # Create dummy images and labels
    images = np.random.randint(0, 256, (num_samples, 64, 64), dtype=np.uint8)
    latents_values = np.random.rand(num_samples, 6).astype(np.float32)
    latents_classes = np.random.randint(0, 10, (num_samples, 6), dtype=np.int64)
    
    # Save to NPZ file
    np.savez(filepath, 
             imgs=images, 
             latents_values=latents_values, 
             latents_classes=latents_classes)
    
    print(f"Created dummy dSprites data: {images.shape}, latents: {latents_values.shape}")


def test_dataset_imports():
    """Test that dataset classes can be imported"""
    print("Testing dataset imports...")
    
    try:
        from dataset import ThreeDShapesDataset, DSpritesDataset, MPI3DDataset, VAEDataset
        print("✅ All dataset classes imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_3dshapes_dataset():
    """Test 3DShapes dataset loading"""
    print("\nTesting 3DShapes dataset...")
    
    try:
        from dataset import ThreeDShapesDataset
        
        # Create temporary directory and dummy data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "3dshapes"
            data_dir.mkdir()
            
            filepath = data_dir / "3dshapes.h5"
            create_dummy_3dshapes_data(filepath, num_samples=100)
            
            # Test dataset loading
            dataset = ThreeDShapesDataset(
                data_path=str(temp_path),
                split='train',
                transform=None
            )
            
            print(f"✅ 3DShapes dataset loaded: {len(dataset)} samples")
            
            # Test getting an item
            image, label = dataset[0]
            print(f"   Image shape: {image.shape}, Label shape: {label.shape}")
            
            return True
            
    except Exception as e:
        print(f"❌ 3DShapes dataset test failed: {e}")
        return False


def test_dsprites_dataset():
    """Test dSprites dataset loading"""
    print("\nTesting dSprites dataset...")
    
    try:
        from dataset import DSpritesDataset
        
        # Create temporary directory and dummy data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "dsprites"
            data_dir.mkdir()
            
            filepath = data_dir / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
            create_dummy_dsprites_data(filepath, num_samples=100)
            
            # Test dataset loading
            dataset = DSpritesDataset(
                data_path=str(temp_path),
                split='train',
                transform=None
            )
            
            print(f"✅ dSprites dataset loaded: {len(dataset)} samples")
            
            # Test getting an item
            image, label = dataset[0]
            print(f"   Image shape: {image.shape}, Label shape: {label.shape}")
            
            return True
            
    except Exception as e:
        print(f"❌ dSprites dataset test failed: {e}")
        return False


def test_vae_dataset_module():
    """Test VAEDataset module with different datasets"""
    print("\nTesting VAEDataset module...")
    
    try:
        from dataset import VAEDataset
        
        # Test with 3DShapes
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy 3DShapes data
            data_dir = temp_path / "3dshapes"
            data_dir.mkdir()
            filepath = data_dir / "3dshapes.h5"
            create_dummy_3dshapes_data(filepath, num_samples=100)
            
            # Test VAEDataset module
            data_module = VAEDataset(
                data_path=str(temp_path),
                dataset_name="3dshapes",
                train_batch_size=8,
                val_batch_size=8,
                patch_size=(64, 64)
            )
            
            # Setup the module
            data_module.setup()
            
            print(f"✅ VAEDataset module setup successful")
            print(f"   Train dataset: {len(data_module.train_dataset)} samples")
            print(f"   Val dataset: {len(data_module.val_dataset)} samples")
            
            # Test dataloader
            train_loader = data_module.train_dataloader()
            batch = next(iter(train_loader))
            print(f"   Batch shape: {batch[0].shape}, Labels: {batch[1].shape}")
            
            return True
            
    except Exception as e:
        print(f"❌ VAEDataset module test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*50)
    print("DATASET LOADING TESTS")
    print("="*50)
    
    tests = [
        ("Import Test", test_dataset_imports),
        ("3DShapes Dataset", test_3dshapes_dataset),
        ("dSprites Dataset", test_dsprites_dataset),
        ("VAEDataset Module", test_vae_dataset_module),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour dataset loading code is working correctly.")
        print("You can now proceed with downloading the actual datasets:")
        print("  python setup_datasets.py --dataset 3dshapes")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check the error messages above and fix any issues.")
    
    print("\nNext steps:")
    print("1. Download datasets: python setup_datasets.py")
    print("2. Update config file to use desired dataset")
    print("3. Run training: python run_enhanced.py --config configs/mine_disentangle_vae.yaml")


if __name__ == "__main__":
    main() 