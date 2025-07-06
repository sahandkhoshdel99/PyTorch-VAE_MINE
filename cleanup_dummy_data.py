#!/usr/bin/env python3
"""
Script to clean up dummy data after real datasets are downloaded
"""

import shutil
from pathlib import Path


def cleanup_dummy_data():
    """Remove dummy data files"""
    data_dir = Path("Data")
    
    print("Cleaning up dummy data...")
    
    # Clean up dummy MPI3D files
    mpi3d_dir = data_dir / "mpi3d" / "mpi3d_realistic"
    if mpi3d_dir.exists():
        dummy_files = list(mpi3d_dir.glob("dummy_mpi3d_*.npz"))
        for file in dummy_files:
            file.unlink()
            print(f"Removed: {file}")
    
    # Clean up dummy CelebA files
    celeba_dir = data_dir / "celeba" / "img_align_celeba"
    if celeba_dir.exists():
        dummy_files = list(celeba_dir.glob("*.jpg"))
        for file in dummy_files:
            file.unlink()
            print(f"Removed: {file}")
    
    print("✅ Dummy data cleanup completed!")


def verify_real_data():
    """Verify that real data is present"""
    data_dir = Path("Data")
    
    print("\nVerifying real datasets...")
    
    # Check MPI3D
    mpi3d_dir = data_dir / "mpi3d" / "mpi3d_realistic"
    if mpi3d_dir.exists():
        npz_files = list(mpi3d_dir.glob("*.npz"))
        dummy_files = list(mpi3d_dir.glob("dummy_mpi3d_*.npz"))
        
        if npz_files and not dummy_files:
            print(f"✅ MPI3D: {len(npz_files)} real .npz files found")
        elif dummy_files:
            print(f"⚠️  MPI3D: {len(dummy_files)} dummy files still present")
        else:
            print("❌ MPI3D: No files found")
    
    # Check CelebA
    celeba_dir = data_dir / "celeba" / "img_align_celeba"
    if celeba_dir.exists():
        jpg_files = list(celeba_dir.glob("*.jpg"))
        if len(jpg_files) > 1000:  # Real CelebA has ~202K images
            print(f"✅ CelebA: {len(jpg_files)} real images found")
        elif jpg_files:
            print(f"⚠️  CelebA: {len(jpg_files)} files found (may be dummy)")
        else:
            print("❌ CelebA: No images found")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_dummy_data()
    else:
        verify_real_data()
        print("\nTo clean up dummy data, run: python cleanup_dummy_data.py cleanup") 