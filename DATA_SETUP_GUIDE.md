# Data Setup Guide for MINE Disentangle VAE

This guide provides comprehensive instructions for setting up all datasets required for the MINE Disentangle VAE project.

## Quick Start

For automatic setup, use the provided script:

```bash
# Install required packages
pip install requests tqdm

# Setup all datasets automatically
python setup_datasets.py

# Or setup specific datasets
python setup_datasets.py --dataset 3dshapes
python setup_datasets.py --dataset dsprites
python setup_datasets.py --dataset mpi3d
python setup_datasets.py --dataset celeba

# Verify existing datasets
python setup_datasets.py --verify
```

## Manual Setup Instructions

### 1. 3DShapes Dataset (Recommended for Paper)

**Download Link:** https://storage.googleapis.com/3d-shapes/3dshapes.h5

**Setup Steps:**
1. Create directory: `mkdir -p Data/3dshapes`
2. Download the file: `wget https://storage.googleapis.com/3d-shapes/3dshapes.h5 -O Data/3dshapes/3dshapes.h5`
3. Verify: File should be ~2.8 GB

**Dataset Info:**
- **Size:** ~2.8 GB
- **Images:** 480,000 (64×64×3)
- **Factors:** 6 (floor hue, wall hue, object hue, object size, object shape, camera elevation)
- **Factor Values:** 10, 10, 10, 8, 4, 15 respectively

**Expected Structure:**
```
Data/
└── 3dshapes/
    └── 3dshapes.h5
```

### 2. dSprites Dataset

**Download Link:** https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true

**Setup Steps:**
1. Create directory: `mkdir -p Data/dsprites`
2. Download the file: `wget "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true" -O Data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz`
3. Verify: File should be ~26 MB

**Dataset Info:**
- **Size:** ~26 MB
- **Images:** 737,280 (64×64 grayscale)
- **Factors:** 6 (color, shape, scale, orientation, position X, position Y)
- **Factor Values:** 1, 3, 6, 40, 32, 32 respectively

**Expected Structure:**
```
Data/
└── dsprites/
    └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

### 3. MPI3D Dataset

**Download Link:** https://www.dropbox.com/s/6q1z9v2v3q2q3g7/mpi3d_realistic.zip?dl=1

**Setup Steps:**
1. Create directory: `mkdir -p Data/mpi3d`
2. Download and extract:
   ```bash
   wget "https://www.dropbox.com/s/6q1z9v2v3q2q3g7/mpi3d_realistic.zip?dl=1" -O Data/mpi3d/mpi3d_realistic.zip
   cd Data/mpi3d
   unzip mpi3d_realistic.zip
   rm mpi3d_realistic.zip
   ```
3. Verify: Directory should contain multiple .npz files

**Dataset Info:**
- **Size:** ~1.1 GB (compressed)
- **Images:** Multiple .npz files with realistic 3D objects
- **Factors:** 7 (object color, object shape, object size, camera height, background color, horizontal axis, vertical axis)
- **Factor Values:** 4, 4, 2, 3, 3, 40, 40 respectively

**Expected Structure:**
```
Data/
└── mpi3d/
    └── mpi3d_realistic/
        ├── file1.npz
        ├── file2.npz
        └── ...
```

### 4. CelebA Dataset

**Download Link:** https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing

**Setup Steps:**
1. Create directory: `mkdir -p Data/celeba`
2. Download and extract:
   ```bash
   # Download from Google Drive (may require manual download)
   # Place img_align_celeba.zip in Data/celeba/
   cd Data/celeba
   unzip img_align_celeba.zip
   rm img_align_celeba.zip
   ```
3. Verify: Directory should contain ~202,599 .jpg files

**Dataset Info:**
- **Size:** ~1.4 GB (compressed)
- **Images:** ~202,599 (178×218×3)
- **Factors:** 40 binary attributes
- **Note:** This dataset is used for face generation, not disentanglement evaluation

**Expected Structure:**
```
Data/
└── celeba/
    └── img_align_celeba/
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
```

## Complete Directory Structure

After setup, your Data folder should look like this:

```
Data/
├── 3dshapes/
│   └── 3dshapes.h5                    # ~2.8 GB
├── dsprites/
│   └── dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz  # ~26 MB
├── mpi3d/
│   └── mpi3d_realistic/
│       ├── file1.npz
│       ├── file2.npz
│       └── ...                        # Multiple .npz files
└── celeba/
    └── img_align_celeba/
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...                        # ~202,599 .jpg files
```

## Configuration

Update your config file to use the desired dataset:

```yaml
# configs/mine_disentangle_vae.yaml
data_params:
  data_path: "Data/"
  dataset_name: "3dshapes"  # Options: "3dshapes", "dsprites", "mpi3d", "celeba"
  train_batch_size: 64
  val_batch_size: 64
  patch_size: 64
  num_workers: 4
```

## Dataset Usage in Code

The updated `dataset.py` supports all datasets:

```python
from dataset import VAEDataset

# For 3DShapes (recommended for paper)
data_module = VAEDataset(
    data_path="Data/",
    dataset_name="3dshapes",
    train_batch_size=64,
    val_batch_size=64,
    patch_size=(64, 64)
)

# For dSprites
data_module = VAEDataset(
    data_path="Data/",
    dataset_name="dsprites",
    train_batch_size=64,
    val_batch_size=64,
    patch_size=(64, 64)
)

# For MPI3D
data_module = VAEDataset(
    data_path="Data/",
    dataset_name="mpi3d",
    train_batch_size=64,
    val_batch_size=64,
    patch_size=(64, 64)
)

# For CelebA
data_module = VAEDataset(
    data_path="Data/",
    dataset_name="celeba",
    train_batch_size=64,
    val_batch_size=64,
    patch_size=(64, 64)
)
```

## Verification

Verify your datasets are properly set up:

```bash
# Check all datasets
python setup_datasets.py --verify

# Check specific dataset
python setup_datasets.py --verify --dataset 3dshapes
```

## Troubleshooting

### Common Issues

1. **Download fails for large files:**
   - Use the setup script which includes progress bars and error handling
   - For Google Drive links, you may need to download manually

2. **Insufficient disk space:**
   - 3DShapes: ~2.8 GB
   - dSprites: ~26 MB
   - MPI3D: ~1.1 GB
   - CelebA: ~1.4 GB
   - Total: ~5.3 GB

3. **Permission errors:**
   - Ensure you have write permissions to the Data directory
   - Use `chmod` if needed

4. **Import errors:**
   - Install required packages: `pip install h5py numpy torch torchvision pytorch-lightning`

### Manual Download Links

If automatic download fails, use these direct links:

- **3DShapes:** https://storage.googleapis.com/3d-shapes/3dshapes.h5
- **dSprites:** https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true
- **MPI3D:** https://www.dropbox.com/s/6q1z9v2v3q2q3g7/mpi3d_realistic.zip?dl=1
- **CelebA:** https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing

## Next Steps

After setting up the datasets:

1. **Update your config file** to use the desired dataset
2. **Run training:** `python run_enhanced.py --config configs/mine_disentangle_vae.yaml`
3. **For paper reproduction:** Use 3DShapes dataset as it's the primary dataset in the paper
4. **For evaluation:** The model will automatically compute disentanglement metrics (Z-diff, Z-min Var, IRS, MIG, JEMMIG)

## Dataset Statistics

| Dataset | Size | Images | Factors | Primary Use |
|---------|------|--------|---------|-------------|
| 3DShapes | 2.8 GB | 480K | 6 | **Paper Primary** |
| dSprites | 26 MB | 737K | 6 | Disentanglement |
| MPI3D | 1.1 GB | Multiple | 7 | Disentanglement |
| CelebA | 1.4 GB | 202K | 40 | Face Generation |

**Recommendation:** Start with 3DShapes dataset as it's the primary dataset used in the MINE Disentangle VAE paper and provides the best baseline for reproducing the results. 