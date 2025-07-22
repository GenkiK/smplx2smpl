# smplx2smpl
Convert SMPL-X to SMPL using optimization with temporal constraints. This repository provides an example implementation using the [ParaHome](https://github.com/snuvclab/ParaHome) dataset.

## Description
The `smplx` library provides basic functionality to convert SMPL-X models to SMPL, but it does not ensure temporal smoothness. While the LBFGS optimizer generally yields better results compared to optimizers like Adam, it can occasionally return NaNs.  
This code addresses these issues by incorporating temporal continuity, interpolating NaN values, and producing smoother motion sequences.

## Setup
### 1. Clone this repository
```
git clone git@github.com:GenkiK/smplx2smpl.git
cd smplx2smpl
```

### 2. Build environment with Singularity (Apptainer)
We use SingularityCE 3.10.4 to build the environment. Please install Apptainer/Singularity beforehand by following the official documentation.
To build and run the environment:
```
singularity build  --fakeroot smplx2smpl.sif smplx2smpl.def
singularity run --nv smplx2smpl.sif
```

Alternatively, you might manually set up a Python virtual environment by referring to smplx2smpl.def and installing the necessary Python packages.
Note that you might need to adjust either the CUDA version or the package versions to ensure compatibility.

### 3. Download required files
1. SMPL-X models

Download the SMPL-X body models from [the SMPL-X website](https://smpl-x.is.tue.mpg.de/). (Navigate to the Download page and click "Download SMPL-X v1.1 (NPZ+PKL, 830MB)".)
Place the downloaded files in `models/smplx`.

2. SMPL models
Download the SMPL body models from [the SMPL website](https://smpl.is.tue.mpg.de/). (Navigate to the Download page and click "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)".)
Place the files in `models/smpl` and rename them to:
    - SMPL_NEUTRAL.pkl
    - SMPL_FEMALE.pkl
    - SMPL_MALE.pkl

3. Transfer data
Download `transfer_data/smplx2smpl_deftrafo_setup.pkl` by following [this section in the `smplx` README](https://github.com/vchoutas/smplx/tree/main/transfer_model#data).

The resulting directory structure should look like this:
```
smplx2smpl/
├─ models/
│  ├─ smpl/
│  │  ├─ SMPL_FEMALE.pkl
│  │  ├─ SMPL_MALE.pkl
│  │  ├─ SMPL_NEUTRAL.pkl
│  ├─ smplx/
│  │  ├─ SMPLX_FEMALE.npz
│  │  ├─ SMPLX_FEMALE.pkl
│  │  ├─ SMPLX_MALE.npz
│  │  ├─ SMPLX_MALE.pkl
│  │  ├─ SMPLX_NEUTRAL.npz
│  │  ├─ SMPLX_NEUTRAL.pkl
├─ transfer_data/
    ├─ smplx2smpl_deftrafo_setup.pkl
```

## Run
This implementation has been tested on the [Parahome](https://github.com/snuvclab/ParaHome) dataset. To use this code with another dataset, please create a new dataset class by referring to parahome.py.

We provide example shell scripts for running each step of the pipeline. **Make sure to update the file paths in the `.env` file before running the scripts**.

1. Per-frame fitting
```
bash /path/to/smplx2smpl/scripts/per_frame.sh
```

2. Interpolating NaNs
```
bash /path/to/smplx2smpl/scripts/interp_nan.sh
```

3. Smoothing fitted data to remove outliers
```
bash /path/to/smplx2smpl/scripts/smoothen.sh
```

4. Per-window fitting
```
bash /path/to/smplx2smpl/scripts/temporal_fit.sh
```

5. (optional) Visualization
```
bash /path/to/smplx2smpl/scripts/visualize.sh
```

## Notes
In per-window fitting, there are rare cases where the fitting of the first frame fails, causing all subsequent fittings to fail. Since PyTorch's LBFGS seems to include randomness, solving this issue is challenging. We strongly recommend visualizing the fitting process to check if it is successful, and if it fails, retrying the process.