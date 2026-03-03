# Data Setup

## MNIST
MNIST is auto-downloaded to `data/MNIST/` on first run via torchvision.

## Cardiac LGE Synthetic (5-class)
128x128 grayscale cardiac MRI images. 5 classes:
- healthy
- ischemic_subendo (subendocardial ischemia)
- ischemic_transmural (transmural ischemia)
- dcm_mid_wall_line (dilated cardiomyopathy)
- myocarditis_epicardial (epicardial myocarditis)

Pre-loaded in `data/cardiac/` as NPZ files. Each file contains:
- `images`: Array of shape (N, 128, 128)
- `labels`: Array of shape (N,)

## OASIS Ventricle
128x128 grayscale brain MRI coronal slices from the OASIS-3 dataset,
cropped and resized to focus on the ventricle region.

Pre-loaded in `data/oasis/` as:
- `coronal.npz`: Images and metadata
- `coronal_bboxes.npz`: Ventricle bounding box annotations
- `bbox_metadata.json`: Bounding box metadata

This is an unsupervised clustering task (no ground truth labels).
