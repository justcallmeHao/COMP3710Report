import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from process2D import _class_count, _prepare_names, _read_image, _view_example, to_channels


def load_data_3D(
    imageNames, normImage=False, categorical=False, dtype=np.float32,
    getAffines=False, orient=False, early_stop=False, *, num_classes=None,
):
    """Load volumes into (N, H, W, D), or (N, H, W, D, C) for labels.

    normImage applies per-volume z-score normalization, not 0-1 scaling.
    orient=True reorders/flips voxels into canonical RAS+ orientation and
    updates their affines; it does NOT resample voxel spacing or register scans.
    Only singleton fourth dimensions are removed.

    As in Listing 2, the first volume sets the output shape: excess depth is
    clipped and smaller volumes are zero-padded at the high-index ends.
    Larger in-plane dimensions raise an error. Padding in categorical outputs
    is encoded as background (label 0). Returned affines describe each volume's
    own grid, not a shared registered grid.

    early_stop loads at most 22 volumes without unused output rows.
    num_classes fixes the channel count across calls; otherwise all selected
    files are scanned to infer it. getAffines returns (images, affines).
    """
    names = _prepare_names(imageNames, early_stop, normImage, categorical, dtype)

    def read_volume(filename):
        image = _read_image(filename, 3)
        return nib.as_closest_canonical(image) if orient else image

    first = read_volume(names[0])
    rows, cols, depth = first.shape
    channels = _class_count(names, 3, num_classes) if categorical else None
    output_shape = (len(names), rows, cols, depth)
    if categorical:
        output_shape += (channels,)
    images = np.zeros(output_shape, dtype=dtype)
    if categorical:
        images[..., 0] = 1
    affines = []

    for i, filename in enumerate(tqdm(names, desc="Loading 3D NIfTI")):
        nifti_image = read_volume(filename)
        data = nifti_image.get_fdata(caching="unchanged")[:, :, :depth]
        if data.shape[0] > rows or data.shape[1] > cols:
            raise ValueError(
                f"{filename}: in-plane shape {data.shape[:2]} exceeds "
                f"the first volume's shape {(rows, cols)}."
            )
        if categorical:
            data = to_channels(data, dtype=dtype, num_classes=channels)
        elif normImage:
            data = (data - data.mean()) / (data.std() + 1e-8)
        images[i, :data.shape[0], :data.shape[1], :data.shape[2]] = data
        affines.append(nifti_image.affine.copy())

    return (images, affines) if getAffines else images


def main():
    """View the middle slice of one matching 3D MRI and segmentation."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Dataset root containing semantic_MRs and semantic_labels_only.",
    )
    args = parser.parse_args()
    for image_path in sorted((args.data_dir / "semantic_MRs").glob("*_LFOV.nii*")):
        label_path = (
            args.data_dir / "semantic_labels_only"
            / image_path.name.replace("_LFOV.nii", "_SEMANTIC.nii")
        )
        if label_path.is_file():
            _view_example(image_path, label_path, load_data_3D)
            return
    raise FileNotFoundError(
        f"No matching 3D MRI/segmentation pair found under {args.data_dir}."
    )


if __name__ == "__main__":
    main()
