import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8, num_classes=None) -> np.ndarray:
    """One-hot encode nonnegative integer labels, with channel c representing c.

    Pass the same num_classes for all datasets/splits to keep channels consistent.
    Missing labels retain an all-zero channel instead of shifting other labels.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("Cannot encode an empty label array.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr != np.floor(arr)):
        raise ValueError("Categorical labels must be finite nonnegative integers.")
    required = int(arr.max()) + 1
    if num_classes is None:
        num_classes = required
    if not isinstance(num_classes, (int, np.integer)) or num_classes < required:
        raise ValueError(f"num_classes must be an integer >= {required}.")
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in np.unique(arr):
        res[..., int(c)] = arr == c
    return res


def _read_image(filename, ndim):
    """Read an image, removing only a singleton trailing dimension."""
    image = nib.load(filename)
    if len(image.shape) == ndim + 1 and image.shape[-1] == 1:
        image = image.slicer[..., 0]
    if len(image.shape) != ndim:
        raise ValueError(f"{filename}: expected {ndim}D data, got {image.shape}.")
    return image


def _class_count(image_names, ndim, num_classes):
    """Find a common channel count without holding all images in memory."""
    required = 1
    for filename in image_names:
        data = _read_image(filename, ndim).get_fdata(caching="unchanged")
        if data.size == 0 or not np.all(np.isfinite(data)):
            raise ValueError(f"{filename}: labels must be nonempty and finite.")
        if np.any(data < 0) or np.any(data != np.floor(data)):
            raise ValueError(f"{filename}: labels must be nonnegative integers.")
        required = max(required, int(data.max()) + 1)
    if num_classes is None:
        return required
    if not isinstance(num_classes, (int, np.integer)) or num_classes < required:
        raise ValueError(f"num_classes must be an integer >= {required}.")
    return num_classes


def _prepare_names(image_names, early_stop, norm_image, categorical, dtype):
    names = list(image_names)
    if not names:
        raise ValueError("imageNames must contain at least one NIfTI filename.")
    if norm_image and categorical:
        raise ValueError("Categorical segmentation labels must not be normalized.")
    if norm_image and not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError("Normalization requires a floating-point dtype.")
    # The listing's i > 20 condition loads 22 cases; avoid unused output rows.
    return names[:22] if early_stop else names


def load_data_2D(
    imageNames, normImage=False, categorical=False, dtype=np.float32,
    getAffines=False, early_stop=False, *, num_classes=None,
):
    """Load equally sized NIfTI slices into a preallocated NumPy array.

    Returns (N, H, W), or (N, H, W, C) for categorical labels. A trailing
    singleton axis is removed; actual volumes are rejected instead of silently
    taking their first slice. normImage applies per-image z-score normalization,
    not 0-1 scaling. early_stop loads at most 22 images. getAffines returns
    (images, affines). num_classes fixes the label channel count across calls;
    otherwise it is inferred from all selected files.
    """
    names = _prepare_names(imageNames, early_stop, normImage, categorical, dtype)
    first = _read_image(names[0], 2)
    shape = first.shape
    channels = _class_count(names, 2, num_classes) if categorical else None
    output_shape = (len(names),) + shape
    if categorical:
        output_shape += (channels,)
    images = np.zeros(output_shape, dtype=dtype)
    affines = []

    for i, filename in enumerate(tqdm(names, desc="Loading 2D NIfTI")):
        nifti_image = _read_image(filename, 2)
        if nifti_image.shape != shape:
            raise ValueError(f"{filename}: expected shape {shape}, got {nifti_image.shape}.")
        data = nifti_image.get_fdata(caching="unchanged")
        if categorical:
            data = to_channels(data, dtype=dtype, num_classes=channels)
        elif normImage:
            data = (data - data.mean()) / (data.std() + 1e-8)
        images[i] = data
        affines.append(nifti_image.affine.copy())

    return (images, affines) if getAffines else images


def _view_example(image_path, label_path, loader):
    """Display one matched pair, using the middle axis-2 slice for a volume."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    images, image_affines = loader([image_path], normImage=True, getAffines=True)
    labels, label_affines = loader([label_path], getAffines=True)
    if images.shape != labels.shape:
        raise ValueError(f"Image and label shapes differ: {images.shape}, {labels.shape}.")
    if not np.allclose(image_affines[0], label_affines[0], rtol=1e-5, atol=1e-5):
        raise ValueError("Image and label affines differ; this pair needs alignment.")

    image, segmentation = images[0], labels[0]
    title = image_path.name
    if image.ndim == 3:
        middle = image.shape[2] // 2
        image = image[:, :, middle]
        segmentation = segmentation[:, :, middle]
        title += f" | axis-2 slice {middle}"

    class_ids = np.arange(int(labels.max()) + 1)
    cmap = plt.get_cmap("tab10", len(class_ids))
    norm = BoundaryNorm(np.arange(len(class_ids) + 1) - 0.5, cmap.N)
    # Preserve the relative voxel sizes within the displayed array plane.
    spacing = np.linalg.norm(image_affines[0][:3, :3], axis=0)
    aspect = spacing[0] / spacing[1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    axes[0].imshow(image, cmap="gray", interpolation="nearest", aspect=aspect)
    axes[0].set_title("MRI (normalized intensity)")
    mask_plot = axes[1].imshow(
        segmentation, cmap=cmap, norm=norm, interpolation="nearest", aspect=aspect
    )
    axes[1].set_title("Segmentation (class IDs)")
    axes[2].imshow(image, cmap="gray", interpolation="nearest", aspect=aspect)
    axes[2].imshow(
        np.ma.masked_equal(segmentation, 0), cmap=cmap, norm=norm,
        alpha=0.4, interpolation="nearest", aspect=aspect,
    )
    axes[2].set_title("Overlay (label 0 hidden)")
    for axis in axes:
        axis.set_xlabel("Axis 1 / column")
        axis.set_ylabel("Axis 0 / row")
    fig.colorbar(mask_plot, ax=axes[1:], ticks=class_ids, label="Class ID")
    fig.suptitle(title)
    plt.show()


def main():
    """View the first matching training slice and segmentation."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Dataset root containing keras_slices_data.",
    )
    args = parser.parse_args()
    root = args.data_dir / "keras_slices_data"
    for image_path in sorted((root / "keras_slices_train").glob("case_*.nii*")):
        label_path = root / "keras_slices_seg_train" / image_path.name.replace("case_", "seg_", 1)
        if label_path.is_file():
            _view_example(image_path, label_path, load_data_2D)
            return
    raise FileNotFoundError(f"No matching 2D image/segmentation pair found under {root}.")


if __name__ == "__main__":
    main()
