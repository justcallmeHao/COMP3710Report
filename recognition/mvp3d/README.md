# Small-subset 3D U-Net

This implements `MVP_CODEX_PROMPT.md` and Part 3 of `FEASIBILITY_REVIEW.md`.
It is a baseline pipeline prototype, with six generic class IDs, not the Improved
U-Net. No measured accuracy, memory usage or execution speed is claimed.

## Environment

Use Python 3.10 or newer in a dedicated environment. Install a PyTorch build
appropriate for your operating system, GPU and driver using PyTorch's installation
selector at https://pytorch.org/get-started/locally/; choose CPU explicitly if that
is the intended device. No CUDA wheel version is prescribed here. Then install
the repository dependencies with `python -m pip install -r requirements.txt`.
The code uses the PyTorch 2.3+ `torch.amp.GradScaler` API. Dependencies have not been
installed or runtime-tested as part of this implementation.

Run the following commands from the repository root. Output directories must not
already exist, to prevent accidental mixing of runs. All examples save under the
already ignored `outputs/` directory.

Direct script execution is also supported: from inside `mvp3d/`, use
`python train.py --mode subset --epochs 20`, or from the repository root use
`python mvp3d/train.py --mode subset --epochs 20`. The same applies to
`predict.py`. Default configuration, manifest and output paths resolve relative
to the source files, so changing the working directory does not change them.
Explicit relative `--output`, `--config` and `--manifest` paths resolve from your
working directory.

```powershell
# Exactly one optimizer update on B006.
python -m mvp3d.train --mode step --device cuda --output outputs/mvp3d/step

# Repeated B006 training: default 20 updates (a small diagnostic budget).
python -m mvp3d.train --mode one-pair --steps 20 --device cuda --output outputs/mvp3d/one-pair

# B006 + B037 training, B038 validation: 3 epochs, 6 optimizer updates.
python -m mvp3d.train --mode subset --epochs 3 --device cuda --output outputs/mvp3d/subset

# Image-only prediction followed by reference-based review.
python -m mvp3d.predict --checkpoint outputs/mvp3d/subset/best.pt --case B038_Week0 --device cuda --output outputs/mvp3d/review-B038

# Inspect learning on the training pair separately from validation.
python -m mvp3d.predict --checkpoint outputs/mvp3d/one-pair/last.pt --case B006_Week0 --device cuda --output outputs/mvp3d/review-B006
```

Use `--device cpu` to explicitly request CPU execution; unavailable CUDA raises an
error instead of falling back. Add `--amp` to training for CUDA mixed precision.
Prediction uses float32. Set `--config path/to/config.json` for a changed grid,
learning rate or seed; all preprocessing settings are stored in checkpoints and
reused by prediction. `--steps` controls one-pair updates; `--epochs` controls
subset epochs. Step mode always performs exactly one update. There is no automatic
extension of the budget, augmentation, random split or resume mode.

## Data and spatial conventions

Only the six paths in `subset_data/manifest.json` can be used. Paths resolve
relative to that file, independent of the working directory. The loader checks
their SHA-256 hashes against the manifest when loading. The exact fixed split is
validated: B006 and B037 training, B038 validation. B006 alone is used for the
first two modes. These are provisional filename patient groups, not independently
verified identities, and there is no test set.

Each selected image or pair is loaded independently. Source volumes must be
strictly 3D, finite and have invertible finite affines. Masks must contain integer
IDs 0-5. Pair shapes and affines must match; no alignment repair, cropping or
padding is attempted. MRI units must be mm. Unknown mask units inherit mm only
after the pair passes shape and affine checks; explicit incompatible units are
rejected. The assumption is recorded in each preprocessing JSON.

The target grid is constructed from MRI geometry alone and shared with its mask.
In each NIfTI array axis, source length `n` and target length `m` give scale `s=n/m`.
Target voxel centre `j` maps to source coordinate `s*j + (s-1)/2`.
Thus outer voxel-cell edges at `-0.5` and `n-0.5` retain the same physical coverage.
The new affine is the source affine multiplied by this scale/translation.
This preserves obliquity and does not canonicalize orientation or register cases.
Nearest boundary extension is used where interpolation reaches beyond the first
or last centre, particularly when mapping predictions back to the original grid.

MRI uses linear interpolation; labels and predicted IDs use nearest neighbour.
Class counts before and after resampling are saved. A disappearing class raises
an error listing the counts; increase the grid resolution before proceeding.
Surviving counts alone do not prove that boundaries or small structures are usable.

After resampling, MRI is z-scored using **all image voxels**, including background.
No mask selects normalization voxels or a crop. Mean and standard deviation use
float64 accumulation; normalized tensors use float32. Standard deviation at or
below `1e-8` produces a zero image and a metadata flag requiring investigation.
NIfTI `(H,W,D)` becomes tensor `(D,H,W)` via `transpose(2,0,1)`.
Items are image `(1,D,H,W)` float32, target `(D,H,W)` int64, and metadata.
The DataLoader adds the batch dimension, with batch size 1 and zero workers.
Default target `(64,128,128)` produces bottleneck `(8,16,16)`.

## Model, objective and scoring

Three two-convolution encoder blocks have widths 16/32/64, with a 128-channel
bottleneck. Every block convolution is padded 3x3x3, followed by affine instance
normalization and LeakyReLU. Max pooling reduces dimensions by two. Decoder stages
use nearest upsampling, a channel-reducing convolution, concatenated skips and a
two-convolution block at widths 64/32/16. A 1x1x1 head returns six raw logits.
Dimensions must be at least 16 and divisible by eight.

The loss is `0.5 * foreground soft Dice loss + 0.5 * cross-entropy`. CE includes
all six classes. Soft Dice uses float32 softmax and six-channel one-hot targets,
reduces spatially per case/class and averages classes 1-5. Its formula is
`(2*intersection + 1e-5)/(prediction_mass + target_mass + 1e-5)`.
All five foreground channels remain in the objective, even when absent from the
target: predicted mass lowers their score and is also penalized by CE. Soft Dice
is not the reported hard evaluation metric. Class 0 is only provisionally assumed
to be background; no anatomical class names have been inferred.

Hard metrics use argmax and report Dice and IoU for every class, predicted and
reference voxel counts, and foreground macro Dice. When both sets are empty the
score is JSON null/blank CSV and excluded from the foreground mean; absent-target
false positives score zero. There is no smoothing in hard metrics. An explicit
`all_background_prediction` flag prevents interpreting background-only output as
successful learning. Metrics are computed on the transformed grid.

## Artifacts and review

Runs record configuration, seed, manifest snapshot/hash, software versions,
device name, parameter count, preprocessing metadata, class counts, alignment
panels, CSV/JSON metrics, learning curves and `last.pt`. Subset mode additionally
saves `best.pt`, selected solely by B038 hard foreground macro Dice. A best
checkpoint can still be a poor model; selection is not a success criterion.
One-pair metrics are training metrics, never validation scores.

Training rows describe predictions **before** that row's optimizer update.
One-pair/step modes also log a final evaluation of the updated model on B006.
Finite loss, finite gradients for all trainable parameters and output/target
dimensions are checked before committing an optimizer update.

Step timing excludes data loading/resampling and includes transfers, forward,
loss, backward/update when training, hard metrics and synchronized CUDA completion.
CUDA allocated/reserved peaks refer to PyTorch's allocator; sampled device-used
bytes cover device memory at one instant and are not peak total GPU usage.
CPU memory is not measured (memory fields are null on CPU). These measurements do
not establish the project's peak-total-GPU budget. There is no CPU/GPU speed claim.

Prediction checks manifest identity and reloads the checkpoint's preprocessing.
No reference mask is opened until image inference and original-grid remapping
finish. It saves target-grid and original-grid NIfTI predictions with their
respective affines and mm units. Original-grid predictions use nearest-neighbour
mapping through the affines, not an array resize. Source header fields unrelated
to geometry are not copied. The report defines the full-case timing boundaries:
image hash/read through preprocessing, forward, CPU transfer and original-grid
remapping; model initialization, disk output and reference review are excluded.
This is one measured call without warm-up, not a benchmark distribution.

Review PNGs show quarter, middle and three-quarter depth slices, identified by
case and tensor-D index (original array axis 2), with voxel-spacing aspect ratio.
Training panels show MRI/reference/alignment overlay; prediction adds predicted
IDs and errors. Red is a wrong nonzero predicted class (FP), blue is a missed
nonzero reference class (FN), and magenta is a nonzero wrong-class substitution
(both). Black is correct. These are array-axis views, not assured anatomical planes.
Inspect additional slices in a NIfTI viewer if necessary; three panels cannot
establish whole-volume annotation correctness.

A useful one-pair check shows finite updates, decreasing loss, improving hard
foreground Dice and visible non-background structures approaching the reference.
Twenty updates may be insufficient; explicitly raise the budget after checking
the pipeline. For constant predictions, check image variance, target counts,
normalization, overlays and per-class predicted counts. For disappearing labels,
use a finer divisible-by-eight grid. For invalid losses/gradients, inspect finite
inputs and labels and try float32 without AMP before tuning. For out-of-memory
errors, record the failed setting; do not accept a coarser grid that erases classes.

Patient metadata, anatomical label meanings, visual alignment and generalization
remain unverified. Tiny-subset scores cannot demonstrate the assignment's held-out
Dice target or clinical usefulness.

## Implementation verification

Only static Python syntax and JSON configuration checks were performed during
implementation. No training, model forward/backward pass, NIfTI processing,
prediction, graphical review, dependency installation or existing loader/viewer
execution was performed. Runtime compatibility and numerical behavior remain to
be checked by running the explicit commands above.
