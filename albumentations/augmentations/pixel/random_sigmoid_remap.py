"""Random sigmoid-based grayscale remapping for AlbumentationsX.

This module ports the MATLAB pseudo-sigmoid LUT construction used by the
project's latest helper script. The transform is image-only, so bounding boxes,
keypoints, masks, and labels are not modified.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from albucore import sz_lut
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.type_definitions import ImageType

__all__ = ["RandomSigmoidRemap"]

# Keep the MATLAB curve-generation resolution fixed at 256 samples. For wider
# integer image types, the normalized curve is expanded to a direct-index LUT
# before it is applied. This preserves the original curve discretization while
# avoiding per-pixel interpolation for uint16 images.
_CURVE_BINS = 256
_CURVE_X = np.linspace(0.0, 1.0, _CURVE_BINS, dtype=np.float32)
_UINT16_X = np.linspace(0.0, 1.0, 1 << 16, dtype=np.float32)

# Parameters ported from the MATLAB development script.
_SMOOTHING_FACTOR = 10
_C_D_SHAPE_FACTOR = 1.2
_B_SHAPE_FACTOR = 1
_B_RANGE = (0.1, 10.0)
_C_RANGE = (0.2, 0.8)
_D_RANGE = (0.4, 0.6)

def _matlab_movmean_shrink(
    values: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Reproduce movmean(values, k, 'Endpoints', 'shrink') for a 1-D array.

    MATLAB centers odd windows on the current element.

    For an even window size, MATLAB centers the window on the current and
    previous elements. Therefore, an even window contains one more sample
    before the current element than after it.

    Endpoint windows are shortened to contain only samples that exist.

    A cumulative-sum implementation is used so runtime and memory usage are
    linear in the LUT length.
    """

    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")

    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    num_values = values.size

    if window_size == 1 or num_values == 0:
        return values.copy()

    if window_size % 2 == 0:
        num_before = window_size // 2
        num_after = window_size // 2 - 1
    else:
        num_before = window_size // 2
        num_after = window_size // 2

    indices = np.arange(num_values, dtype=np.int64)

    start_indices = np.maximum(indices - num_before, 0)
    stop_indices = np.minimum(indices + num_after + 1, num_values)

    cumulative_sum = np.empty(num_values + 1, dtype=np.float64)
    cumulative_sum[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=cumulative_sum[1:])

    moving_sums = (
        cumulative_sum[stop_indices]
        - cumulative_sum[start_indices]
    )

    sample_counts = stop_indices - start_indices

    return moving_sums / sample_counts


def _generate_sigmoid_lut(
    b: float,
    c: float,
    d: float,
    doFlip: bool,
    num_points: int = _CURVE_BINS,
) -> np.ndarray:
    """Generate the normalized pseudo-sigmoid LUT.

    This is a direct NumPy port of the available MATLAB curve construction for
    the linear-window path. The latest MATLAB helper applies the windowed curve
    to the image, so this function returns that curve rather than the separate
    diagnostic/smoothed output.

    Args:
        b: Sigmoid sharpness.
        c: Sigmoid center in normalized input-intensity coordinates.
        d: Fraction of the sigmoid output range used to choose the split point.
        num_points: Number of normalized samples in the generated curve.

    Returns:
        A contiguous float32 array of shape ``(num_points,)`` in ``[0, 1]``.
    """
    if num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {num_points}.")

    if num_points == _CURVE_BINS:
        x = _CURVE_X
    else:
        x = np.linspace(0.0, 1.0, num_points, dtype=np.float32)

    # MATLAB: 1 ./ (1 + exp(-b * (x - c)))
    z_original = np.reciprocal(
        1.0 + np.exp(-np.float32(b) * (x - np.float32(c))),
        dtype=np.float32,
    )

    split_value = z_original.min() + np.ptp(z_original) * np.float32(d)
    flip_mask = z_original > split_value
    split_indices = np.flatnonzero(flip_mask)

    if split_indices.size == 0:
        raise RuntimeError("The sampled sigmoid did not produce a valid split point.")

    midpoint = int(split_indices[0])

    # MATLAB's two assignments both include winMidPt. The second assignment
    # overwrites the same midpoint with 1, which is reproduced here.
    window = np.zeros(num_points, dtype=np.float32)
    window[: midpoint + 1] = np.linspace(
        0.0,
        1.0,
        midpoint + 1,
        dtype=np.float32,
    )
    window[midpoint:] = np.linspace(
        1.0,
        0.0,
        num_points - midpoint,
        dtype=np.float32,
    )

    z_flipped = z_original.copy()
    z_flipped[flip_mask] = 1.0 - z_flipped[flip_mask]

    z_windowed = z_flipped * window
    z_windowed[flip_mask] = 1.0 - z_windowed[flip_mask]

    z_smoothed = _matlab_movmean_shrink(z_windowed,int(round(num_points/_SMOOTHING_FACTOR)))

    z_final = (z_smoothed - z_smoothed.min())/np.ptp(z_smoothed)

    if doFlip:
        z_final = np.flip(z_final)

    return np.ascontiguousarray(z_final, dtype=np.float32)


def _validate_single_grayscale_image(image: np.ndarray) -> None:
    if image.ndim == 2:
        return
    if image.ndim == 3 and image.shape[-1] == 1:
        return
    #raise TypeError(
    #    "RandomSigmoidRemap expects a grayscale image with shape (H, W) "
    #    "or (H, W, 1).",
    #)


def _validate_grayscale_batch(images: np.ndarray) -> None:
    if images.ndim == 3:
        return
    if images.ndim == 4 and images.shape[-1] == 1:
        return
    raise TypeError(
        "RandomSigmoidRemap expects grayscale image batches with shape "
        "(N, H, W) or (N, H, W, 1).",
    )


def _normalized_to_uint8_lut(lut: np.ndarray) -> np.ndarray:
    return np.rint(lut * 255.0).astype(np.uint8)


def _normalized_to_uint16_lut(lut: np.ndarray) -> np.ndarray:
    # Build the 65,536-entry table once per sampled transform, then apply it
    # through direct integer indexing. This is typically faster than performing
    # linear interpolation for every uint16 pixel in a training image.
    expanded = np.interp(_UINT16_X, _CURVE_X, lut)
    return np.rint(expanded * 65535.0).astype(np.uint16)


def _apply_float_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    # Albumentations float images are conventionally normalized to [0, 1].
    # Linear interpolation mirrors the MATLAB remapping function for a LUT whose
    # length differs from the number of representable input values.
    work_dtype = np.float32 if image.dtype == np.float32 else np.float64
    scaled = image.astype(work_dtype, copy=False) * (_CURVE_BINS - 1)
    np.clip(scaled, 0.0, _CURVE_BINS - 1, out=scaled)

    lower = scaled.astype(np.uint16)
    upper = np.minimum(lower + 1, _CURVE_BINS - 1)
    fraction = scaled - lower

    output = lut[lower] + fraction * (lut[upper] - lut[lower])
    return output.astype(image.dtype, copy=False)


def _apply_normalized_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a normalized LUT while preserving shape and dtype."""
    if image.dtype == np.uint8:
        return sz_lut(image, _normalized_to_uint8_lut(lut), inplace=False)

    if image.dtype == np.uint16:
        return _normalized_to_uint16_lut(lut)[image]

    if image.dtype in (np.float32, np.float64):
        return _apply_float_lut(image, lut)

    raise TypeError(
        "RandomSigmoidRemap supports uint8, uint16, float32, and float64 "
        f"images, but received {image.dtype}.",
    )


class RandomSigmoidRemap(ImageOnlyTransform):
    """Apply a random pseudo-sigmoid LUT to a grayscale image.

    The curve-generation behavior is ported from the supplied MATLAB code:

    * sigmoid sharpness is sampled uniformly from ``[0.1, 10]``;
    * sigmoid center is sampled from ``N(0.5, 0.4)`` and clipped to
      ``[0.1, 0.9]``;
    * the split/window location is sampled from ``N(0.5, 0.4)`` and clipped to
      ``[0.1, 0.9]``;
    * the linear window and flip/window/unflip construction are unchanged.

    Args:
        p: Probability of applying the transform. Defaults to 0.5.

    Targets:
        image, images

    Image types:
        uint8, uint16, float32, float64

    Number of channels:
        1

    Examples:
        >>> import albumentations as A
        >>> import numpy as np
        >>> image = np.random.randint(0, 256, (640, 640), dtype=np.uint8)
        >>> transform = A.Compose(
        ...     [RandomSigmoidRemap(p=1.0)],
        ...     seed=137,
        ... )
        >>> transformed = transform(image=image)["image"]
    """

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)

    def get_params(self) -> dict[str, float]:
        b = float(_B_RANGE[0] + (_B_RANGE[1] - _B_RANGE[0])*float(self.random_generator.beta(_B_SHAPE_FACTOR, _B_SHAPE_FACTOR)))
        c = float(_C_RANGE[0] + (_C_RANGE[1] - _C_RANGE[0])*float(self.random_generator.beta(_C_D_SHAPE_FACTOR, _C_D_SHAPE_FACTOR)))
        d = float(_D_RANGE[0] + (_D_RANGE[1] - _D_RANGE[0])*float(self.random_generator.beta(_C_D_SHAPE_FACTOR, _C_D_SHAPE_FACTOR)))
        doFlip = True if self.random_generator.uniform(0,1) >= 0.5 else False

        return {"b": b, "c": c, "d": d, "doFlip": doFlip}

    def apply(
        self,
        img: ImageType,
        b: float,
        c: float,
        d: float,
        doFlip: bool,
        **params: Any,
    ) -> ImageType:
        _validate_single_grayscale_image(img)
        lut = _generate_sigmoid_lut(b=b, c=c, d=d, doFlip=doFlip)
        return _apply_normalized_lut(img, lut)

    def apply_to_images(
        self,
        images: ImageType,
        b: float,
        c: float,
        d: float,
        doFlip: bool,
        **params: Any,
    ) -> ImageType:
        _validate_grayscale_batch(images)
        lut = _generate_sigmoid_lut(b=b, c=c, d=d, doFlip=doFlip)

        if images.dtype == np.uint8:
            return _normalized_to_uint8_lut(lut)[images]

        if images.dtype == np.uint16:
            return _normalized_to_uint16_lut(lut)[images]

        if images.dtype in (np.float32, np.float64):
            return _apply_float_lut(images, lut)

        raise TypeError(
            "RandomSigmoidRemap supports uint8, uint16, float32, and float64 "
            f"images, but received {images.dtype}.",
        )
