"""Bounding-box-safe random aspect-ratio crop."""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Self

from albumentations.augmentations.crops.base import BaseCrop, CropSizeError
from albumentations.core.transforms_interface import BaseTransformInitSchema


__all__ = ["BBoxSafeRandomZoom"]


def _get_reference_crop_size(
    image_height: int,
    image_width: int,
    aspect_ratio: float,
) -> tuple[float, float]:
    """Return the largest requested-aspect crop that fits in the image.

    Args:
        image_height: Input image height in pixels.
        image_width: Input image width in pixels.
        aspect_ratio: Requested crop width divided by crop height.

    Returns:
        Reference crop height and width as floating-point values.
    """

    image_aspect_ratio = image_width / image_height

    if image_aspect_ratio >= aspect_ratio:
        reference_height = float(image_height)
        reference_width = reference_height * aspect_ratio
    else:
        reference_width = float(image_width)
        reference_height = reference_width / aspect_ratio

    return reference_height, reference_width


def _get_padded_bbox_union(
    bboxes: np.ndarray,
    image_height: int,
    image_width: int,
    padding: int,
) -> tuple[float, float, float, float] | None:
    """Return the padded union of normalized Albumentations bounding boxes.

    Albumentations supplies bounding boxes to transforms in its normalized
    internal x_min, y_min, x_max, y_max representation. Additional columns,
    such as labels, are ignored.

    Padding is added independently to all four sides and clipped to the
    image boundaries.
    """

    if len(bboxes) == 0:
        return None

    coordinates = bboxes[:, :4]

    x_min = np.floor(float(np.min(coordinates[:, 0]) * image_width))
    y_min = np.floor(float(np.min(coordinates[:, 1]) * image_height))
    x_max = np.ceil(float(np.max(coordinates[:, 2]) * image_width))
    y_max = np.ceil(float(np.max(coordinates[:, 3]) * image_height))

    return (
        max(0.0, x_min - padding),
        max(0.0, y_min - padding),
        min(float(image_width), x_max + padding),
        min(float(image_height), y_max + padding),
    )


def _get_valid_crop_origin_range(
    crop_size: int,
    image_size: int,
    required_min: float | None,
    required_max: float | None,
) -> tuple[int, int]:
    """Return the valid inclusive integer origin range along one axis."""

    minimum_origin = 0
    maximum_origin = image_size - crop_size

    if required_min is not None and required_max is not None:
        # The crop must begin at or before the padded bbox minimum.
        maximum_origin = min(
            maximum_origin,
            int(np.floor(required_min)),
        )

        # The crop must end at or after the padded bbox maximum.
        minimum_origin = max(
            minimum_origin,
            int(np.ceil(required_max - crop_size)),
        )

    return minimum_origin, maximum_origin


class BBoxSafeRandomZoom(BaseCrop):
    """Randomly crop an image while retaining all padded bounding boxes.

    The transform first calculates the largest crop with the requested
    aspect ratio that fits inside the image. This reference crop is defined
    as zoom factor 1.

    The sampled crop dimensions are then calculated as:

        crop size = reference crop size / zoom factor

    All bounding boxes are combined into one enclosing rectangle. The
    rectangle is expanded by ``bbox_padding`` pixels on every side before
    the maximum feasible zoom and valid crop positions are calculated.

    The output is not resized. A later resize transform can be added to the
    augmentation pipeline when fixed output dimensions are required.

    Args:
        aspect_ratio:
            Requested crop width divided by crop height.

        zoom_range:
            Minimum and maximum zoom factors.

            Both values must be at least 1. A value of 1 uses the largest
            requested-aspect crop that fits inside the image. Larger values
            produce smaller crops.

        bbox_padding:
            Number of pixels added to every side of the union of all
            bounding boxes before calculating the maximum safe zoom and
            valid crop locations.

            Padding is clipped at the image boundaries.

        p:
            Probability of applying the transform.

    Targets:
        image, mask, bboxes, keypoints, volume, mask3d

    Image types:
        uint8, float32

    Supported bounding boxes:
        Horizontal and oriented bounding boxes, as supported by BaseCrop.

    Notes:
        If the requested minimum zoom is larger than the maximum bbox-safe
        zoom, the maximum bbox-safe zoom is used. This preserves every
        padded bounding box even though the sampled zoom falls below the
        requested minimum.

        If the padded bbox union cannot fit inside the reference crop even
        at zoom 1, no requested-aspect crop can satisfy the constraints and
        CropSizeError is raised.

        Sampling an integer crop origin is equivalent to sampling an
        integer crop-center location because the crop size is fixed after
        the zoom factor is selected.

    Examples:
        >>> import albumentations as A
        >>>
        >>> transform = A.Compose(
        ...     [
        ...         A.BBoxSafeRandomZoom(
        ...             aspect_ratio=1.0,
        ...             zoom_range=(1.0, 3.0),
        ...             bbox_padding=10,
        ...             p=1.0,
        ...         ),
        ...         A.Resize(height=640, width=640),
        ...     ],
        ...     bbox_params=A.BboxParams(
        ...         coord_format="yolo",
        ...         label_fields=["class_labels"],
        ...     ),
        ...     seed=137,
        ... )
        >>>
        >>> result = transform(
        ...     image=image,
        ...     bboxes=bboxes,
        ...     class_labels=class_labels,
        ... )
    """

    class InitSchema(BaseTransformInitSchema):
        aspect_ratio: Annotated[float, Field(gt=0)]
        zoom_range: tuple[float, float]
        bbox_padding: Annotated[int, Field(ge=0)]

        @model_validator(mode="after")
        def validate_zoom_range(self) -> Self:
            minimum_zoom, maximum_zoom = self.zoom_range

            if minimum_zoom < 1.0 or maximum_zoom < 1.0:
                raise ValueError(
                    "zoom_range values must be greater than or equal to 1.0.",
                )

            if minimum_zoom > maximum_zoom:
                raise ValueError(
                    "zoom_range must be ordered as "
                    "(minimum_zoom, maximum_zoom).",
                )

            return self

    def __init__(
        self,
        aspect_ratio: float,
        zoom_range: tuple[float, float] = (1.0, 2.0),
        bbox_padding: int = 0,
        p: float = 0.5,
    ):
        super().__init__(p=p)

        self.aspect_ratio = aspect_ratio
        self.zoom_range = zoom_range
        self.bbox_padding = bbox_padding

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Sample a bbox-safe zoom factor and crop location."""

        image_height, image_width = params["shape"][:2]

        reference_height, reference_width = _get_reference_crop_size(
            image_height=image_height,
            image_width=image_width,
            aspect_ratio=self.aspect_ratio,
        )

        bboxes = np.asarray(
            data.get("bboxes", []),
            dtype=np.float32,
        )

        padded_bbox_union = _get_padded_bbox_union(
            bboxes=bboxes,
            image_height=image_height,
            image_width=image_width,
            padding=self.bbox_padding,
        )

        user_minimum_zoom, user_maximum_zoom = self.zoom_range

        if padded_bbox_union is None:
            maximum_bbox_safe_zoom = user_maximum_zoom

        else:
            (
                bbox_x_min,
                bbox_y_min,
                bbox_x_max,
                bbox_y_max,
            ) = padded_bbox_union

            padded_bbox_width = bbox_x_max - bbox_x_min
            padded_bbox_height = bbox_y_max - bbox_y_min

            horizontal_zoom_limit = (
                reference_width / padded_bbox_width
                if padded_bbox_width > 0
                else np.inf
            )

            vertical_zoom_limit = (
                reference_height / padded_bbox_height
                if padded_bbox_height > 0
                else np.inf
            )

            maximum_bbox_safe_zoom = min(
                horizontal_zoom_limit,
                vertical_zoom_limit,
            )

            if maximum_bbox_safe_zoom < 1.0:
                raise CropSizeError(
                    "The padded bounding-box union cannot fit inside a crop "
                    f"with aspect ratio {self.aspect_ratio} at zoom 1. "
                    "Reduce bbox_padding, change aspect_ratio, or use images "
                    "whose bounding boxes occupy a smaller region.",
                )

        effective_maximum_zoom = min(
            user_maximum_zoom,
            maximum_bbox_safe_zoom,
        )

        # Preserve all padded boxes even when the user's minimum zoom is
        # larger than the feasible maximum.
        effective_minimum_zoom = min(
            user_minimum_zoom,
            effective_maximum_zoom,
        )

        if np.isclose(
            effective_minimum_zoom,
            effective_maximum_zoom,
        ):
            zoom_factor = effective_maximum_zoom
        else:
            zoom_factor = self.py_random.uniform(
                effective_minimum_zoom,
                effective_maximum_zoom,
            )

        # Ceil prevents rounding from making the crop smaller than the
        # continuous reference-size / zoom calculation.
        crop_height = min(
            image_height,
            max(
                1,
                int(np.ceil(reference_height / zoom_factor)),
            ),
        )

        crop_width = min(
            image_width,
            max(
                1,
                int(np.ceil(reference_width / zoom_factor)),
            ),
        )

        if padded_bbox_union is None:
            bbox_x_min = None
            bbox_y_min = None
            bbox_x_max = None
            bbox_y_max = None

        x_origin_min, x_origin_max = _get_valid_crop_origin_range(
            crop_size=crop_width,
            image_size=image_width,
            required_min=bbox_x_min,
            required_max=bbox_x_max,
        )

        y_origin_min, y_origin_max = _get_valid_crop_origin_range(
            crop_size=crop_height,
            image_size=image_height,
            required_min=bbox_y_min,
            required_max=bbox_y_max,
        )

        if (
            x_origin_min > x_origin_max
            or y_origin_min > y_origin_max
        ):
            raise CropSizeError(
                "No valid integer crop location was found for the sampled "
                "zoom and padded bounding boxes."
                f"bbox_y_min: {bbox_y_min}"
                f"bbox_y_max: {bbox_y_max}"
                f"bbox_x_min: {bbox_x_min}"
                f"bbox_x_max: {bbox_x_max}"
                f"crop_height: {crop_height}"
                f"crop_width: {crop_width}",
            )

        crop_x_min = self.py_random.randint(
            x_origin_min,
            x_origin_max,
        )

        crop_y_min = self.py_random.randint(
            y_origin_min,
            y_origin_max,
        )

        crop_coords = (
            crop_x_min,
            crop_y_min,
            crop_x_min + crop_width,
            crop_y_min + crop_height,
        )

        return {
            "crop_coords": crop_coords,
            "zoom_factor": float(zoom_factor),
        }