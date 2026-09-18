#!/usr/bin/env python3

"""Calibrate a standard pinhole camera and emit a ROS camera_info YAML file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationInputs:
    """Inputs required to run pinhole camera calibration."""

    images_dir: Path
    output_file: Path
    checkerboard_cols: int
    checkerboard_rows: int
    square_size_m: float
    camera_name: str


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration outputs written to the ROS camera_info YAML file."""

    image_width: int
    image_height: int
    camera_name: str
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rectification_matrix: np.ndarray
    projection_matrix: np.ndarray
    rms_error: float
    used_image_count: int


def _parse_args(argv: Sequence[str] | None) -> CalibrationInputs:
    """Parse command-line arguments for the calibration run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        default=(
            "/home/talhen/docker_ws/tiny_drone/"
            "tiny_drone/config/usb_cam_driver/radtan/camera_bottom_vga_images"
        ),
        help="Directory containing chessboard images.",
    )
    parser.add_argument(
        "--output-file",
        default=(
            "/home/talhen/docker_ws/tiny_drone/"
            "tiny_drone/config/usb_cam_driver/radtan/camera_info_camera_bottom_vga.yaml"
        ),
        help="Output ROS camera_info YAML path.",
    )
    parser.add_argument(
        "--checkerboard-cols",
        type=int,
        default=10,
        help="Number of inner corners along the checkerboard width.",
    )
    parser.add_argument(
        "--checkerboard-rows",
        type=int,
        default=7,
        help="Number of inner corners along the checkerboard height.",
    )
    parser.add_argument(
        "--square-size-m",
        type=float,
        default=0.025,
        help="Checkerboard square size in meters.",
    )
    parser.add_argument(
        "--camera-name",
        default="camera_bottom",
        help="Camera name stored in the ROS camera_info YAML file.",
    )
    args = parser.parse_args(argv)
    return CalibrationInputs(
        images_dir=Path(args.images_dir).expanduser(),
        output_file=Path(args.output_file).expanduser(),
        checkerboard_cols=args.checkerboard_cols,
        checkerboard_rows=args.checkerboard_rows,
        square_size_m=args.square_size_m,
        camera_name=args.camera_name,
    )


def _build_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    """Create the ideal 3D checkerboard points for one calibration frame."""
    object_points = np.zeros((cols * rows, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points *= square_size_m
    return object_points


def _collect_calibration_points(
    inputs: CalibrationInputs,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Collect object points and image points from all usable images."""
    image_paths = sorted(
        path for path in inputs.images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {inputs.images_dir}")

    checkerboard_shape = (inputs.checkerboard_cols, inputs.checkerboard_rows)
    object_point_template = _build_object_points(
        inputs.checkerboard_cols,
        inputs.checkerboard_rows,
        inputs.square_size_m,
    )
    corner_refinement_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-4,
    )
    detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (grayscale_image.shape[1], grayscale_image.shape[0])
        elif image_size != (grayscale_image.shape[1], grayscale_image.shape[0]):
            print(f"Skipping {image_path}: inconsistent resolution.")
            continue

        found, corners = cv2.findChessboardCorners(grayscale_image, checkerboard_shape, detection_flags)
        if not found:
            print(f"Skipping {image_path}: checkerboard not found.")
            continue

        refined_corners = cv2.cornerSubPix(
            grayscale_image,
            corners,
            (3, 3),
            (-1, -1),
            corner_refinement_criteria,
        )
        object_points.append(object_point_template.copy())
        image_points.append(refined_corners)

    if image_size is None or len(object_points) < 8:
        raise RuntimeError(
            "Not enough valid calibration images. Capture at least 8-12 well-spread checkerboard views."
        )

    return object_points, image_points, image_size


def _run_calibration(inputs: CalibrationInputs) -> CalibrationResult:
    """Calibrate the camera using OpenCV's pinhole model."""
    object_points, image_points, image_size = _collect_calibration_points(inputs)
    rms_error, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    rectification_matrix = np.eye(3, dtype=np.float64)
    projection_matrix = np.hstack((camera_matrix, np.zeros((3, 1), dtype=np.float64)))
    return CalibrationResult(
        image_width=image_size[0],
        image_height=image_size[1],
        camera_name=inputs.camera_name,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients.reshape(1, -1),
        rectification_matrix=rectification_matrix,
        projection_matrix=projection_matrix,
        rms_error=float(rms_error),
        used_image_count=len(object_points),
    )


def _format_matrix_data(matrix: np.ndarray) -> str:
    """Serialize matrix data in ROS camera_info YAML list format."""
    flattened = matrix.reshape(-1)
    return ", ".join(f"{float(value):.12f}" for value in flattened)


def _write_camera_info_yaml(result: CalibrationResult, output_file: Path) -> None:
    """Write the calibration result in ROS camera_info YAML format."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = f"""image_width: {result.image_width}
image_height: {result.image_height}
camera_name: {result.camera_name}
camera_matrix:
  rows: 3
  cols: 3
  data: [{_format_matrix_data(result.camera_matrix)}]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: {result.distortion_coefficients.shape[1]}
  data: [{_format_matrix_data(result.distortion_coefficients)}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [{_format_matrix_data(result.rectification_matrix)}]
projection_matrix:
  rows: 3
  cols: 4
  data: [{_format_matrix_data(result.projection_matrix)}]
"""
    output_file.write_text(yaml_text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the pinhole calibration pipeline."""
    inputs = _parse_args(argv)
    result = _run_calibration(inputs)
    _write_camera_info_yaml(result, inputs.output_file)
    print(f"Used {result.used_image_count} calibration images")
    print(f"RMS reprojection error: {result.rms_error:.6f}")
    print(f"Wrote camera_info YAML to {inputs.output_file}")


if __name__ == "__main__":
    main()
