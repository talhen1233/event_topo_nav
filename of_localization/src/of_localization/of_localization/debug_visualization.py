from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureDepthDebug:
    """Per-feature debug information for local depth matching."""

    prev_points_xy: np.ndarray
    next_points_xy: np.ndarray
    valid_mask: np.ndarray
    selected_mask: np.ndarray
    matched_depths_m: np.ndarray
    matched_rows: np.ndarray
    matched_cols: np.ndarray
    pointcloud_age_sec: float
    valid_ratio: float


@dataclass(frozen=True)
class VelocityComparisonDebug:
    """Quantities rendered on the debug overlays."""

    local_depth_velocity_xy: np.ndarray | None
    single_height_velocity_xy: np.ndarray | None
    scalar_height_m: float
    single_height_m: float | None


FLOW_COLOR = (0, 210, 80)
MATCH_COLOR = (0, 220, 255)
USED_FEATURE_COLOR = (255, 255, 255)
CENTER_SAMPLE_COLOR = (0, 170, 255)
LOCAL_VELOCITY_COLOR = (255, 255, 0)
SURFACE_EDGE_COLOR = (46, 46, 46)
TEXT_COLOR = (245, 245, 245)
SUBTLE_COLOR = (190, 190, 190)
AXIS_COLOR = (210, 210, 210)
BOX_BORDER_COLOR = (95, 95, 95)
FONT = cv2.FONT_HERSHEY_DUPLEX


def render_depth_debug_canvas(
    gray_image: np.ndarray,
    overlap_bounds_xyxy: tuple[int, int, int, int],
    grid_depths_m: np.ndarray,
    grid_centers_xy: np.ndarray,
    feature_debug: FeatureDepthDebug | None,
    comparison_debug: VelocityComparisonDebug,
    render_scale: float = 2.0,
) -> np.ndarray:
    """Render a thesis-ready figure centered on the undistorted bottom-camera image."""
    del overlap_bounds_xyxy
    scale = max(float(render_scale), 1.0)
    image_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(
        image_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )

    scaled_centers = np.asarray(grid_centers_xy, dtype=np.float32) * scale
    scaled_feature_debug = _scale_feature_debug(feature_debug, scale)

    _draw_depth_grid(canvas, grid_depths_m, scaled_centers, scale)
    if scaled_feature_debug is not None:
        _draw_feature_matches(canvas, scaled_centers, scaled_feature_debug, scale)

    center_u = int(round(canvas.shape[1] * 0.5))
    center_v = int(round(canvas.shape[0] * 0.5))
    _draw_center_sample_marker(canvas, center_u, center_v, scale)
    _draw_surface_inset(canvas, grid_depths_m, scale)
    _draw_velocity_overlay(canvas, comparison_debug, scale)
    _draw_legend_overlay(canvas, comparison_debug, scale)
    return canvas


def _draw_depth_grid(
    image_bgr: np.ndarray,
    grid_depths_m: np.ndarray,
    grid_centers_xy: np.ndarray,
    scale: float,
) -> None:
    finite_depths = grid_depths_m[np.isfinite(grid_depths_m)]
    if finite_depths.size == 0:
        return

    depth_min = float(np.min(finite_depths))
    depth_span = max(float(np.max(finite_depths)) - depth_min, 1e-6)
    point_radius = max(5, int(round(3.8 * scale)))
    ring_radius = max(point_radius + 2, int(round(5.6 * scale)))

    for row_idx in range(grid_depths_m.shape[0]):
        for col_idx in range(grid_depths_m.shape[1]):
            depth = float(grid_depths_m[row_idx, col_idx])
            if not np.isfinite(depth):
                continue
            center = grid_centers_xy[row_idx, col_idx]
            pt = (int(round(center[0])), int(round(center[1])))
            color = _depth_to_bgr(depth, depth_min, depth_span)
            cv2.circle(image_bgr, pt, point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(image_bgr, pt, ring_radius, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_feature_matches(
    image_bgr: np.ndarray,
    grid_centers_xy: np.ndarray,
    feature_debug: FeatureDepthDebug,
    scale: float,
) -> None:
    selected_indices = np.flatnonzero(feature_debug.selected_mask)
    label_stride = max(1, int(np.ceil(len(selected_indices) / 3.0))) if len(selected_indices) else 1
    line_thickness = max(2, int(round(1.3 * scale)))
    end_radius = max(3, int(round(2.8 * scale)))

    for idx, (point_prev, point_next) in enumerate(zip(feature_debug.prev_points_xy, feature_debug.next_points_xy)):
        start_pt = (int(round(point_prev[0])), int(round(point_prev[1])))
        end_pt = (int(round(point_next[0])), int(round(point_next[1])))
        is_valid = bool(feature_debug.valid_mask[idx])
        is_selected = bool(feature_debug.selected_mask[idx])

        flow_color = FLOW_COLOR if is_valid else (95, 95, 95)
        cv2.arrowedLine(
            image_bgr,
            start_pt,
            end_pt,
            flow_color,
            line_thickness + (1 if is_selected else 0),
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.circle(image_bgr, end_pt, end_radius, flow_color, -1, cv2.LINE_AA)

        if not is_valid:
            continue

        row_idx = int(feature_debug.matched_rows[idx])
        col_idx = int(feature_debug.matched_cols[idx])
        if row_idx >= 0 and col_idx >= 0:
            grid_point = grid_centers_xy[row_idx, col_idx]
            match_pt = (int(round(grid_point[0])), int(round(grid_point[1])))
            cv2.line(image_bgr, end_pt, match_pt, MATCH_COLOR, max(1, line_thickness - 1), cv2.LINE_AA)
            cv2.circle(
                image_bgr,
                match_pt,
                max(5, int(round((6.0 if is_selected else 4.5) * scale * 0.5))),
                MATCH_COLOR,
                1,
                cv2.LINE_AA,
            )

        if is_selected:
            cv2.circle(
                image_bgr,
                end_pt,
                max(5, int(round(4.8 * scale))),
                USED_FEATURE_COLOR,
                1,
                cv2.LINE_AA,
            )

    shown = 0
    for idx in selected_indices:
        if shown % label_stride != 0:
            shown += 1
            continue
        shown += 1
        depth_m = float(feature_debug.matched_depths_m[idx])
        if not np.isfinite(depth_m):
            continue
        point_next = feature_debug.next_points_xy[idx]
        anchor_xy = (
            int(round(point_next[0])) + int(round(12 * scale)),
            int(round(point_next[1])) - int(round(10 * scale)),
        )
        _draw_text_tag(image_bgr, f"{depth_m:.2f} m", anchor_xy, scale)


def _draw_center_sample_marker(
    image_bgr: np.ndarray,
    center_u: int,
    center_v: int,
    scale: float,
) -> None:
    radius = max(8, int(round(7.0 * scale)))
    thickness = max(2, int(round(1.6 * scale)))
    cv2.circle(image_bgr, (center_u, center_v), radius, CENTER_SAMPLE_COLOR, thickness, cv2.LINE_AA)
    cv2.line(image_bgr, (center_u - radius, center_v), (center_u + radius, center_v), CENTER_SAMPLE_COLOR, thickness, cv2.LINE_AA)
    cv2.line(image_bgr, (center_u, center_v - radius), (center_u, center_v + radius), CENTER_SAMPLE_COLOR, thickness, cv2.LINE_AA)


def _draw_surface_inset(
    image_bgr: np.ndarray,
    grid_depths_m: np.ndarray,
    scale: float,
) -> None:
    image_h, image_w = image_bgr.shape[:2]
    del image_h, image_w
    top_margin = int(round(6 * scale))
    left_margin = int(round(8 * scale))
    label_reserve = int(round(52 * scale))
    colorbar_w = max(10, int(round(12 * scale)))
    gap = int(round(12 * scale))
    surface_w = int(round(200 * scale))
    surface_h = int(round(110 * scale))

    colorbar_rect = (
        left_margin + label_reserve,
        top_margin + int(round(6 * scale)),
        colorbar_w,
        max(surface_h - int(round(12 * scale)), int(round(60 * scale))),
    )
    surface_rect = (
        colorbar_rect[0] + colorbar_w + gap,
        top_margin,
        surface_w,
        surface_h,
    )

    depth_min, depth_max = _depth_limits(grid_depths_m, None)
    _draw_surface_box(image_bgr, surface_rect, grid_depths_m, depth_min, depth_max, scale)
    _draw_vertical_colorbar(image_bgr, colorbar_rect, depth_min, depth_max, scale, labels_on_left=True)


def _draw_velocity_overlay(
    image_bgr: np.ndarray,
    comparison_debug: VelocityComparisonDebug,
    scale: float,
) -> None:
    image_h, image_w = image_bgr.shape[:2]
    del image_h
    right_margin = int(round(6 * scale))
    top_margin = int(round(6 * scale))
    readout_w = int(round(150 * scale))
    axes_size = int(round(105 * scale))
    gap = int(round(6 * scale))

    total_w = axes_size + gap + readout_w
    axes_rect = (image_w - right_margin - total_w, top_margin, axes_size, axes_size)
    _draw_velocity_axes(image_bgr, axes_rect, comparison_debug, scale)

    readout_x = axes_rect[0] + axes_rect[2] + gap
    readout_y = axes_rect[1] + int(round(22 * scale))
    readout_y = _draw_velocity_readout(
        image_bgr,
        readout_x,
        readout_y,
        "Single height",
        comparison_debug.single_height_velocity_xy,
        CENTER_SAMPLE_COLOR,
        scale,
    )
    _draw_velocity_readout(
        image_bgr,
        readout_x,
        readout_y + int(round(10 * scale)),
        "Local ToF",
        comparison_debug.local_depth_velocity_xy,
        LOCAL_VELOCITY_COLOR,
        scale,
    )


def _draw_velocity_axes(
    image_bgr: np.ndarray,
    rect_xywh: tuple[int, int, int, int],
    comparison_debug: VelocityComparisonDebug,
    scale: float,
) -> None:
    x0, y0, width, height = rect_xywh
    cx = x0 + int(round(width * 0.42))
    cy = y0 + int(round(height * 0.58))
    axis_len = int(round(min(width, height) * 0.42))
    axis_thickness = _thickness(scale, 1.0)
    tip_length = 0.12
    cv2.arrowedLine(
        image_bgr,
        (cx, cy + axis_len // 6),
        (cx, cy - axis_len),
        AXIS_COLOR,
        axis_thickness,
        cv2.LINE_AA,
        tipLength=tip_length,
    )
    cv2.arrowedLine(
        image_bgr,
        (cx - axis_len // 6, cy),
        (cx + axis_len, cy),
        AXIS_COLOR,
        axis_thickness,
        cv2.LINE_AA,
        tipLength=tip_length,
    )

    axis_font = _font_scale(scale, 0.38)
    axis_thickness_text = _thickness(scale, 1.0)
    _put_text(
        image_bgr,
        "+X (forward)",
        (cx + int(round(6 * scale)), cy - axis_len - int(round(6 * scale))),
        axis_font,
        color=AXIS_COLOR,
        thickness=axis_thickness_text,
    )
    _put_text(
        image_bgr,
        "+Y (right)",
        (cx + axis_len - int(round(28 * scale)), cy + int(round(16 * scale))),
        axis_font,
        color=AXIS_COLOR,
        thickness=axis_thickness_text,
    )

    vector_specs = [
        (comparison_debug.single_height_velocity_xy, CENTER_SAMPLE_COLOR),
        (comparison_debug.local_depth_velocity_xy, LOCAL_VELOCITY_COLOR),
    ]
    valid_vectors = [vector for vector, _ in vector_specs if vector is not None]
    max_speed = max((float(np.linalg.norm(vector)) for vector in valid_vectors), default=0.0)
    vector_scale = (axis_len * 0.90) / max(max_speed, 0.12)

    for vector, color in vector_specs:
        _draw_body_vector(image_bgr, (cx, cy), vector, vector_scale, color, scale)


def _draw_velocity_readout(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    label: str,
    vector_xy: np.ndarray | None,
    color_bgr: tuple[int, int, int],
    scale: float,
) -> int:
    dot_radius = max(5, int(round(4.0 * scale)))
    label_font = _font_scale(scale, 0.42)
    value_font = _font_scale(scale, 0.52)
    label_thickness = _thickness(scale, 1.1)
    value_thickness = _thickness(scale, 1.4)

    cv2.circle(image_bgr, (x + dot_radius, y), dot_radius, color_bgr, -1, cv2.LINE_AA)
    _put_text(
        image_bgr,
        label,
        (x + 3 * dot_radius, y + int(round(4 * scale))),
        label_font,
        color=TEXT_COLOR,
        thickness=label_thickness,
    )
    value_y = y + int(round(22 * scale))
    if vector_xy is None:
        text = "unavailable"
    else:
        speed = float(np.linalg.norm(vector_xy))
        text = f"{speed:.3f} m/s"
    _put_text(
        image_bgr,
        text,
        (x + 3 * dot_radius, value_y + int(round(4 * scale))),
        value_font,
        thickness=value_thickness,
    )
    return value_y + int(round(18 * scale))


def _draw_legend_overlay(
    image_bgr: np.ndarray,
    comparison_debug: VelocityComparisonDebug,
    scale: float,
) -> None:
    image_h, image_w = image_bgr.shape[:2]
    margin = int(round(16 * scale))
    row_y = image_h - margin

    if comparison_debug.single_height_m is not None:
        sample_text = f"Single-height sample ({comparison_debug.single_height_m:.2f} m)"
    else:
        sample_text = "Single-height sample"

    entries = (
        ("flow", "Tracked flow", FLOW_COLOR),
        ("link", "Feature-ToF match", MATCH_COLOR),
        ("ring", sample_text, CENTER_SAMPLE_COLOR),
    )

    font_scale = _font_scale(scale, 0.44)
    text_thickness = _thickness(scale, 1.2)
    glyph_length = int(round(34 * scale))
    label_gap = int(round(10 * scale))
    entry_gap = int(round(28 * scale))

    entry_widths: list[int] = []
    for _, label, _color in entries:
        text_size, _ = cv2.getTextSize(label, FONT, font_scale, text_thickness)
        entry_widths.append(glyph_length + label_gap + text_size[0])

    total_w = sum(entry_widths) + entry_gap * (len(entries) - 1)
    x_cursor = max(margin, (image_w - total_w) // 2)

    for (kind, label, color), width in zip(entries, entry_widths):
        _draw_legend_row_entry(
            image_bgr,
            x_cursor,
            row_y,
            kind,
            label,
            color,
            font_scale,
            text_thickness,
            glyph_length,
            label_gap,
            scale,
        )
        x_cursor += width + entry_gap


def _draw_legend_row_entry(
    image_bgr: np.ndarray,
    x: int,
    baseline_y: int,
    kind: str,
    label: str,
    color: tuple[int, int, int],
    font_scale: float,
    text_thickness: int,
    glyph_length: int,
    label_gap: int,
    scale: float,
) -> None:
    glyph_center_y = baseline_y - int(round(6 * scale))
    start_pt = (x, glyph_center_y)
    end_pt = (x + glyph_length, glyph_center_y)
    if kind == "flow":
        cv2.arrowedLine(
            image_bgr,
            start_pt,
            end_pt,
            color,
            _thickness(scale, 1.6),
            cv2.LINE_AA,
            tipLength=0.22,
        )
        cv2.circle(image_bgr, end_pt, max(4, int(round(3.2 * scale))), color, -1, cv2.LINE_AA)
    elif kind == "link":
        cv2.line(image_bgr, start_pt, end_pt, color, _thickness(scale, 1.6), cv2.LINE_AA)
        cv2.circle(image_bgr, end_pt, max(4, int(round(3.8 * scale))), color, 1, cv2.LINE_AA)
    else:
        cv2.circle(
            image_bgr,
            (x + glyph_length // 2, glyph_center_y),
            max(5, int(round(4.8 * scale))),
            color,
            max(1, _thickness(scale, 1.1)),
            cv2.LINE_AA,
        )

    _put_text(
        image_bgr,
        label,
        (x + glyph_length + label_gap, baseline_y),
        font_scale,
        thickness=text_thickness,
    )


def _draw_text_tag(
    image_bgr: np.ndarray,
    text: str,
    anchor_xy: tuple[int, int],
    scale: float,
) -> None:
    font_scale = _font_scale(scale, 0.40)
    thickness = _thickness(scale, 1.1)
    _put_text(
        image_bgr,
        text,
        anchor_xy,
        font_scale,
        thickness=thickness,
    )


def _draw_surface_box(
    image_bgr: np.ndarray,
    rect_xywh: tuple[int, int, int, int],
    height_grid_m: np.ndarray,
    depth_min: float,
    depth_max: float,
    scale: float,
) -> None:
    x0, y0, width, height = rect_xywh
    if width <= 0 or height <= 0:
        return

    inner_pad = int(round(10 * scale))
    origin = (
        x0 + width // 2,
        y0 + int(round(height * 0.34)),
    )
    tile_w = max(7.0, float(width - 2 * inner_pad) / max(height_grid_m.shape[1] + height_grid_m.shape[0] - 1, 2))
    tile_h = tile_w * 0.54
    height_span = max(depth_max - depth_min, 0.08)
    height_scale = (height - 2 * inner_pad) * 0.24 / height_span

    vertices = _surface_vertices(height_grid_m, origin, tile_w, tile_h, height_scale, depth_min)
    draw_order: list[tuple[int, int, int]] = []
    for row_idx in range(height_grid_m.shape[0] - 1):
        for col_idx in range(height_grid_m.shape[1] - 1):
            draw_order.append((row_idx + col_idx, row_idx, col_idx))
    draw_order.sort()

    for _, row_idx, col_idx in draw_order:
        quad = np.asarray(
            [
                vertices[row_idx, col_idx],
                vertices[row_idx, col_idx + 1],
                vertices[row_idx + 1, col_idx + 1],
                vertices[row_idx + 1, col_idx],
            ],
            dtype=np.int32,
        )
        cell_values = height_grid_m[row_idx:row_idx + 2, col_idx:col_idx + 2]
        finite_values = cell_values[np.isfinite(cell_values)]
        if finite_values.size == 0:
            fill_color = (55, 55, 55)
        else:
            fill_color = _depth_to_bgr(float(np.mean(finite_values)), depth_min, max(depth_max - depth_min, 1e-6))
        cv2.fillConvexPoly(image_bgr, quad, fill_color, cv2.LINE_AA)
        cv2.polylines(image_bgr, [quad], True, SURFACE_EDGE_COLOR, 1, cv2.LINE_AA)


def _draw_vertical_colorbar(
    image_bgr: np.ndarray,
    rect_xywh: tuple[int, int, int, int],
    depth_min: float,
    depth_max: float,
    scale: float,
    *,
    labels_on_left: bool = False,
) -> None:
    x0, y0, width, height = rect_xywh
    if width <= 0 or height <= 0:
        return
    for offset in range(height):
        ratio = 1.0 - offset / max(height - 1, 1)
        depth = depth_min + ratio * (depth_max - depth_min)
        color = _depth_to_bgr(depth, depth_min, max(depth_max - depth_min, 1e-6))
        cv2.line(image_bgr, (x0, y0 + offset), (x0 + width, y0 + offset), color, 1)
    cv2.rectangle(image_bgr, (x0, y0), (x0 + width, y0 + height), BOX_BORDER_COLOR, 1, cv2.LINE_AA)

    tick_font = _font_scale(scale, 0.38)
    tick_thickness = _thickness(scale, 1.0)
    min_label = f"{depth_min:.2f} m"
    max_label = f"{depth_max:.2f} m"
    min_text_h = _text_height(tick_font, tick_thickness)
    anchor = "right" if labels_on_left else "left"
    if labels_on_left:
        text_x = x0 - int(round(6 * scale))
    else:
        text_x = x0 + width + int(round(6 * scale))
    _put_text(
        image_bgr,
        max_label,
        (text_x, y0 + min_text_h),
        tick_font,
        color=SUBTLE_COLOR,
        thickness=tick_thickness,
        anchor=anchor,
    )
    _put_text(
        image_bgr,
        min_label,
        (text_x, y0 + height),
        tick_font,
        color=SUBTLE_COLOR,
        thickness=tick_thickness,
        anchor=anchor,
    )


def _draw_body_vector(
    image_bgr: np.ndarray,
    origin_xy: tuple[int, int],
    vector_xy: np.ndarray | None,
    scale_px: float,
    color_bgr: tuple[int, int, int],
    scale: float,
) -> None:
    if vector_xy is None:
        return
    dx = int(round(float(vector_xy[1]) * scale_px))
    dy = int(round(-float(vector_xy[0]) * scale_px))
    end_pt = (origin_xy[0] + dx, origin_xy[1] + dy)
    cv2.arrowedLine(
        image_bgr,
        origin_xy,
        end_pt,
        color_bgr,
        _thickness(scale, 2.2),
        cv2.LINE_AA,
        tipLength=0.22,
    )
    cv2.circle(image_bgr, end_pt, max(4, int(round(3.6 * scale))), color_bgr, -1, cv2.LINE_AA)


def _put_text(
    image_bgr: np.ndarray,
    text: str,
    origin_xy: tuple[int, int],
    font_scale: float,
    *,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 2,
    anchor: str = "left",
) -> None:
    text_size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = origin_xy
    if anchor == "right":
        x -= text_size[0]
    elif anchor == "center":
        x -= text_size[0] // 2

    cv2.putText(
        image_bgr,
        text,
        (x, y),
        FONT,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        text,
        (x, y),
        FONT,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _text_height(font_scale: float, thickness: int) -> int:
    size, _ = cv2.getTextSize("Agjpy", FONT, font_scale, thickness)
    return size[1]


def _font_scale(scale: float, base: float) -> float:
    return max(base, base * scale * 0.72)


def _thickness(scale: float, base: float) -> int:
    return max(1, int(round(base * scale * 0.9)))


def _depth_to_bgr(depth_m: float, depth_min: float, depth_span: float) -> tuple[int, int, int]:
    normalized = 1.0 - np.clip((depth_m - depth_min) / depth_span, 0.0, 1.0)
    color = cv2.applyColorMap(
        np.asarray([[int(round(255.0 * normalized))]], dtype=np.uint8),
        cv2.COLORMAP_TURBO,
    )[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def _scale_feature_debug(
    feature_debug: FeatureDepthDebug | None,
    scale: float,
) -> FeatureDepthDebug | None:
    if feature_debug is None:
        return None
    return FeatureDepthDebug(
        prev_points_xy=feature_debug.prev_points_xy * scale,
        next_points_xy=feature_debug.next_points_xy * scale,
        valid_mask=feature_debug.valid_mask,
        selected_mask=feature_debug.selected_mask,
        matched_depths_m=feature_debug.matched_depths_m,
        matched_rows=feature_debug.matched_rows,
        matched_cols=feature_debug.matched_cols,
        pointcloud_age_sec=feature_debug.pointcloud_age_sec,
        valid_ratio=feature_debug.valid_ratio,
    )


def _surface_vertices(
    height_grid_m: np.ndarray,
    origin_xy: tuple[int, int],
    tile_w: float,
    tile_h: float,
    height_scale: float,
    depth_min: float,
) -> np.ndarray:
    rows, cols = height_grid_m.shape
    vertices = np.zeros((rows + 1, cols + 1, 2), dtype=np.float32)
    row_values = _edge_extended(height_grid_m, axis=0)
    col_values = _edge_extended(row_values, axis=1)
    for row_idx in range(rows + 1):
        for col_idx in range(cols + 1):
            depth = float(col_values[row_idx, col_idx])
            x = origin_xy[0] + (col_idx - row_idx) * tile_w
            y = origin_xy[1] + (col_idx + row_idx) * tile_h * 0.5 + (depth - depth_min) * height_scale
            vertices[row_idx, col_idx] = (x, y)
    return vertices


def _edge_extended(values: np.ndarray, axis: int) -> np.ndarray:
    fill_value = np.nanmedian(values) if np.isfinite(values).any() else 0.0
    filled = np.where(np.isfinite(values), values, fill_value)
    if axis == 0:
        output = np.zeros((values.shape[0] + 1, values.shape[1]), dtype=np.float32)
        output[0] = filled[0]
        output[-1] = filled[-1]
        output[1:-1] = 0.5 * (filled[:-1] + filled[1:])
        return output
    output = np.zeros((values.shape[0], values.shape[1] + 1), dtype=np.float32)
    output[:, 0] = filled[:, 0]
    output[:, -1] = filled[:, -1]
    output[:, 1:-1] = 0.5 * (filled[:, :-1] + filled[:, 1:])
    return output


def _depth_limits(
    grid_depths_m: np.ndarray,
    fallback_depth_m: float | None,
) -> tuple[float, float]:
    finite_depths = grid_depths_m[np.isfinite(grid_depths_m)]
    if finite_depths.size == 0:
        base_depth = float(fallback_depth_m) if fallback_depth_m is not None else 1.0
        return base_depth, base_depth + 1e-3

    depth_min = float(np.min(finite_depths))
    depth_max = float(np.max(finite_depths))
    if fallback_depth_m is not None:
        depth_min = min(depth_min, float(fallback_depth_m))
        depth_max = max(depth_max, float(fallback_depth_m))
    if depth_max <= depth_min:
        depth_max = depth_min + 1e-3
    return depth_min, depth_max
