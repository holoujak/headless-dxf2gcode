#!/usr/bin/env python3
"""DXF -> G-code converter.

Features:
 - config.yaml support (defaults)
 - optional preamble/postamble insertion
 - origin shift to lower-left
 - Z safe/work motion (rapid travel + plunge feed)
 - numerical tool radius compensation using shapely.buffer
   (keeps contours continuous)
 - groups contiguous geometry into contours and mills each contour
   in one continuous pass
 - optional plotting of original vs compensated shapes
 - optional line numbering (N numbers, increment configurable)

Dependencies:
 - ezdxf
 - shapely
 - pyyaml
 - matplotlib (optional, for --plot)

On Arch Linux prefer installing system packages where available, or use a venv.
"""

import argparse
import math
import os
import sys
from typing import List, Tuple

try:
    import ezdxf
except Exception:
    print(
        "Missing dependency: ezdxf. Install with: "
        "sudo pacman -S python-ezdxf or pip install ezdxf",
        file=sys.stderr,
    )
    raise

try:
    from shapely.affinity import translate
    from shapely.geometry import LineString, MultiPolygon, Polygon
    from shapely.ops import unary_union
except Exception:
    print(
        "Missing dependency: shapely. Install with: "
        "sudo pacman -S python-shapely or pip install shapely",
        file=sys.stderr,
    )
    raise

try:
    import yaml
except Exception:
    print(
        "Missing dependency: pyyaml. Install with: "
        "sudo pacman -S python-yaml or pip install pyyaml",
        file=sys.stderr,
    )
    raise

try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

# ------------------------------
# Utilities
# ------------------------------


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to read config {path}: {e}")
            return {}
    return {}


def insert_file_content(out_file, path: str):
    """Insert content from file into output stream."""
    if not path:
        return
    if not os.path.exists(path):
        out_file.write(f"( WARNING: snippet file not found: {path} )\n")
        return
    with open(path, "r", encoding="utf-8") as f:
        out_file.write(f"( --- BEGIN FILE: {path} --- )\n")
        out_file.write(f.read())
        out_file.write(f"\n( --- END FILE: {path} --- )\n")


# ------------------------------
# DXF extraction
# ------------------------------


def extract_geometries(dxf_path: str) -> List:
    """Return list of shapely geometries representing continuous entities."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    geoms = []
    for e in msp:
        et = e.dxftype()
        if et == "LINE":
            x1, y1 = float(e.dxf.start[0]), float(e.dxf.start[1])
            x2, y2 = float(e.dxf.end[0]), float(e.dxf.end[1])
            geoms.append(LineString([(x1, y1), (x2, y2)]))
        elif et in ("LWPOLYLINE", "POLYLINE"):
            pts = []
            try:
                for p in e.get_points():
                    pts.append((float(p[0]), float(p[1])))
            except Exception:
                # fallback for old API
                for p in e.points():
                    pts.append((float(p[0]), float(p[1])))
            if len(pts) < 2:
                continue
            # If polyline is closed, create Polygon, else LineString
            is_closed = getattr(e, "closed", False) or (pts[0] == pts[-1])
            if is_closed:
                geoms.append(Polygon(pts))
            else:
                geoms.append(LineString(pts))
        elif et == "CIRCLE":
            cx, cy = float(e.dxf.center[0]), float(e.dxf.center[1])
            r = float(e.dxf.radius)
            geoms.append(
                Polygon(
                    LineString(
                        [
                            (cx + math.cos(a) * r, cy + math.sin(a) * r)
                            for a in [i * 2 * math.pi / 64 for i in range(64)]
                        ]
                    ).coords
                )
            )
        elif et == "ARC":
            # approximate arc as LineString
            cx, cy = float(e.dxf.center[0]), float(e.dxf.center[1])
            r = float(e.dxf.radius)
            start_ang, end_ang = float(e.dxf.start_angle), float(e.dxf.end_angle)
            # convert to radians
            sa = math.radians(start_ang)
            ea = math.radians(end_ang)
            # handle wrap
            if ea <= sa:
                ea += 2 * math.pi
            segments = max(6, int((ea - sa) / (math.pi / 16)))
            pts = [
                (
                    cx + math.cos(sa + t * (ea - sa) / segments) * r,
                    cy + math.sin(sa + t * (ea - sa) / segments) * r,
                )
                for t in range(segments + 1)
            ]
            geoms.append(LineString(pts))
        else:
            # ignore other entities for now
            continue
    return geoms


# ------------------------------
# Helper: shift geometries
# ------------------------------


def compute_bbox_offset(geoms: List, origin_lower_left: bool) -> Tuple[float, float]:
    """Compute offset to shift origin to lower-left corner."""
    if not origin_lower_left:
        return 0.0, 0.0
    all_bounds = [g.bounds for g in geoms if not g.is_empty]
    if not all_bounds:
        return 0.0, 0.0
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    return minx, miny


def shift_geom(geom, dx: float, dy: float):
    """Shift geometry by specified offset."""
    return (
        type(geom)([(x - dx, y - dy) for x, y in geom.coords])
        if isinstance(geom, LineString)
        else geom.translate(-dx, -dy)
    )


# Note: shapely's geometry objects don't have a generic 'translate' method


def shift_geometries(geoms: List, dx: float, dy: float) -> List:
    """Shift list of geometries by specified offset."""
    return [translate(g, xoff=-dx, yoff=-dy) for g in geoms]


# ------------------------------
# Tool compensation using buffer
# ------------------------------


def compensate_geometries_with_buffer(
    geoms: List,
    tool_radius: float,
    tool_side: str,
    buffer_resolution: int = 16,
    join_style: int = 1,
) -> List[Polygon]:
    """Return list of polygons representing compensated contours.

    join_style: 1=round, 2=mitre, 3=bevel (shapely conventions differ:
    actually 1 round, 2 mitre, 3 bevel)
    """
    if tool_radius == 0:
        return geoms
    sign = 1 if (tool_side == "left") else -1
    buffered = []
    for g in geoms:
        try:
            b = g.buffer(
                sign * tool_radius, resolution=buffer_resolution, join_style=join_style
            )
            if b.is_empty:
                continue
            if isinstance(b, (Polygon, MultiPolygon)):
                # unify into polygons
                if isinstance(b, Polygon):
                    buffered.append(b)
                else:
                    for p in b.geoms:
                        buffered.append(p)
        except Exception as e:
            print(f"Buffer failed for geometry: {e}")
    # optional union to merge overlapping buffers
    if not buffered:
        return []
    try:
        merged = unary_union(buffered)
        polys = []
        if isinstance(merged, Polygon):
            polys = [merged]
        else:
            for p in merged.geoms:
                if isinstance(p, Polygon):
                    polys.append(p)
        return polys
    except Exception:
        # fallback
        return buffered


# ------------------------------
# G-code optimization functions
# ------------------------------


def merge_close_points(
    coords: List[Tuple[float, float]], tolerance: float = 0.01
) -> List[Tuple[float, float]]:
    """Merge points that are closer than tolerance to reduce micro-movements."""
    if len(coords) <= 1:
        return coords

    merged = [coords[0]]

    for i in range(1, len(coords)):
        current = coords[i]
        previous = merged[-1]

        # Calculate distance between current and previous point
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Only add point if it's far enough from the previous one
        if distance >= tolerance:
            merged.append(current)

    return merged


def remove_short_segments(
    coords: List[Tuple[float, float]], min_length: float = 0.05
) -> List[Tuple[float, float]]:
    """Remove segments shorter than min_length to avoid jerky movements."""
    if len(coords) <= 2:
        return coords

    filtered = [coords[0]]  # Always keep first point

    for i in range(1, len(coords)):
        current = coords[i]
        previous = filtered[-1]

        # Calculate segment length
        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        length = math.sqrt(dx * dx + dy * dy)

        # Keep point if segment is long enough or if it's the last point
        if length >= min_length or i == len(coords) - 1:
            filtered.append(current)

    return filtered


def douglas_peucker_simplify(
    coords: List[Tuple[float, float]], tolerance: float = 0.02
) -> List[Tuple[float, float]]:
    """Simplify polygon using Douglas-Peucker algorithm to reduce number of points."""
    if len(coords) <= 2:
        return coords

    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # If line is actually a point
        if x1 == x2 and y1 == y2:
            return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

        # Calculate perpendicular distance
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        return numerator / denominator if denominator > 0 else 0

    def douglas_peucker_recursive(points, start_idx, end_idx):
        """Recursive Douglas-Peucker implementation."""
        if end_idx <= start_idx + 1:
            return [start_idx, end_idx] if end_idx > start_idx else [start_idx]

        max_distance = 0
        max_index = start_idx

        # Find point with maximum distance from line
        for i in range(start_idx + 1, end_idx):
            distance = perpendicular_distance(
                points[i], points[start_idx], points[end_idx]
            )
            if distance > max_distance:
                max_distance = distance
                max_index = i

        # If max distance is greater than tolerance, recursively simplify
        if max_distance > tolerance:
            # Recursively simplify both parts
            left_result = douglas_peucker_recursive(points, start_idx, max_index)
            right_result = douglas_peucker_recursive(points, max_index, end_idx)

            # Combine results (remove duplicate middle point)
            return left_result[:-1] + right_result
        else:
            # All points between start and end are within tolerance
            return [start_idx, end_idx]

    # Apply Douglas-Peucker algorithm
    keep_indices = douglas_peucker_recursive(coords, 0, len(coords) - 1)

    # Return simplified coordinates
    return [coords[i] for i in keep_indices]


def calculate_segment_feedrate(
    length: float, base_feed: float, min_feed_ratio: float = 0.3
) -> float:
    """Calculate adaptive feedrate based on segment length."""
    # Shorter segments get slower feedrate to reduce machine stress
    if length < 1.0:  # Very short segments
        return base_feed * min_feed_ratio
    elif length < 5.0:  # Short segments
        ratio = min_feed_ratio + (1.0 - min_feed_ratio) * (length / 5.0)
        return base_feed * ratio
    else:  # Normal and long segments
        return base_feed


# ------------------------------
# G-code generation
# ------------------------------


def format_line_num(n: int) -> str:
    """Format line number for G-code."""
    return f"N{n:04d} "


def generate_gcode_from_polygons(
    polygons: List[Polygon],
    cfg: dict,
    enable_line_numbers: bool = False,
    line_inc: int = 10,
) -> List[str]:
    """Generate G-code lines from polygon shapes using configuration."""
    lines = []
    ln = 10

    def emit(s: str):
        nonlocal ln
        if enable_line_numbers:
            lines.append(f"{format_line_num(ln)}{s}")
            ln += line_inc
        else:
            lines.append(s)

    # header
    emit("(Generated by headless-dxf2gcode with optimizations)")

    # Basic settings
    z_safe = cfg.get("z_safe_height", 5)
    z_work = cfg.get("z_work_height", -1)
    z_approach = cfg.get("z_approach_feed", 300)
    z_travel = cfg.get("z_travel_feed", 1500)
    base_feed = cfg.get("feed_rate", 1000)

    # Optimization settings
    opt_cfg = cfg.get("optimization", {})
    optimization_enabled = opt_cfg.get("enable", True)
    merge_tolerance = opt_cfg.get("merge_tolerance", 0.01)
    min_segment_length = opt_cfg.get("min_segment_length", 0.05)
    douglas_peucker_tolerance = opt_cfg.get("douglas_peucker_tolerance", 0.02)
    adaptive_feedrate = opt_cfg.get("adaptive_feedrate", True)
    min_feed_ratio = opt_cfg.get("min_feed_ratio", 0.3)
    sort_by_size = opt_cfg.get("sort_by_size", True)

    if optimization_enabled:
        emit(
            f"(Optimization enabled: merge_tol={merge_tolerance}, "
            f"min_seg={min_segment_length}, dp_tol={douglas_peucker_tolerance})"
        )

    # Display milling order info (polygons are already sorted in main())
    if sort_by_size:
        emit(f"(Processing {len(polygons)} contours from smallest to largest)")
    else:
        emit(f"(Processing {len(polygons)} contours in original order)")

    for i, poly in enumerate(polygons, 1):
        emit(f"(Contour {i}: area = {poly.area:.2f} mm²)")

        # We'll mill polygon exterior as one continuous contour
        exterior = list(poly.exterior.coords)
        if len(exterior) < 2:
            continue

        # Apply optimizations if enabled
        if optimization_enabled:
            # Step 1: Merge close points
            exterior = merge_close_points(exterior, merge_tolerance)

            # Step 2: Remove very short segments
            exterior = remove_short_segments(exterior, min_segment_length)

            # Step 3: Simplify using Douglas-Peucker
            exterior = douglas_peucker_simplify(exterior, douglas_peucker_tolerance)

            if len(exterior) < 2:
                continue

        # Move to safe height and XY start
        x0, y0 = exterior[0]
        emit(f"G0 Z{z_safe:.3f} F{z_travel}")
        emit(f"G0 X{x0:.4f} Y{y0:.4f} F{z_travel}")
        emit(f"G1 Z{z_work:.4f} F{z_approach}")

        # Follow exterior points with adaptive feedrate
        prev_x, prev_y = x0, y0

        for x, y in exterior[1:]:
            # Calculate feedrate based on segment length if adaptive mode is enabled
            if optimization_enabled and adaptive_feedrate:
                segment_length = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                current_feed = calculate_segment_feedrate(
                    segment_length, base_feed, min_feed_ratio
                )
                emit(f"G1 X{x:.4f} Y{y:.4f} F{current_feed:.0f}")
            else:
                emit(f"G1 X{x:.4f} Y{y:.4f} F{base_feed}")

            prev_x, prev_y = x, y

        # Retract
        emit(f"G0 Z{z_safe:.3f} F{z_travel}")

    # footer
    if optimization_enabled:
        emit("(Optimization complete)")
    return lines


# ------------------------------
# Plotting
# ------------------------------


def plot_polygons(original_geoms, compensated_polys, sorted_polys=None):
    """Plot original and compensated geometries with milling order numbers."""
    if not _HAS_MATPLOTLIB:
        print("matplotlib not installed; cannot plot")
        return
    fig, ax = plt.subplots(figsize=(10, 8))

    # original
    for g in original_geoms:
        try:
            xs, ys = g.xy if hasattr(g, "xy") else zip(*list(g.exterior.coords))
        except Exception:
            try:
                xs, ys = zip(*list(g.coords))
            except Exception:
                continue
        ax.plot(
            xs,
            ys,
            color="blue",
            linewidth=0.8,
            label=(
                "Original"
                if "Original" not in ax.get_legend_handles_labels()[1]
                else ""
            ),
        )

    # compensated with milling order numbers
    for i, p in enumerate(compensated_polys):
        xs, ys = zip(*list(p.exterior.coords))
        ax.plot(
            xs,
            ys,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=(
                "Compensated"
                if "Compensated" not in ax.get_legend_handles_labels()[1]
                else ""
            ),
        )

        # Add milling order number at polygon centroid
        centroid = p.centroid
        order_num = i + 1  # 1-based numbering
        ax.annotate(
            str(order_num),
            xy=(centroid.x, centroid.y),
            xytext=(centroid.x, centroid.y),
            fontsize=12,
            fontweight="bold",
            color="red",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="circle,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8
            ),
        )

    # origin marker
    ax.plot(0, 0, marker="+", color="black", markersize=8)
    ax.annotate("Origin", xy=(0, 0), xytext=(5, 5), fontsize=8)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title("Milling Visualization")
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.show()


# ------------------------------
# Main
# ------------------------------


def main():
    """Convert DXF to G-code with continuous contours and tool offset."""
    parser = argparse.ArgumentParser(
        description="DXF -> G-code with continuous contours and numerical tool offset"
    )
    parser.add_argument("input", help="Input DXF file")
    parser.add_argument("output", help="Output G-code file")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--origin-lower-left",
        action="store_true",
        help="Shift origin to lower-left corner (overrides config)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show visualization of original and compensated shapes",
    )
    parser.add_argument(
        "--line-numbers",
        action="store_true",
        help="Enable N-line numbering (overrides config)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable G-code optimizations (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.origin_lower_left:
        cfg["origin_lower_left"] = True
    if args.line_numbers:
        cfg["line_numbers"] = True
    if args.no_optimize:
        if "optimization" not in cfg:
            cfg["optimization"] = {}
        cfg["optimization"]["enable"] = False

    # extract geometries
    geoms = extract_geometries(args.input)
    if not geoms:
        print("No supported geometry found in DXF")
        return

    # compute bbox offset
    minx, miny = compute_bbox_offset(geoms, cfg.get("origin_lower_left", False))

    # shift all geometries the same way
    geoms_shifted = shift_geometries(geoms, minx, miny)

    # compensate using shapely.buffer to keep contours continuous
    tool_radius = float(cfg.get("tool_radius", 0) or 0)
    tool_side = cfg.get("tool_side", "left")
    buffer_resolution = int(cfg.get("buffer_resolution", 16))
    join_style = int(cfg.get("join_style", 1))

    if tool_radius != 0:
        compensated_polys = compensate_geometries_with_buffer(
            geoms_shifted,
            tool_radius,
            tool_side,
            buffer_resolution,
            join_style,
        )
    else:
        # if no compensation requested, convert geoms_shifted into polygons
        # by buffering with tiny epsilon and taking their outlines
        compensated_polys = []
        for g in geoms_shifted:
            if isinstance(g, Polygon):
                compensated_polys.append(g)
            else:
                # create a very small buffer to get a polygon-like contour
                # for milling along geometry
                p = g.buffer(0.0001)
                if not p.is_empty:
                    if isinstance(p, Polygon):
                        compensated_polys.append(p)
                    else:
                        for sub in p.geoms:
                            if isinstance(sub, Polygon):
                                compensated_polys.append(sub)

    # Sort polygons by area if enabled (for consistent milling order)
    opt_cfg = cfg.get("optimization", {})
    sort_by_size = opt_cfg.get("sort_by_size", True)

    if sort_by_size:
        compensated_polys_sorted = sorted(compensated_polys, key=lambda p: p.area)
    else:
        compensated_polys_sorted = compensated_polys

    # generate gcode lines
    enable_ln = bool(cfg.get("line_numbers", False))
    line_inc = int(cfg.get("line_number_increment", 10))

    # start with optional preamble content
    header_lines = []
    if cfg.get("preamble"):
        if os.path.exists(cfg["preamble"]):
            with open(cfg["preamble"], "r", encoding="utf-8") as f:
                header_lines.extend([ln.rstrip("\n") for ln in f.readlines()])
        else:
            header_lines.append(
                f"(WARNING: preamble file not found: {cfg['preamble']})"
            )

    footer_lines = []
    if cfg.get("postamble"):
        if os.path.exists(cfg["postamble"]):
            with open(cfg["postamble"], "r", encoding="utf-8") as f:
                footer_lines.extend([ln.rstrip("\n") for ln in f.readlines()])
        else:
            footer_lines.append(
                f"(WARNING: postamble file not found: {cfg['postamble']})"
            )

    body_lines = generate_gcode_from_polygons(
        compensated_polys_sorted, cfg, enable_line_numbers=enable_ln, line_inc=line_inc
    )

    # assemble final lines: header, body, footer
    final_lines = []
    # apply line numbering to pre/post if requested
    if header_lines:
        if enable_ln:
            ln_base = 10
            for i, hl in enumerate(header_lines):
                final_lines.append(f"N{ln_base + i * line_inc:04d} {hl}")
            # ensure body starts after header numbers - but
            # generate_gcode_from_polygons already started at N10;
            # keep it simple and just concatenate
            final_lines.extend(body_lines)
        else:
            final_lines.extend(header_lines)
    else:
        final_lines.extend(body_lines)

    if footer_lines:
        final_lines.extend(footer_lines)

    # write to file
    with open(args.output, "w", encoding="utf-8") as f:
        for row in final_lines:
            f.write(str(row).rstrip("\n") + "\n")

    print(f"Saved G-code to: {args.output}")

    # plotting
    if args.plot:
        # For plotting original shapes, shift them by minx/miny so both align
        original_shifted = shift_geometries(geoms, minx, miny)
        plot_polygons(original_shifted, compensated_polys_sorted)


if __name__ == "__main__":
    main()
