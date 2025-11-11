# Headless DXF2GCode - DXF to G-code Converter

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A headless (command-line) DXF to G-code converter with intelligent optimizations designed to minimize machine jerkiness and improve CNC machining quality.

## 🌟 Key Features

- **G-code Optimization**: Reduces machine jerking by up to 50% through advanced path optimization
- **Contour Ordering**: Automatically mills inner shapes first, outer perimeter last
- **Continuous Contour Processing**: Groups geometry into continuous paths for smooth machining
- **Tool Radius Compensation**: Numerical tool offset using Shapely buffer operations
- **Adaptive Feedrates**: Automatic speed adjustment for different segment lengths
- **Flexible Configuration**: YAML-based configuration with CLI overrides
- **Multi-format Support**: Lines, polylines, circles, arcs from DXF files
- **Visual Feedback**: Interactive plotting with numbered milling sequence visualization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/holoujak/headless-dxf2gcode.git
cd headless-dxf2gcode

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```bash
# Convert DXF to G-code with optimizations
headless-dxf2gcode input.dxf output.gcode

# Disable optimizations for maximum precision
headless-dxf2gcode input.dxf output.gcode --no-optimize

# Visualize the conversion with milling order numbers
headless-dxf2gcode input.dxf output.gcode --plot
```
## Configuration

The converter uses a flexible YAML configuration system with organized sections:

```yaml
# File Input/Output
preamble: "start.gcode"    # G-code file to insert at the beginning
postamble: "end.gcode"     # G-code file to insert at the end

# Coordinate System
origin_lower_left: true    # Shift origin to lower-left corner

# Feed Rates (mm/min)
feed_rate: 5              # Default cutting feedrate for X/Y movements
z_approach_feed: 3        # Feedrate when plunging down to work height
z_travel_feed: 5          # Feedrate when lifting to safe height

# Z-Axis Heights (mm)
z_safe_height: 40         # Safe height for rapid travel
z_work_height: 1          # Working height for cutting operations

# Tool Configuration
tool_radius: 5.5          # Radius of cutting tool (mm) - set to 0 to disable compensation
tool_side: "left"         # Tool compensation: "left" or "right" (ignored when radius=0)

# Path Generation
buffer_resolution: 16     # Segments to approximate rounded corners/curves
join_style: 1             # Corner join: 1=round, 2=mitre, 3=bevel

# G-code Formatting
line_numbers: true        # Add line numbers (N codes) to output
line_number_increment: 10 # Increment between line numbers

# Optimization Settings
optimization:
  enable: true                      # Master switch for all optimizations
  merge_tolerance: 0.01             # Merge points closer than this (mm)
  min_segment_length: 0.05          # Remove segments shorter than this (mm)
  douglas_peucker_tolerance: 0.02   # Curve simplification tolerance (mm)
  adaptive_feedrate: true           # Reduce feedrate for short segments
  min_feed_ratio: 0.3               # Minimum feedrate ratio (30% of normal)
  sort_by_size: true                # Sort contours by size (inner first, outer last)
  arc_detection: false              # [Future] Convert to G02/G03 arcs
```

### Optimization Algorithms

1. **Point Merging**: Eliminates micro-movements by merging nearby points
2. **Short Segment Removal**: Removes segments shorter than minimum length
3. **Douglas-Peucker Simplification**: Simplifies curves while maintaining precision
4. **Adaptive Feedrates**: Slower speeds for short segments, full speed for long runs
5. **Contour Ordering**: Sorts shapes by size to mill inner contours first, outer perimeter last

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting and style checking
- **isort**: Import sorting
- **pytest**: Testing framework
- **pre-commit**: Git hooks for quality checks

## 📈 Usage Examples

### Roughing Operations
```yaml
optimization:
  merge_tolerance: 0.05      # More aggressive
  min_segment_length: 0.1    # Remove more small segments
  douglas_peucker_tolerance: 0.05
```

### Finishing Operations
```yaml
optimization:
  merge_tolerance: 0.005     # More conservative
  min_segment_length: 0.02   # Keep more detail
  douglas_peucker_tolerance: 0.01
```

### Maximum Precision (No Optimization)
```bash
headless-dxf2gcode input.dxf output.gcode --no-optimize
```

### No Tool Compensation (Exact Path Following)
```yaml
# Option 1: Zero radius (no smoothing buffer)
tool_radius: 0

# Option 2: Center mode (with buffer smoothing but no offset)
tool_radius: 6
tool_side: "center"  # Buffer out+in for smoothing, returns to original size

# Useful for: Wire EDM, laser cutting, plotting, drawing
# Note: Optimizations still apply to create smooth continuous paths
```

## 🔧 Command Line Options

```
usage: headless-dxf2gcode [-h] [--config CONFIG] [--origin-lower-left]
                          [--plot] [--line-numbers] [--no-optimize] input output

positional arguments:
  input                Input DXF file
  output               Output G-code file

options:
  --config CONFIG      YAML config file (default: config.yaml)
  --origin-lower-left  Shift origin to lower-left corner
  --plot               Show visualization with milling order numbers
  --line-numbers       Enable N-line numbering
  --no-optimize        Disable G-code optimizations
```

## 📊 Visualization

The `--plot` option provides interactive visualization showing:

- **Original geometry** (blue solid lines)
- **Tool-compensated paths** (red dashed lines)
- **Milling sequence numbers** (red circles with numbers)
- **Origin marker** (black cross at 0,0)

Each contour is numbered according to its milling order (smallest to largest area), making it easy to verify the machining sequence before running on your CNC machine.

## 🧪 Testing

## 📋 Requirements

- Python 3.8+
- ezdxf (DXF file reading)
- shapely (Geometric operations)
- PyYAML (Configuration)
- matplotlib (Optional, for plotting)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run pre-commit checks
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for CNC machining optimization
- Inspired by the need for smoother machine operation
- Uses advanced computational geometry algorithms

## 📞 Support

- 🐛 Issues: Use GitHub Issues for bug reports
- 💡 Features: Submit feature requests via GitHub Issues
