# Knolling Robot: Automated Object Arrangement System

> A robotic system that arranges objects in organized patterns using computer vision and precise manipulation

## Overview

The Knolling Robot is the physical execution component of a larger object arrangement system. It uses a combination of computer vision and robotics to detect objects in a workspace and precisely arrange them according to predetermined patterns or generated layouts.

This README focuses on the robot control and object detection modules that enable a WidowX 200 robot arm to identify, pick, and place objects in an organized manner.

## System Components

### 1. Object Detection (`knolling_detector.py`)

The detection module uses YOLO with oriented bounding boxes (OBB) to accurately detect objects in the workspace:

- **YOLODetector**: Class that integrates ROS camera feeds with YOLO models
- **2D-to-3D Transformation**: Converts image coordinates to robot workspace coordinates
- **ROI Selection**: Allows defining a region of interest for focused detection
- **Color Processing**: Extracts average color for object matching

### 2. Robot Control (`knolling_control.py`)

The control module coordinates the robotic arm's movements to arrange objects:

- **WorkspaceManager**: Maintains the current state of all objects in the workspace
- **ArrangeObjects**: Main class for the execution pipeline
- **Collision Detection**: Advanced algorithm for detecting object collisions and planning movements
- **Pick-and-Place**: Specialized routines for reliable object manipulation

## Installation Requirements

- ROS2 (Robot Operating System) - Foxy distribution
- Python 3.8+
- PyTorch 1.12+
- OpenCV 4.5+
- Interbotix WidowX 200 robot arm (or compatible)
- Intel RealSense camera (or compatible)

## Configuration

The system is configured through the `CONFIG` dictionary in both main modules:

```python
CONFIG = {
    'ROBOT': {
        'MODEL': 'wx200',  # Robot model
        'NAME': 'wx200',   # ROS node name
        'GRIPPER_PRESSURE': 0.5,  # Default gripper pressure
        # ... other robot parameters
    },
    'WORKSPACE': {
        'X_MIN': 0.10,  # Workspace boundaries (meters)
        'X_MAX': 0.35,
        'Y_MIN': -0.20,
        'Y_MAX': 0.20,
        # ... other workspace parameters
    },
    # ... other configuration categories
}
```

## Usage

### 1. Starting the System

```bash
# Terminal 1: Start ROS core
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model=wx200

# Terminal 2: Start camera node
ros2 launch realsense2_camera rs_launch.py
```

### 2. Running Object Arrangement

```bash
# Arrange objects according to a layout image
python knolling_control.py path/to/layout_image.png
```

The system will:
1. Detect objects in the current scene
2. Detect target positions from the layout image
3. Match objects between current and target scenes
4. Execute a series of movements to arrange objects

## Implementation Details

### Object Detection Process

1. **Camera Calibration**: Uses camera intrinsics for accurate 2D-to-3D projection
2. **YOLO Detection**: Detects objects with oriented bounding boxes
3. **Coordinate Transformation**: Projects detected objects to robot coordinate frame
4. **Object Properties**: Calculates position, orientation, dimensions, and color

### Robot Manipulation Strategy

1. **Object Matching**: Uses Hungarian algorithm to match detected objects to target layout
2. **Motion Planning**:
   - First moves objects blocking other target positions
   - Then places objects in their final positions
   - Uses temporary positions when necessary
3. **Collision Avoidance**: Uses Separating Axis Theorem for precise collision detection
4. **Gripper Control**: Custom grasp and release functions for reliable manipulation

### Key Classes

#### `DetectedObject`
```python
@dataclass(frozen=False, eq=True)
class DetectedObject:
    object_position: Tuple[float, float, float]
    bbox: Optional[Dict[str, Any]] = None
    class_name: str = ""
    confidence: float = 0.0
    angle: float = 0.0
    avg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dimensions: Tuple[float, float] = (0.0, 0.0)
```

#### `WorkspaceManager`
Manages the state of objects in the workspace, including:
- Tracking object positions and dimensions
- Finding empty positions for temporary placement
- Collision checking between objects

#### `ArrangeObjects`
Main execution class that:
- Initializes robot and detector
- Identifies objects in scene and target layout
- Plans and executes movement sequences
- Handles object dependencies and blocking objects

## Troubleshooting

### Common Issues

1. **Detection Issues**:
   - Ensure proper lighting in the workspace
   - Check camera calibration and positioning
   - Adjust YOLO confidence thresholds in CONFIG

2. **Robot Movement Issues**:
   - Verify robot control node is running
   - Check for joint limits or singularities
   - Adjust approach heights in CONFIG if gripper collides with objects

3. **Object Grasping Issues**:
   - Adjust gripper pressure settings
   - Modify grasp pitch angle for better approach
   - Check object dimensions detection

## Future Improvements

- Dynamic obstacle avoidance
- Real-time visual servoing for more precise placement
- Support for more complex object geometries
- Integration with reinforcement learning for adaptive manipulation

## Algorithm Highlights

### Collision Detection
The system uses the Separating Axis Theorem (SAT) to detect collisions between oriented rectangles:

```python
def _check_oriented_rectangle_collision(self, pos1, size1, angle1, pos2, size2, angle2):
    # Quick check using bounding circles
    max_radius1 = math.sqrt(width1**2 + height1**2) / 2
    max_radius2 = math.sqrt(width2**2 + height2**2) / 2
    
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    min_dist = CONFIG['WORKSPACE']['OBJECT_CLEARANCE']
    
    if dist > (max_radius1 + max_radius2 + min_dist):
        return False
        
    # Calculate corners and check separating axes
    # [algorithm implementation...]
    
    # No separating axis found; rectangles collide
    return True
```

### Movement Sequence
The system uses a three-phase approach to arrange objects:

1. Move blockers to temporary positions
2. Place objects at their target positions
3. Move objects from temporary positions to their final locations

This strategy handles complex dependencies and ensures collision-free movements.