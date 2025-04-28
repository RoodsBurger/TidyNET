# Knolling YOLO Training Module

This module handles training and dataset preparation for the YOLO (You Only Look Once) object detection model used in the Knolling Robot system. It specifically focuses on training a model that can detect objects with oriented bounding boxes (OBB), which is crucial for precise object arrangement.

## Overview

The Knolling YOLO Training Module consists of two main components:

1. **Dataset Preparation** (`dataset_obb.py`): Creates a YOLO-compatible dataset with oriented bounding boxes from raw image data.
2. **Model Training** (`train_merge.py`): Trains and evaluates a YOLO model on the prepared dataset.

## Dataset Preparation (`dataset_obb.py`)

This script converts a collection of images with object annotations into a format suitable for training a YOLO OBB model.

### Key Features

- **Pre-detection**: Uses a pre-trained Faster R-CNN model to detect objects in images
- **Automatic Labeling**: Matches detected boxes to known object positions using the Hungarian algorithm
- **Orientation Handling**: Determines if objects are horizontal or vertical and assigns appropriate angles
- **Train/Validation Split**: Automatically splits the dataset for proper evaluation

### Usage

```bash
python dataset_obb.py
```

The script will:

1. Scan the base directory for object images and labels
2. Detect objects in each image
3. Match detections to known object classes and positions
4. Create YOLO-format labels with orientation information
5. Output a structured dataset ready for training

### Implementation Details

- `SimpleObjectDetector`: A lightweight detector using torchvision's Faster R-CNN
- `calculate_iou`: Calculates Intersection over Union between bounding boxes
- `filter_overlapping_boxes`: Removes duplicate detections
- `match_objects_to_labels`: Performs optimal assignment between detections and ground truth
- `create_yolo_dataset`: Main function that orchestrates the entire process

## Model Training (`train_merge.py`)

This script handles training a YOLO model on the prepared dataset, with special considerations for multi-task learning and resource efficiency.

### Key Features

- **Multi-task Dataset Creation**: Combines data from multiple sources
- **Memory-Efficient Training**: Includes garbage collection for Apple Silicon (MPS) and CUDA devices
- **Robust Augmentation**: Implements data augmentation strategies optimized for small object detection
- **Comprehensive Evaluation**: Tests the model on both combined and individual datasets

### Usage

```bash
python train_merge.py
```

The script will:

1. Create a combined dataset from 'output' and 'output_v' directories
2. Load a pre-trained YOLO model
3. Train the model with optimized hyperparameters
4. Evaluate performance on validation data
5. Test the model on individual tasks to ensure balanced performance

### Implementation Details

- `create_multitask_dataset`: Combines multiple datasets with appropriate train/val splits
- `train_yolo_multitask`: Configures and executes the training process
- `train_with_memory_cleanup`: Wrapper function that handles memory management
- `evaluate_on_individual_tasks`: Tests model performance on specific subsets

## Configuration

The training process uses the following key parameters:

```python
# Model parameters
imgsz=640                  # Input image size
batch=batch_size           # Batch size (auto-adjusted)
device='mps'/'cuda'/'cpu'  # Device selection based on availability

# Augmentation parameters
hsv_h=0.7                  # Hue variation
hsv_s=0.3                  # Saturation variation
hsv_v=0.5                  # Value/brightness variation
degrees=180                # Rotation range
scale=0.7                  # Scale variation
mosaic=1.0                 # Mosaic augmentation
# ... and more augmentation options

# Training parameters
epochs=300                 # Number of training epochs
patience=100               # Early stopping patience
optimizer='AdamW'          # Optimizer selection
lr0=0.0005                 # Initial learning rate
warmup_epochs=10           # Learning rate warmup
```

## Dataset Structure

After preparation, the dataset follows this structure:

```
yolo_dataset_obb/
├── images/
│   ├── train/            # Training images
│   └── val/              # Validation images
├── labels/
│   ├── train/            # Training labels (YOLO format with angle)
│   └── val/              # Validation labels (YOLO format with angle)
└── classes.txt           # List of object classes
```

Each label file follows the YOLO OBB format:
```
<class_id> <x_center> <y_center> <width> <height> <angle>
```

Where angle is either 0° (horizontal) or 90° (vertical) based on the object's orientation.

## Object Classes

The system is trained to recognize various objects commonly found in workspaces:

```
0: gear_3
1: motor_1
2: motor_2
3: charger_3
4: spiralnotebook_1
5: wrench_2
6: ballpointpen_1
7: utilityknife_1
8: stapler_1
9: highlighter_3
10: highlighter_1
11: utilityknife_2
```

## Hardware Compatibility

The training module is designed to work efficiently on:

- NVIDIA GPUs (using CUDA)
- Apple Silicon (using MPS)
- CPU (as a fallback)

## Dependencies

- PyTorch 1.12+
- torchvision
- Ultralytics YOLO
- NumPy
- SciPy
- PIL (Pillow)

## Integration with Robot Control

The YOLO model trained by this module is used by the robot control system to:

1. Detect objects in the current workspace
2. Determine their position, orientation, and dimensions
3. Create matching between detected objects and desired arrangements
4. Guide the robot arm for precise placement

The model exports to standard YOLO format (`.pt` files) which are loaded by the `YOLODetector` class in the robot control module.