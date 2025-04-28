import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F

from PIL import Image
import numpy as np

from scipy.optimize import linear_sum_assignment

class SimpleObjectDetector:
    def __init__(self, confidence_threshold=0.2):
        # Use CPU or MPS (Apple Silicon). Adjust if you have CUDA available.
        self.device = torch.device('cpu' if torch.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image_path):
        """
        Returns a list of bounding boxes in standard YOLO format:
            [x_center, y_center, width, height]
        filtered by confidence score > confidence_threshold.
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)[0]

        boxes = []
        img_width, img_height = image.size

        for box, score in zip(predictions['boxes'], predictions['scores']):
            if score > self.confidence_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                boxes.append([x_center, y_center, width, height])

        return boxes


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two *axis-aligned* boxes
    in YOLO format: [cx, cy, w, h].
    """
    def get_corners(b):
        cx, cy, w, h = b
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = get_corners(box1)
    x1_2, y1_2, x2_2, y2_2 = get_corners(box2)

    # intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    # union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def filter_overlapping_boxes(boxes, overlap_threshold=0.1):
    """
    Filter out overlapping boxes, keeping the ones that appear higher (lowest y_center).
    """
    if not boxes:
        return []

    # Sort boxes by y-center ascending
    boxes_with_idx = [(i, b) for i, b in enumerate(boxes)]
    boxes_with_idx.sort(key=lambda x: x[1][1])  # box[1] = y_center

    filtered_boxes = []
    for i, box1 in boxes_with_idx:
        overlaps = False
        for kept_box in filtered_boxes:
            if calculate_iou(box1, kept_box) > overlap_threshold:
                overlaps = True
                break
        if not overlaps:
            filtered_boxes.append(box1)

    return filtered_boxes


def match_objects_to_labels(detected_boxes, class_names, pos_values, class_to_id, max_distance=2.0):
    """
    Example that uses Hungarian algorithm to do minimal-distance matching
    between 'detected_boxes' and 'original_positions'.
    Each input box is [cx, cy, w, h] in YOLO format (axis-aligned).
    """
    if not detected_boxes or not class_names:
        return []

    # Filter out overlapping detections
    filtered_boxes = filter_overlapping_boxes(detected_boxes)
    if len(filtered_boxes) != len(class_names):
        return []

    # Parse original positions
    MAX_X, MAX_Y = 0.3, 0.25
    original_positions = []
    for i in range(len(class_names)):
        idx = i * 7
        if idx + 6 >= len(pos_values):
            return []
        # pos_values structure: [y, x, z, z, v4, v5, v6]
        y, x = pos_values[idx: idx + 2]
        # ignoring some extra fields
        x_norm = x / MAX_X
        y_norm = y / MAX_Y
        original_positions.append({
            'coords': (x_norm, y_norm),  # used for matching
            'index': i
        })

    N = len(filtered_boxes)
    cost_matrix = np.zeros((N, N), dtype=np.float32)

    def euclid(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    # Build NxN cost matrix
    for i, pos_dict in enumerate(original_positions):
        (x_exp, y_exp) = pos_dict['coords']
        for j, box in enumerate(filtered_boxes):
            (x_det, y_det, w_det, h_det) = box
            cost_matrix[i, j] = euclid((x_exp, y_exp), (x_det, y_det))

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # If any distance is > max_distance => skip
    new_annotations = []
    for i in range(N):
        assigned_col = col_ind[i]
        dist_ij = cost_matrix[i, assigned_col]
        if dist_ij > max_distance:
            return []
        class_id = class_to_id[class_names[ original_positions[i]['index'] ]]
        matched_box = filtered_boxes[assigned_col]  # [cx, cy, w, h]
        new_annotations.append([class_id, *matched_box])

    return new_annotations


def create_yolo_dataset(base_dir, output_dir, train_ratio=0.8, max_batch=2000):
    """
    Creates a YOLO-OBB dataset from your base directory.
    Key difference: we output 5 parameters (x, y, w, h, angle),
    where angle is either 0 or 90 depending on orientation (w > h or h > w).
    """
    detector = SimpleObjectDetector(confidence_threshold=0.2)

    # Create subdirs
    yolo_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in yolo_dirs:
        Path(os.path.join(output_dir, dir_path)).mkdir(parents=True, exist_ok=True)

    classes_list = []
    class_to_id = {}
    processed_count = 0
    skipped_count = 0
    dir_counts = defaultdict(int)

    # Example: directories named "VAE_1118_obj2" up to "VAE_1118_obj8"
    vae_dirs = [f"VAE_1118_obj{i}" for i in range(2, 9)]

    print("Collecting unique classes...")
    for vae_dir in os.listdir(base_dir):
        if vae_dir not in vae_dirs:
            continue

        labels_dir = os.path.join(base_dir, vae_dir, 'labels_after_0')
        if not os.path.isdir(labels_dir):
            continue

        # Read all *name.txt to build your class list
        for name_file in os.listdir(labels_dir):
            if not name_file.startswith('num_') or not name_file.endswith('_name.txt'):
                continue
            with open(os.path.join(labels_dir, name_file), 'r') as f:
                for line in f:
                    for class_name in line.strip().split():
                        if class_name not in class_to_id:
                            class_to_id[class_name] = len(classes_list)
                            classes_list.append(class_name)

    print(f"Found {len(classes_list)} unique classes")

    # Process each subdir
    for vae_dir in vae_dirs:
        dir_path = os.path.join(base_dir, vae_dir)
        if not os.path.exists(dir_path):
            continue

        print(f"\nProcessing directory: {vae_dir}")
        num_objects = int(vae_dir.split('obj')[-1])

        images_dir = os.path.join(dir_path, 'origin_images_after')
        labels_dir = os.path.join(dir_path, 'labels_after_0')
        if not (os.path.isdir(images_dir) and os.path.isdir(labels_dir)):
            continue

        image_files = sorted(os.listdir(images_dir))

        for img_file in image_files:
            if not img_file.endswith('_0.png'):
                continue

            try:
                # Example: "img_45_0.png"
                img_id = int(img_file.split('_')[1])

                # batch logic
                batch_num = ((img_id // 20)) * 20 + 20
                if batch_num > max_batch:
                    skipped_count += 1
                    continue

                # label filenames
                name_pattern = f"num_{num_objects}_{batch_num}_name.txt"
                pos_pattern = f"num_{num_objects}_{batch_num}.txt"
                name_file = os.path.join(labels_dir, name_pattern)
                pos_file = os.path.join(labels_dir, pos_pattern)
                if not (os.path.exists(name_file) and os.path.exists(pos_file)):
                    continue

                with open(name_file, 'r') as f:
                    class_lines = f.readlines()
                with open(pos_file, 'r') as f:
                    pos_lines = f.readlines()

                line_idx = img_id % 20
                if line_idx >= len(class_lines) or line_idx >= len(pos_lines):
                    continue

                class_names = class_lines[line_idx].strip().split()
                pos_values = [float(x) for x in pos_lines[line_idx].strip().split()]

                # Detect objects (axis-aligned)
                img_path = os.path.join(images_dir, img_file)
                detected_boxes = detector.detect_objects(img_path)

                # Match detection results to ground truth positions
                new_annotations = match_objects_to_labels(
                    detected_boxes,
                    class_names,
                    pos_values,
                    class_to_id
                )

                if new_annotations:
                    is_train = (random.random() < train_ratio)
                    subset = 'train' if is_train else 'val'

                    # Copy image
                    dst_img = os.path.join(output_dir, 'images', subset, f"{vae_dir}_{img_file}")
                    shutil.copy2(img_path, dst_img)

                    # Write OBB labels (5 parameters: x_center, y_center, w, h, angle)
                    label_path = os.path.join(
                        output_dir, 'labels', subset,
                        f"{vae_dir}_{img_file.replace('.png', '.txt')}"
                    )

                    with open(label_path, 'w') as f:
                        for ann in new_annotations:
                            class_id = ann[0]
                            x_center, y_center, w, h = ann[1], ann[2], ann[3], ann[4]

                            # Decide orientation based on which side is longest
                            if w >= h:
                                # horizontal => angle 0
                                angle_deg = 0.0
                            else:
                                # vertical => angle 90, swap w,h
                                angle_deg = 90.0
                                w, h = h, w

                            line = f"{int(class_id)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {angle_deg:.1f}\n"
                            f.write(line)

                    processed_count += 1
                    dir_counts[vae_dir] += 1

                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} images...")
                else:
                    skipped_count += 1

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                skipped_count += 1
                continue

    print("\nProcessing Summary:")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped:   {skipped_count}")
    for dir_name in sorted(dir_counts.keys()):
        print(f"{dir_name}: {dir_counts[dir_name]} images")

    # Write out classes
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes_list))

    return classes_list


if __name__ == "__main__":
    classes = create_yolo_dataset(
        base_dir="../../Downloads/data/dataset",
        output_dir="yolo_dataset_obb"
    )
    print("\nClasses:", classes)
