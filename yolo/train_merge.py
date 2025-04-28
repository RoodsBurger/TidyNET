from ultralytics import YOLO
import gc
import torch
import os
import shutil
from pathlib import Path
import random

def create_multitask_dataset():
    """
    Creates a dataset.yaml that combines images from both 'output' and 'output_v' directories.
    """
    print("Creating multi-task dataset with both output and output_v...")
    
    # Get current working directory
    cwd = os.getcwd()
    output_v_dir = os.path.join(cwd, "output_v")
    output_dir = os.path.join(cwd, "output")
    
    # Create a combined directory for the multi-task dataset
    combined_dir = os.path.join(cwd, "output_combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    # Check if directories exist
    if not os.path.exists(output_v_dir):
        raise FileNotFoundError(f"The 'output_v' directory doesn't exist at {output_v_dir}")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"The 'output' directory doesn't exist at {output_dir}")
    
    # Function to copy valid images and labels to combined directory
    def copy_valid_images_from_dir(source_dir, prefix):
        valid_images = []
        image_files = [f for f in os.listdir(source_dir) if f.endswith(".png")]
        
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            
            if os.path.exists(os.path.join(source_dir, label_file)):
                # Check if the label file has content
                if os.path.getsize(os.path.join(source_dir, label_file)) > 0:
                    # Copy with prefix to avoid filename conflicts
                    new_img_name = f"{prefix}_{img_file}"
                    new_label_name = f"{prefix}_{label_file}"
                    
                    shutil.copy2(
                        os.path.join(source_dir, img_file),
                        os.path.join(combined_dir, new_img_name)
                    )
                    shutil.copy2(
                        os.path.join(source_dir, label_file),
                        os.path.join(combined_dir, new_label_name)
                    )
                    valid_images.append(new_img_name)
        
        return valid_images
    
    # Copy images from both directories
    valid_v_images = copy_valid_images_from_dir(output_v_dir, "v")
    valid_images = copy_valid_images_from_dir(output_dir, "o")
    
    print(f"Found {len(valid_v_images)} valid labeled images from output_v")
    print(f"Found {len(valid_images)} valid labeled images from output")
    print(f"Total: {len(valid_v_images) + len(valid_images)} images for multi-task learning")
    
    # Create train-val split for better evaluation
    all_images = valid_v_images + valid_images
    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Write train.txt and val.txt
    with open(os.path.join(combined_dir, 'train.txt'), 'w') as f:
        for img in train_images:
            f.write(f"./{img}\n")
            
    with open(os.path.join(combined_dir, 'val.txt'), 'w') as f:
        for img in val_images:
            f.write(f"./{img}\n")
    
    # Create dataset YAML with absolute paths
    with open('dataset_multitask.yaml', 'w') as f:
        f.write(f"# Multi-task dataset configuration with absolute paths\n")
        f.write(f"path: {combined_dir}  # Absolute path to combined directory\n")
        f.write(f"train: {os.path.join(combined_dir, 'train.txt')}  # Train images list\n")
        f.write(f"val: {os.path.join(combined_dir, 'val.txt')}  # Validation images list\n")
        f.write(f"\n# Classes\n")
        f.write(f"names:\n")
        f.write(f"  0: gear_3\n")
        f.write(f"  1: motor_1\n")
        f.write(f"  2: motor_2\n")
        f.write(f"  3: charger_3\n")
        f.write(f"  4: spiralnotebook_1\n")
        f.write(f"  5: wrench_2\n")
        f.write(f"  6: ballpointpen_1\n")
        f.write(f"  7: utilityknife_1\n")
        f.write(f"  8: stapler_1\n")
        f.write(f"  9: highlighter_3\n")
        f.write(f"  10: highlighter_1\n")
        f.write(f"  11: utilityknife_2\n")
    
    print(f"Created dataset_multitask.yaml with combined data path: {combined_dir}")
    return len(train_images), len(val_images)

def train_yolo_multitask(run_name, epochs=300):
    """
    Train the YOLO model on multiple tasks (output and output_v datasets combined).
    
    Args:
        run_name: Name for this training run
        epochs: Number of epochs to train for
        
    Returns:
        Training results
    """
    # Create multi-task dataset
    num_train, num_val = create_multitask_dataset()
    total_images = num_train + num_val
    
    # Adjust batch size based on available images
    batch_size = min(16, total_images // 2)  # Ensure batch size is reasonable
    
    # Load a model
    #model_path = os.path.abspath('./runs/obb/yolo_knolling_obb/weights/best.pt')
    model_path = os.path.abspath('./yolo11m-obb.pt')
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get the original train method
    original_train = model.train
    
    def train_with_memory_cleanup(*args, **kwargs):
        """Wrapper around the original train method that adds memory cleanup"""
        # Run the original training
        results = original_train(*args, **kwargs)
        
        # Clean up memory
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
    
    # Replace the train method
    model.train = train_with_memory_cleanup
    
    # Determine the device
    if torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon
    elif torch.cuda.is_available():
        device = 0      # CUDA
    else:
        device = 'cpu'  # CPU
    
    print(f"Training on {device} device")
    
    # Path to dataset config
    dataset_path = os.path.abspath('dataset_multitask.yaml')
    print(f"Using multi-task dataset config at: {dataset_path}")
    
    # Train the model with heavy augmentation
    try:
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device,

            hsv_h=0.7,    # Increased hue adjustment (more color tone variation)
            hsv_s=0.3,    # Increased saturation adjustment (more color intensity variation)
            hsv_v=0.5,    # Increased value adjustment (more brightness variation)

            degrees=180,             # Less extreme rotation (was 180)
            scale=0.7,              # Less extreme scaling (was 0.5)
            shear=10,               # Less shear (was 20)
            perspective=0.0003,     # Less perspective distortion
            flipud=0.3,             # Reduce vertical flipping (was 0.5)
            fliplr=0.3,             # Keep horizontal flipping
            mosaic=1.0,             # Keep mosaic
            mixup=0.3,              # Reduce mixup (was 0.5)
            copy_paste=0.2,         # Reduce copy-paste (was 0.3)
            auto_augment=None,      # Disable random auto augmentation
            erasing=0.2,            # Reduce erasing (was 0.3)
            
            # Training parameters
            name=run_name,
            patience=100,           # Increased patience for early stopping
            save=True,
            save_period=50,        # Save checkpoints every 50 epochs
            cache=True,            # Cache images for faster training
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',     # More suitable for small datasets
            lr0=0.0005,             # Initial learning rate
            lrf=0.01,              # Final learning rate factor
            weight_decay=0.001,    # Weight decay
            warmup_epochs=10,      # Longer warmup with small dataset
            cos_lr=True,           # Use cosine learning rate scheduler
            verbose=True,
            seed=42,
            close_mosaic=150,       # Disable mosaic in final epochs for fine-tuning
            overlap_mask=True,     # Helps with small objects
        )
        return results
    
    except Exception as e:
        print(f"Training error occurred: {e}")
        return None
    finally:
        # Clean up memory even if training fails
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

def evaluate_on_individual_tasks(model_path):
    """
    Evaluate the multi-task trained model on each individual task separately.
    
    Args:
        model_path: Path to the trained model
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Create individual dataset configs for evaluation
    cwd = os.getcwd()
    
    # For output_v
    with open('dataset_output_v.yaml', 'w') as f:
        output_v_dir = os.path.join(cwd, "output_v")
        f.write(f"path: {output_v_dir}\n")
        f.write(f"test: ./\n")
        f.write(f"names:\n")
        f.write(f"  0: gear_3\n")
        f.write(f"  1: motor_1\n")
        f.write(f"  2: motor_2\n")
        f.write(f"  3: charger_3\n")
        f.write(f"  4: spiralnotebook_1\n")
        f.write(f"  5: wrench_2\n")
        f.write(f"  6: ballpointpen_1\n")
        f.write(f"  7: utilityknife_1\n")
        f.write(f"  8: stapler_1\n")
        f.write(f"  9: highlighter_3\n")
        f.write(f"  10: highlighter_1\n")
        f.write(f"  11: utilityknife_2\n")
    
    # For output
    with open('dataset_output.yaml', 'w') as f:
        output_dir = os.path.join(cwd, "output")
        f.write(f"path: {output_dir}\n")
        f.write(f"test: ./\n")
        f.write(f"names:\n")
        f.write(f"  0: gear_3\n")
        f.write(f"  1: motor_1\n")
        f.write(f"  2: motor_2\n")
        f.write(f"  3: charger_3\n")
        f.write(f"  4: spiralnotebook_1\n")
        f.write(f"  5: wrench_2\n")
        f.write(f"  6: ballpointpen_1\n")
        f.write(f"  7: utilityknife_1\n")
        f.write(f"  8: stapler_1\n")
        f.write(f"  9: highlighter_3\n")
        f.write(f"  10: highlighter_1\n")
        f.write(f"  11: utilityknife_2\n")
    
    # Evaluate on output_v
    print(f"\n\n===== Evaluating on output_v dataset =====")
    try:
        metrics_v = model.val(data='dataset_output_v.yaml')
        print(f"Metrics on output_v: {metrics_v}")
    except Exception as e:
        print(f"Evaluation error on output_v: {e}")
    
    # Evaluate on output
    print(f"\n\n===== Evaluating on output dataset =====")
    try:
        metrics = model.val(data='dataset_output.yaml')
        print(f"Metrics on output: {metrics}")
    except Exception as e:
        print(f"Evaluation error on output: {e}")

if __name__ == "__main__":
    run_name = 'yolo_multitask'
    results = train_yolo_multitask(run_name)
    
    if results is not None:
        print("Multi-task training completed successfully!")
        try:
            # Evaluate on combined dataset
            model_path = f'runs/obb/{run_name}/weights/best.pt'
            model = YOLO(model_path)
            metrics = model.val(data='dataset_multitask.yaml')
            print(f"Validation metrics on combined dataset: {metrics}")
            
            # Evaluate on individual tasks
            evaluate_on_individual_tasks(model_path)
            
        except Exception as e:
            print(f"Validation error occurred: {e}")
        finally:
            # Final cleanup
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()