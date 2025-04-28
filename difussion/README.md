# Knolling Diffusion Model

> Transform messy workspaces into organized arrangements through diffusion-based image generation

## Overview

The Knolling Diffusion Model is a specialized image-to-image translation system that learns to transform images of messy workspaces into tidy, organized arrangements. This forms the "brain" of the Knolling Robot system, determining how objects should be arranged before the robot executes the physical rearrangement.

## Key Features

- **Conditional Diffusion Process**: Generates organized workspace images conditioned on messy input images
- **Enhanced U-Net Architecture**: Specialized design with time embeddings and attention mechanisms
- **Multiple Hardware Support**: Works on CUDA GPUs, Apple Silicon (MPS), and CPUs
- **Classifier-Free Guidance**: Adjustable guidance scale for controlling generation quality
- **Comprehensive Visualization**: Training progress tracking with side-by-side comparisons

## Technical Details

### Model Architecture

The model consists of two main components:

1. **ImprovedDiffusionModel**: Implements the full denoising diffusion probabilistic model (DDPM) with conditioning
   - Forward process (adding noise)
   - Reverse process (denoising)
   - Sampling procedure
   - Loss calculation
   - Classifier-free guidance

2. **EnhancedUNet**: Neural network for noise prediction
   - Encoder-decoder architecture with skip connections
   - Group normalization for stability
   - Time embeddings at multiple levels
   - Self-attention mechanisms
   - Conditional input handling

### Dataset Structure

The model works with paired "before" (messy) and "after" (tidy) images of workspace arrangements:

```
dataset/
├── origin_images_before/  # Messy workspace images
│   ├── label_123_4.png
│   └── ...
└── origin_images_after/   # Corresponding tidy arrangements
    ├── label_123_4.png
    └── ...
```

The `TidyingDataset` class handles loading and processing these image pairs, including:
- Support for multiple dataset formats
- Automatic setup ID and arrangement ID extraction
- Data augmentation during training (random cropping, flipping)
- Normalization to [-1, 1] range

### Training Process

Training the diffusion model involves:

1. Sample random timesteps for each batch
2. Add noise according to the diffusion schedule
3. Predict the noise using the U-Net
4. Calculate MSE loss between predicted and actual noise
5. Update model parameters
6. Generate samples periodically to track progress

Key training parameters:
- Cosine noise schedule
- 100 diffusion timesteps
- AdamW optimizer with weight decay
- Cosine annealing learning rate schedule
- Gradient clipping for stability

## Installation and Setup

### Requirements

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.20.0
matplotlib>=3.5.0
tqdm>=4.64.0
Pillow>=9.0.0
```

### Hardware Compatibility

The code automatically selects the appropriate device:
- CUDA GPU (if available)
- Apple Silicon via MPS (if available)
- CPU (fallback)

## Usage

### Training

```bash
# Train a new model
python knolling_diff_train.py

# Resume training from checkpoint
python knolling_diff_train.py --resume path/to/checkpoint.pt
```

### Inference

```python
# Load model and run inference
from knolling_diff_train import load_model, organize_workspace

# Load the model
model = load_model("path/to/model.pt", device)

# Generate organized workspace image
tidy_img, comparison_path = organize_workspace(
    model, 
    "messy_workspace.png", 
    "tidy_workspace.png", 
    device,
    guidance_scale=2.5
)
```

## Training Progress Visualization

During training, the model generates visualizations showing the progress:

- Individual sample images at each logged epoch
- Side-by-side comparisons of input, generated, and target images
- Vertical stacking of progress images to show improvement over time

Example training progression visualization:

```
Input (Messy) → Generated (Epoch N) → Target (Tidy)
```

## Model Performance

The diffusion model achieves high-quality results after approximately 500 epochs of training. Performance depends on:

- Dataset size and quality
- Image resolution (default 256×256)
- Training duration
- Guidance scale during sampling

## Integration with Robot Control

The diffusion model integrates with the robot control system through:

1. **Generate Target Layout**: Use the model to create a tidy arrangement image
2. **Detect Objects**: YOLO detector identifies objects in both current scene and generated layout
3. **Match Objects**: Pair objects between current and target scenes
4. **Plan Movements**: Robot controller plans and executes the physical rearrangement

## Files and Directory Structure

```
.
├── knolling_diff_train.py     # Main diffusion model implementation
├── experiments/               # Generated during training
│   └── tidying_diffusion_*/   # Experiment directories
│       ├── models/            # Saved model checkpoints
│       ├── logs/              # Training logs
│       ├── training_progress/ # Visualization images
│       ├── config.json        # Experiment configuration
│       └── README.md          # Auto-generated experiment info
```

## Customization

Key parameters that can be adjusted:

- **Image Size**: Default 256×256px, increase for more detail
- **Batch Size**: Adjust based on available GPU memory
- **Timesteps**: More steps = better quality but slower inference
- **Guidance Scale**: Higher values = stronger adherence to conditioning

## Troubleshooting

**Memory Issues**: If encountering CUDA out-of-memory errors:
- Reduce batch size
- Decrease image resolution
- Use mixed precision training

**Training Instability**: If experiencing NaN losses or divergence:
- Reduce learning rate
- Increase gradient clipping threshold
- Check for data normalization issues