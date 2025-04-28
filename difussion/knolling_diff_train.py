import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import datetime
import random

# Device selection logic - prioritizes CUDA, then MPS (Apple Silicon), then CPU
cuda_available = torch.cuda.is_available()
mps_available = hasattr(torch, 'mps') and torch.mps.is_available()

if cuda_available:
    device = torch.device("cuda")
    print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
elif mps_available:
    device = torch.device("mps")
    print(f"Using device: MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print(f"Using device: CPU")

def create_output_dirs(experiment_name=None):
    """
    Creates directory structure for experiment outputs with timestamp
    
    Args:
        experiment_name: Optional name for the experiment
        
    Returns:
        Tuple of directories (output_dir, model_dir, samples_dir, logs_dir)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        output_dir = f"experiments/{experiment_name}_{timestamp}"
    else:
        output_dir = f"experiments/experiment_{timestamp}"
    
    model_dir = os.path.join(output_dir, "models")
    samples_dir = os.path.join(output_dir, "samples")
    logs_dir = os.path.join(output_dir, "logs")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return output_dir, model_dir, samples_dir, logs_dir

class TidyingDataset(Dataset):
    """
    Custom dataset for loading pairs of before (messy) and after (tidy) workspace images
    Handles both old and new dataset formats based on directory structure
    """
    def __init__(self, root_dir, transform=None, use_augmentation=True):
        """
        Initialize the dataset
        
        Args:
            root_dir: Root directory containing the image data
            transform: Optional torchvision transforms to apply to images
            use_augmentation: Whether to apply data augmentation during training
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.before_paths = []
        self.after_paths = []
        self.setup_labels = []
        self.arrangement_ids = []
        
        # Check which dataset format is present and load accordingly
        if os.path.exists(os.path.join(root_dir, 'origin_images_before')):
            before_dir = os.path.join(root_dir, 'origin_images_before')
            after_dir = os.path.join(root_dir, 'origin_images_after')
            self._load_dataset(before_dir, after_dir, is_new_format=True)
        elif os.path.exists(os.path.join(root_dir, 'images_before')):
            before_dir = os.path.join(root_dir, 'images_before')
            after_dir = os.path.join(root_dir, 'images_after')
            self._load_dataset(before_dir, after_dir, is_new_format=False)
        else:
            raise ValueError(f"Could not find image directories in {root_dir}")
        
        print(f"Dataset contains {len(self.before_paths)} image pairs")
        
    def _load_dataset(self, before_dir, after_dir, is_new_format=True):
        """
        Load the dataset from directories
        
        Args:
            before_dir: Directory containing "before" (messy) images
            after_dir: Directory containing "after" (tidy) images
            is_new_format: Whether the dataset uses the new naming convention
        """
        image_files = sorted([f for f in os.listdir(before_dir) 
                             if f.endswith('.png') and not f.startswith('.')])
        setup_arrangements = {}
        
        for img_file in image_files:
            before_path = os.path.join(before_dir, img_file)
            after_path = os.path.join(after_dir, img_file)
            
            if not os.path.exists(after_path):
                print(f"Warning: Missing after image for {img_file}")
                continue
                
            # Extract setup and arrangement IDs from filename based on format
            if is_new_format:
                match = re.match(r'label_(\d+)_(\d+)\.png', img_file)
                if match:
                    setup_id, arrangement_id = map(int, match.groups())
                else:
                    print(f"Warning: Filename format not recognized: {img_file}")
                    continue
            else:
                setup_id = int(os.path.splitext(img_file)[0]) % 1000
                arrangement_id = int(os.path.splitext(img_file)[0]) // 1000
            
            # Group by setup_id to track different arrangements of the same setup
            if setup_id not in setup_arrangements:
                setup_arrangements[setup_id] = []
            
            setup_arrangements[setup_id].append((before_path, after_path, arrangement_id))
        
        # Build the dataset from all valid image pairs
        for setup_id, arrangements in setup_arrangements.items():
            for before_path, after_path, arrangement_id in arrangements:
                self.before_paths.append(before_path)
                self.after_paths.append(after_path)
                self.setup_labels.append(setup_id)
                self.arrangement_ids.append(arrangement_id)
        
        print(f"Found {len(setup_arrangements)} unique setups")
        
    def __len__(self):
        """Return the total number of image pairs in the dataset"""
        return len(self.before_paths)
    
    def __getitem__(self, idx):
        """
        Get a single data item from the dataset
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Tuple of (before_img, after_img, setup_label, arrangement_id)
        """
        before_img = Image.open(self.before_paths[idx]).convert('RGB')
        after_img = Image.open(self.after_paths[idx]).convert('RGB')
        
        # Apply data augmentation if enabled
        if self.transform and self.use_augmentation:
            width, height = before_img.size
            
            # Random crop - slightly smaller than original image
            i, j, h, w = transforms.RandomCrop.get_params(before_img, output_size=(height-20, width-20))
            before_img = transforms.functional.crop(before_img, i, j, h, w)
            after_img = transforms.functional.crop(after_img, i, j, h, w)
            
            # Random horizontal flip (50% chance)
            if random.random() > 0.5:
                before_img = transforms.functional.hflip(before_img)
                after_img = transforms.functional.hflip(after_img)
                
        # Apply transforms (resize, convert to tensor, normalize)
        if self.transform:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)
        
        return before_img, after_img, self.setup_labels[idx], self.arrangement_ids[idx]

class EnhancedUNet(nn.Module):
    """
    Enhanced U-Net architecture for diffusion model's noise prediction network
    Includes time embeddings and attention layers
    
    The model takes both the noisy image and the conditioning image (messy workspace)
    """
    def __init__(self, in_channels=3, time_emb_dim=256):
        """
        Initialize the U-Net model
        
        Args:
            in_channels: Number of input image channels (typically 3 for RGB)
            time_emb_dim: Dimension of time step embeddings
        """
        super().__init__()
        
        # Time embedding layers to encode diffusion timestep
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        
        init_features = 64
        
        # Input convolutional block - takes concatenated noisy and conditioning images (2*in_channels)
        self.inc = self._double_conv(in_channels * 2, init_features)
        
        # Downsampling path (encoder)
        self.down1 = self._down_block(init_features, init_features * 2)
        self.down2 = self._down_block(init_features * 2, init_features * 4)
        self.down3 = self._down_block(init_features * 4, init_features * 8)
        
        # Bottleneck with attention mechanism
        self.bottleneck = nn.Sequential(
            self._double_conv(init_features * 8, init_features * 16),
            self._attention_block(init_features * 16),
            self._double_conv(init_features * 16, init_features * 8)
        )
        
        # Upsampling path (decoder) with skip connections
        self.up3 = self._up_block(init_features * 16, init_features * 4)
        self.up2 = self._up_block(init_features * 8, init_features * 2)
        self.up1 = self._up_block(init_features * 4, init_features)
        
        # Output layer to produce the predicted noise
        self.outc = nn.Conv2d(init_features * 2, in_channels, kernel_size=1)
        
        # Linear projections for time embeddings to each layer in the network
        self.time_to_down1 = nn.Linear(time_emb_dim, init_features * 2)
        self.time_to_down2 = nn.Linear(time_emb_dim, init_features * 4)
        self.time_to_down3 = nn.Linear(time_emb_dim, init_features * 8)
        self.time_to_up3 = nn.Linear(time_emb_dim, init_features * 4)
        self.time_to_up2 = nn.Linear(time_emb_dim, init_features * 2)
        self.time_to_up1 = nn.Linear(time_emb_dim, init_features)
    
    def _double_conv(self, in_channels, out_channels):
        """
        Helper function to create a double convolution block with normalization and activation
        Used in both encoder and decoder paths
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),  # Group normalization for better stability
            nn.SiLU(inplace=True),  # SiLU activation (aka Swish)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True)
        )
    
    def _down_block(self, in_channels, out_channels):
        """
        Helper function to create a downsampling block (encoder path)
        Consists of max pooling followed by double convolution
        """
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._double_conv(in_channels, out_channels)
        )
    
    def _up_block(self, in_channels, out_channels):
        """
        Helper function to create an upsampling block (decoder path)
        Uses bilinear upsampling followed by convolution
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True)
        )
    
    def _attention_block(self, channels):
        """
        Helper function to create a self-attention block
        Applies channel-wise attention mechanism
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, cond, t):
        """
        Forward pass through the U-Net
        
        Args:
            x: Noisy input image
            cond: Conditioning image (messy workspace)
            t: Diffusion timestep (normalized to [0,1])
            
        Returns:
            Predicted noise in the input image
        """
        # Embed the diffusion timestep
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Concatenate noisy image with conditioning image along channel dimension
        x = torch.cat([x, cond], dim=1)
        
        # Encoder path with skip connections
        x1 = self.inc(x)
        
        x2 = self.down1(x1)
        # Add time embedding to features - broadcast to spatial dimensions
        x2 = x2 + self.time_to_down1(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        x3 = self.down2(x2)
        x3 = x3 + self.time_to_down2(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        x4 = self.down3(x3)
        x4 = x4 + self.time_to_down3(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder path with skip connections
        x = self.up3(torch.cat([x, x4], dim=1))
        x = x + self.time_to_up3(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        x = self.up2(torch.cat([x, x3], dim=1))
        x = x + self.time_to_up2(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        x = self.up1(torch.cat([x, x2], dim=1))
        x = x + self.time_to_up1(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Final output layer with skip connection to input features
        x = self.outc(torch.cat([x, x1], dim=1))
        
        return x

class ImprovedDiffusionModel(nn.Module):
    """
    Diffusion model implementation with conditioning for workspace organization
    Implements the full denoising diffusion probabilistic model (DDPM) process
    with classifier-free guidance support
    """
    def __init__(self, model, beta_schedule='cosine', timesteps=1000):
        """
        Initialize the diffusion model
        
        Args:
            model: U-Net model for predicting noise
            beta_schedule: Schedule for noise variance ('linear' or 'cosine')
            timesteps: Number of diffusion steps
        """
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Set up noise schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            # Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        # Pre-compute diffusion parameters and register them as buffers
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1. / self.alphas))
        
        # Posterior variance (for q(x_{t-1} | x_t, x_0))
        self.register_buffer('posterior_variance', self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(self.posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod))
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process (add noise)
        Sample from q(x_t | x_0) - add noise according to schedule
        
        Args:
            x_0: Original clean image
            t: Timestep
            noise: Optional pre-generated noise (if None, will be generated)
            
        Returns:
            Noisy image x_t and the added noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Extract diffusion parameters for this timestep
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # Mix the original image with noise according to the schedule
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and the predicted noise
        Reverses the diffusion process
        """
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        # x_0 = (x_t - sqrt(1-ɑ_bar) * noise) / sqrt(ɑ_bar)
        return sqrt_recip_alphas_t * x_t - sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute mean and variance of posterior q(x_{t-1} | x_t, x_0)
        
        Args:
            x_0: Predicted clean image
            x_t: Current noisy image
            t: Current timestep
            
        Returns:
            Mean and log variance of the posterior distribution
        """
        posterior_mean_coef1_t = extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = extract(self.posterior_mean_coef2, t, x_t.shape)
        
        # Posterior mean: μ(x_t, x_0) = (coef1 * x_0) + (coef2 * x_t)
        posterior_mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, cond, t):
        """
        Compute parameters of p(x_{t-1} | x_t) using the model prediction
        
        Args:
            x: Current noisy image
            cond: Conditioning image (messy workspace)
            t: Current timestep
            
        Returns:
            Mean and log variance for the next step
        """
        # Predict noise using the model
        noise_pred = self.model(x, cond, t / self.timesteps)
        
        # Predict x_0 from the noise
        x_recon = self.predict_start_from_noise(x, t, noise_pred)
        x_recon = torch.clamp(x_recon, -1., 1.)  # Clamp to valid image range
        
        # Get posterior distribution parameters
        model_mean, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x, t)
        
        return model_mean, posterior_log_variance
    
    def p_sample(self, x, cond, t, guidance_scale=1.0):
        """
        Sample one step in the reverse diffusion process (denoising)
        Implements classifier-free guidance if guidance_scale > 1.0
        
        Args:
            x: Current noisy image
            cond: Conditioning image (messy workspace)
            t: Current timestep
            guidance_scale: Scale for classifier-free guidance (1.0 = no guidance)
            
        Returns:
            Sampled image for the previous timestep (t-1)
        """
        # Get model prediction
        model_mean, model_log_variance = self.p_mean_variance(x, cond, t)
        
        # Apply classifier-free guidance if scale > 1.0 and not in final stages
        if guidance_scale > 1.0 and t[0] > self.timesteps // 4:
            # Run model with zero conditioning (unconditional)
            zero_cond = torch.zeros_like(cond)
            uncond_mean, _ = self.p_mean_variance(x, zero_cond, t)
            
            # Adjust mean: mean = uncond_mean + guidance_scale * (cond_mean - uncond_mean)
            model_mean = uncond_mean + guidance_scale * (model_mean - uncond_mean)
        
        # Add noise scaled by variance (no noise for the final step)
        noise = torch.zeros_like(x) if t[0] == 0 else torch.randn_like(x)
        variance = torch.exp(model_log_variance)
        
        return model_mean + torch.sqrt(variance) * noise
    
    def p_losses(self, x_0, cond, t, noise=None):
        """
        Compute training loss for a diffusion model step
        
        Args:
            x_0: Original clean image (target)
            cond: Conditioning image (messy workspace)
            t: Random timestep
            noise: Optional pre-generated noise
            
        Returns:
            MSE loss between predicted and actual noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Sample noisy image at timestep t
        x_noisy, target = self.q_sample(x_0, t, noise)
        
        # Predict the noise at timestep t
        pred = self.model(x_noisy, cond, t / self.timesteps)
        
        # Simple MSE loss on noise prediction
        loss = F.mse_loss(pred, target)
        
        return loss
    
    @torch.no_grad()
    def sample(self, cond, shape, guidance_scale=1.0):
        """
        Generate a new image using the complete reverse diffusion process
        
        Args:
            cond: Conditioning image (messy workspace)
            shape: Output image shape
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Generated image (tidy workspace)
        """
        b = shape[0]
        device = next(self.model.parameters()).device
        
        # Start with pure noise
        img = torch.randn(shape, device=device)
        
        # Gradually denoise the image
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            img = self.p_sample(
                img, 
                cond, 
                torch.full((b,), i, device=device, dtype=torch.long),
                guidance_scale=guidance_scale
            )
            
        return img

def extract(a, t, x_shape):
    """
    Helper function to extract timestep-specific values from the precomputed tensors
    Extracts values at indices t and reshapes them to match x_shape for broadcasting
    
    Args:
        a: Source tensor to extract from 
        t: Timestep indices
        x_shape: Shape of the target tensor (for proper broadcasting)
        
    Returns:
        Extracted values, reshaped for broadcasting
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def save_training_progress_grid(model, fixed_batch, output_dir, epoch, guidance_scale=1.0):
    """
    Generate a grid of images showing training progress
    
    Args:
        model: Trained diffusion model
        fixed_batch: Batch of fixed test images to use across epochs
        output_dir: Output directory
        epoch: Current epoch number
        guidance_scale: Guidance scale for sampling
        
    Returns:
        Path to the saved progress image
    """
    model.eval()
    
    before_imgs, after_imgs, setup_labels, _ = fixed_batch
    before_imgs = before_imgs.to(device)
    after_imgs = after_imgs.to(device)
    
    samples_dir = os.path.join(output_dir, 'training_progress')
    os.makedirs(samples_dir, exist_ok=True)
    
    with torch.no_grad():
        num_samples = min(4, len(before_imgs))
        fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        
        for i in range(num_samples):
            # Generate an image from the current model
            sample_shape = after_imgs[i:i+1].shape
            generated = model.sample(before_imgs[i:i+1], sample_shape, guidance_scale=guidance_scale)
            
            # Convert images to numpy for visualization
            before_img = before_imgs[i].cpu().permute(1, 2, 0).numpy()
            after_img = after_imgs[i].cpu().permute(1, 2, 0).numpy()
            gen_img = generated[0].cpu().permute(1, 2, 0).numpy()
            
            # Denormalize images from [-1,1] to [0,1]
            before_img = np.clip((before_img + 1) / 2, 0, 1)
            after_img = np.clip((after_img + 1) / 2, 0, 1)
            gen_img = np.clip((gen_img + 1) / 2, 0, 1)
            
            setup_id = setup_labels[i].item()
            
            # Handle special case for single image
            if num_samples == 1:
                axs[0].imshow(before_img)
                axs[0].set_title(f'Input (Setup {setup_id})')
                axs[0].axis('off')
                
                axs[1].imshow(gen_img)
                axs[1].set_title(f'Generated (Epoch {epoch})')
                axs[1].axis('off')
                
                axs[2].imshow(after_img)
                axs[2].set_title('Target')
                axs[2].axis('off')
            else:
                axs[i, 0].imshow(before_img)
                axs[i, 0].set_title(f'Input (Setup {setup_id})')
                axs[i, 0].axis('off')
                
                axs[i, 1].imshow(gen_img)
                axs[i, 1].set_title(f'Generated (Epoch {epoch})')
                axs[i, 1].axis('off')
                
                axs[i, 2].imshow(after_img)
                axs[i, 2].set_title('Target')
                axs[i, 2].axis('off')
    
    # Save the progress grid for this epoch
    plt.tight_layout()
    plt.savefig(os.path.join(samples_dir, f'progress_epoch_{epoch}.png'))
    plt.close()
    
    # Create a stacked image of all progress grids (every 10 epochs)
    concat_filepath = os.path.join(output_dir, 'training_progression.png')
    if epoch % 10 == 0 or epoch == 0:
        existing_images = []
        for e in range(0, epoch+1, 10):
            img_path = os.path.join(samples_dir, f'progress_epoch_{e}.png')
            if os.path.exists(img_path):
                existing_images.append(img_path)
        
        if existing_images:
            # Stack images vertically
            images = [Image.open(img_path) for img_path in existing_images]
            widths, heights = zip(*(i.size for i in images))
            
            total_height = sum(heights)
            max_width = max(widths)
            
            new_img = Image.new('RGB', (max_width, total_height))
            
            y_offset = 0
            for img in images:
                new_img.paste(img, (0, y_offset))
                y_offset += img.height
            
            new_img.save(concat_filepath)
    
    return concat_filepath

def generate_sample(model, dataloader, epoch, device, samples_dir, guidance_scale=1.0):
    """
    Generate and save sample images for visualization at a given epoch
    
    Args:
        model: Trained diffusion model
        dataloader: DataLoader to get sample images from
        epoch: Current epoch number 
        device: Device to run the model on
        samples_dir: Directory to save sample images
        guidance_scale: Guidance scale for sampling
    """
    model.eval()
    
    # Create directory for this epoch's samples
    epoch_samples_dir = os.path.join(samples_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_samples_dir, exist_ok=True)
    
    # Get a batch of images for sampling
    before_imgs, after_imgs, setup_labels, arrangement_ids = next(iter(dataloader))
    before_imgs = before_imgs.to(device)
    after_imgs = after_imgs.to(device)
    
    with torch.no_grad():
        # Get unique setup IDs to sample from
        unique_setups = torch.unique(setup_labels)
        
        n_samples = min(3, len(unique_setups))
        fig, axs = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
        
        if n_samples == 1:
            axs = axs.reshape(1, 3)
        
        for i, setup in enumerate(unique_setups[:n_samples]):
            # Find first image with this setup ID
            idx = (setup_labels == setup).nonzero(as_tuple=True)[0][0]
            
            # Generate image using the diffusion model
            sample_shape = after_imgs[idx:idx+1].shape
            samples = model.sample(before_imgs[idx:idx+1], sample_shape, guidance_scale=guidance_scale)
            
            # Convert to numpy for visualization
            before_img = before_imgs[idx].cpu().permute(1, 2, 0).numpy()
            after_img = after_imgs[idx].cpu().permute(1, 2, 0).numpy()
            generated_img = samples[0].cpu().permute(1, 2, 0).numpy()
            
            # Denormalize from [-1,1] to [0,1]
            before_img = (before_img + 1) / 2
            after_img = (after_img + 1) / 2
            generated_img = (generated_img + 1) / 2
            
            before_img = np.clip(before_img, 0, 1)
            after_img = np.clip(after_img, 0, 1)
            generated_img = np.clip(generated_img, 0, 1)
            
            setup_id = setup.item()
            
            # Save individual images
            plt.figure(figsize=(5, 5))
            plt.imshow(before_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_samples_dir, f"setup_{setup_id}_input.png"))
            plt.close()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(generated_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_samples_dir, f"setup_{setup_id}_generated.png"))
            plt.close()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(after_img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_samples_dir, f"setup_{setup_id}_target.png"))
            plt.close()
            
            # Add to comparison grid
            axs[i, 0].imshow(before_img)
            axs[i, 0].set_title(f'Messy (Input) - Setup {setup_id}')
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(generated_img)
            axs[i, 1].set_title('Generated Tidy')
            axs[i, 1].axis('off')
            
            axs[i, 2].imshow(after_img)
            axs[i, 2].set_title('Ground Truth Tidy')
            axs[i, 2].axis('off')
        
        # Save the combined grid
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f"combined_samples_epoch_{epoch+1}.png"))
        plt.close()

def train(model, dataloader, optimizer, scheduler, epochs, device, output_dirs, start_epoch=0):
    """
    Train the diffusion model
    
    Args:
        model: Diffusion model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer (e.g. Adam)
        scheduler: Learning rate scheduler
        epochs: Number of epochs to train
        device: Device to train on
        output_dirs: Tuple of output directories
        start_epoch: Starting epoch (for resuming training)
    """
    output_dir, model_dir, samples_dir, logs_dir = output_dirs
    
    # Initialize or continue training log
    training_log_path = os.path.join(logs_dir, "training_log.csv")
    if start_epoch == 0:
        with open(training_log_path, "w") as f:
            f.write("epoch,loss,learning_rate\n")
    
    # Get a fixed batch for progress visualization
    fixed_batch = next(iter(dataloader))
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for step, (before_imgs, after_imgs, setup_labels, _) in enumerate(pbar):
                before_imgs = before_imgs.to(device)
                after_imgs = after_imgs.to(device)
                
                optimizer.zero_grad()
                
                # Sample random timesteps for each image in batch
                t = torch.randint(0, model.timesteps, (before_imgs.shape[0],), device=device).long()
                
                # Calculate loss
                loss = model.p_losses(after_imgs, before_imgs, t)
                
                # Backpropagate
                loss.backward()
                
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        # Step the scheduler if it exists
        if scheduler is not None:
            scheduler.step()
        
        # Calculate and log average loss
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6e}')
        
        with open(training_log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{optimizer.param_groups[0]['lr']:.6e}\n")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(model_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
            }, best_model_path)
            
            print(f"New best model saved with loss: {avg_loss:.6f}")
        
        # Generate progress images periodically
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            progress_img_path = save_training_progress_grid(model, fixed_batch, output_dir, epoch, guidance_scale=2.0)
            print(f"Training progress image saved at epoch {epoch+1}")

def organize_workspace(model, image_path, output_path, device, guidance_scale=2.0):
    """
    Generate a tidy workspace image from a messy workspace image
    
    Args:
        model: Trained diffusion model
        image_path: Path to the input messy workspace image
        output_path: Path to save the output tidy workspace image
        device: Device to run inference on
        guidance_scale: Guidance scale for sampling
        
    Returns:
        Tuple of (tidy_image, comparison_image_path)
    """
    # Prepare the transformation for inference
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load and preprocess the input image
    messy_img = Image.open(image_path).convert('RGB')
    original_size = messy_img.size
    messy_tensor = transform(messy_img).unsqueeze(0).to(device)
    
    # Generate the tidy image
    model.eval()
    with torch.no_grad():
        tidy_tensor = model.sample(messy_tensor, messy_tensor.shape, guidance_scale=guidance_scale)
    
    # Convert back to image
    tidy_img = tidy_tensor[0].cpu().permute(1, 2, 0).numpy()
    tidy_img = (tidy_img + 1) / 2  # Denormalize from [-1,1] to [0,1]
    tidy_img = np.clip(tidy_img * 255, 0, 255).astype(np.uint8)    
    tidy_pil = Image.fromarray(tidy_img)
    
    # Resize back to original dimensions if needed
    if original_size != (256, 256):
        tidy_pil = tidy_pil.resize(original_size, Image.LANCZOS)
        
    tidy_pil.save(output_path)
    
    # Create a side-by-side comparison
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.array(messy_img))
    axs[0].set_title('Messy Workspace (Input)')
    axs[0].axis('off')
    
    axs[1].imshow(np.array(tidy_pil))
    axs[1].set_title('Organized Workspace (Output)')
    axs[1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.splitext(output_path)[0] + '_comparison.png'
    plt.savefig(comparison_path)
    plt.close()
    
    return tidy_pil, comparison_path

def load_model(checkpoint_path, device):
    """
    Load a saved diffusion model from a checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model to
        
    Returns:
        Loaded diffusion model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model architecture
    model = EnhancedUNet(in_channels=3, time_emb_dim=256).to(device)
    diffusion = ImprovedDiffusionModel(model, beta_schedule='cosine', timesteps=100).to(device)
    
    # Load saved weights
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    
    return diffusion

def main():
    """
    Main function for training or resuming training of the diffusion model
    Handles command line arguments, dataset loading, and training
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or resume training a tidying diffusion model')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    # Handle resuming training from checkpoint
    if args.resume:
        checkpoint_dir = os.path.dirname(args.resume)
        if os.path.basename(os.path.dirname(checkpoint_dir)) == "models":
            output_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            output_dir = os.path.dirname(args.resume)
        
        model_dir = os.path.join(output_dir, "models")
        samples_dir = os.path.join(output_dir, "samples")
        logs_dir = os.path.join(output_dir, "logs")
        
        for directory in [model_dir, logs_dir]:
            os.makedirs(directory, exist_ok=True)
            
        print(f"Resuming training in existing directory: {output_dir}")
    else:
        # Create new experiment directories
        output_dir, model_dir, samples_dir, logs_dir = create_output_dirs(experiment_name="tidying_diffusion_improved")
        print(f"Experiment outputs will be saved to: {output_dir}")
    
    # Load or create configuration
    config_path = os.path.join(output_dir, "config.json")
    if args.resume and os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        print("Loaded configuration from existing experiment")
    else:
        # Default configuration
        config = {
            "image_size": 256,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "epochs": 500,
            "timesteps": 100,
            "beta_schedule": "cosine",
            "device": str(device),
            "guidance_scale": 5.0,
            "use_both_datasets": False,
        }
        
        import json
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    
    # Set up image transformations
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    datasets = []
    
    try:
        new_dataset = TidyingDataset('./new_dataset', transform=transform)
        datasets.append(new_dataset)
        print(f"Loaded new dataset with {len(new_dataset)} image pairs")
    except Exception as e:
        print(f"Error loading new dataset: {e}")
    
    if config["use_both_datasets"]:
        try:
            old_dataset = TidyingDataset('./old_dataset', transform=transform)
            datasets.append(old_dataset)
            print(f"Loaded old dataset with {len(old_dataset)} image pairs")
        except Exception as e:
            print(f"Error loading old dataset: {e}")
    
    # Combine datasets if multiple are loaded
    if len(datasets) > 1:
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(datasets)
        print(f"Combined dataset contains {len(dataset)} image pairs")
    elif len(datasets) == 1:
        dataset = datasets[0]
    else:
        raise ValueError("No datasets were successfully loaded")
    
    # Log dataset information
    if not args.resume:
        with open(os.path.join(logs_dir, "dataset_info.txt"), "w") as f:
            f.write(f"Total dataset contains {len(dataset)} image pairs\n")
            for i, ds in enumerate(datasets):
                if hasattr(ds, 'setup_labels'):
                    setup_counts = {}
                    for label in ds.setup_labels:
                        setup_counts[label] = setup_counts.get(label, 0) + 1
                    
                    f.write(f"\nDataset {i+1} contains {len(ds)} image pairs\n")
                    f.write(f"Found {len(setup_counts)} unique setups\n")
                    f.write("Setup distribution:\n")
                    for setup_id, count in setup_counts.items():
                        f.write(f"  Setup {setup_id}: {count} images\n")
    
    # Configure DataLoader with appropriate workers for device
    num_workers = 0 if device.type == 'cpu' else 4 if device.type == 'cuda' else 2
    pin_memory = (device.type != 'cpu')
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Create model, optimizer, and scheduler
    noise_predictor = EnhancedUNet(in_channels=3, time_emb_dim=256).to(device)
    model = ImprovedDiffusionModel(
        noise_predictor, 
        beta_schedule=config["beta_schedule"], 
        timesteps=config["timesteps"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["epochs"], 
        eta_min=config["learning_rate"] / 10
    )
    
    start_epoch = 0
    
    # Load checkpoint if resuming training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model state from checkpoint: {args.resume}")
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state from checkpoint")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    # Train the model
    train(
        model, 
        dataloader, 
        optimizer, 
        scheduler,
        epochs=config["epochs"], 
        device=device, 
        output_dirs=(output_dir, model_dir, samples_dir, logs_dir),
        start_epoch=start_epoch
    )
    
    print("Training complete!")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'tidying_diffusion_final.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_model_path)
    
    # Create README for experiment
    if not args.resume:
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write(f"# Improved Tidying Diffusion Model\n\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Configuration\n\n")
            f.write(f"```\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write(f"```\n\n")
            f.write(f"## Directory Structure\n\n")
            f.write(f"- models/: Contains best model only\n")
            f.write(f"- logs/: Contains training logs and dataset information\n")
            f.write(f"- training_progress/: Contains visual progression of training\n")

def inference(model_path, input_image_path, output_path=None):
    """
    Run inference with a trained model on a single image
    
    Args:
        model_path: Path to the trained model
        input_image_path: Path to input messy workspace image
        output_path: Path to save output tidy workspace image (optional)
        
    Returns:
        Tuple of (tidy_image, comparison_image_path)
    """
    model = load_model(model_path, device)
    
    if output_path is None:
        output_dir = os.path.dirname(input_image_path)
        filename = os.path.basename(input_image_path)
        output_path = os.path.join(output_dir, f"tidy_{filename}")
    
    tidy_img, comparison_path = organize_workspace(model, input_image_path, output_path, device, guidance_scale=2.5)
    
    print(f"Organized workspace image saved to: {output_path}")
    print(f"Comparison image saved to: {comparison_path}")
    
    return tidy_img, comparison_path

if __name__ == "__main__":
    main()