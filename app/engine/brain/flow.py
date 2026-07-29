import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, time):
    """
    Injects the continuous time 't' into a high-dimensional vector space.
    time: Tensor of shape (Batch,) containing values between 0.0 and 1.0
    """
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=time.dtype) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, time_emb_dim):
    super().__init__()
    # Project the time embedding to the channel dimension of this block
    self.time_mlp = nn.Linear(time_emb_dim, out_channels)
    
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.gn1 = nn.GroupNorm(8, out_channels)
    self.gn2 = nn.GroupNorm(8, out_channels)
    self.act = nn.SiLU()

    # Residual projection if channel dimensions change
    self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

  def forward(self, x, t):
    h = self.conv1(x)
    h = self.gn1(h)
    h = self.act(h)
    
    # Inject Time Conditioning via Shift (AdaGN style)
    time_emb = self.time_mlp(t)[:, :, None, None]
    h = h + time_emb
    
    h = self.conv2(h)
    h = self.gn2(h)
    h = self.act(h)
    
    return h + self.shortcut(x)

class RectifiedFlowUNet(nn.Module):
  def __init__(self, in_channels=10, out_channels=3, base_dim=64):
    """
    in_channels = 10: [Noisy Radiance (3) + Normals (3) + Albedo (3) + Depth (1)]
    out_channels = 3: [Predicted Velocity for RGB]
    """
    super().__init__()
    
    time_dim = base_dim * 4
    self.time_mlp = nn.Sequential(
      SinusoidalPositionEmbeddings(base_dim),
      nn.Linear(base_dim, time_dim),
      nn.SiLU(),
      nn.Linear(time_dim, time_dim)
    )

    # Initial Projection
    self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

    # Encoder (Downsampling)
    self.downs = nn.ModuleList([
      Block(base_dim, base_dim, time_dim),
      Block(base_dim, base_dim * 2, time_dim),
      Block(base_dim * 2, base_dim * 2, time_dim),
      Block(base_dim * 2, base_dim * 4, time_dim)
    ])
    self.pool = nn.MaxPool2d(2)

    # Bottleneck
    self.mid_block1 = Block(base_dim * 4, base_dim * 4, time_dim)
    self.mid_block2 = Block(base_dim * 4, base_dim * 4, time_dim)

    # Decoder (Upsampling)
    self.ups = nn.ModuleList([
      Block(base_dim * 8, base_dim * 2, time_dim),
      Block(base_dim * 4, base_dim * 2, time_dim),
      Block(base_dim * 4, base_dim, time_dim),
      Block(base_dim * 2, base_dim, time_dim)
    ])
    self.upconvs = nn.ModuleList([
      nn.ConvTranspose2d(base_dim * 4, base_dim * 4, kernel_size=2, stride=2),
      nn.ConvTranspose2d(base_dim * 2, base_dim * 2, kernel_size=2, stride=2),
      nn.ConvTranspose2d(base_dim * 2, base_dim * 2, kernel_size=2, stride=2),
      nn.ConvTranspose2d(base_dim, base_dim, kernel_size=2, stride=2)
    ])

    # Final Output Projection (Predicting straight-line velocity)
    self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

  def forward(self, x_cond):
    """
    x_cond: The 10-channel G-buffer from Taichi. 
    In Rectified Flow inference, we solve the ODE from t=0 to t=1.
    Because paths are straight, we do 1 Euler step:
    X_clean = X_noisy + velocity
    """
    batch_size = x_cond.shape[0]
    device = x_cond.device
    
    # 1. Prepare Time Conditioning (For inference, t=0)
    t = torch.zeros(batch_size, device=device, dtype=x_cond.dtype) 
    time_emb = self.time_mlp(t)

    # 2. Encode
    x = self.init_conv(x_cond)
    
    # We store skip connections for the U-Net architecture
    skips = []
    
    # Down 1
    x = self.downs[0](x, time_emb)
    skips.append(x)
    x = self.pool(x)
    
    # Down 2
    x = self.downs[1](x, time_emb)
    skips.append(x)
    x = self.pool(x)
    
    # Down 3
    x = self.downs[2](x, time_emb)
    skips.append(x)
    x = self.pool(x)
    
    # Down 4
    x = self.downs[3](x, time_emb)
    skips.append(x)
    x = self.pool(x)

    # 3. Bottleneck
    x = self.mid_block1(x, time_emb)
    x = self.mid_block2(x, time_emb)

    # 4. Decode
    # Up 1
    x = self.upconvs[0](x)
    x = torch.cat([x, skips.pop()], dim=1)
    x = self.ups[0](x, time_emb)
    
    # Up 2
    x = self.upconvs[1](x)
    x = torch.cat([x, skips.pop()], dim=1)
    x = self.ups[1](x, time_emb)
    
    # Up 3
    x = self.upconvs[2](x)
    x = torch.cat([x, skips.pop()], dim=1)
    x = self.ups[2](x, time_emb)
    
    # Up 4
    x = self.upconvs[3](x)
    x = torch.cat([x, skips.pop()], dim=1)
    x = self.ups[3](x, time_emb)

    # 5. Output Velocity
    velocity = self.final_conv(x)
    
    # 6. Apply Euler Integration Step
    # The first 3 channels of x_cond are the noisy radiance (X_0)
    x_noisy = x_cond[:, 0:3, :, :]
    
    # Rectified Flow: X_clean = X_0 + (1.0 * v)
    x_clean = x_noisy + velocity
    
    # Clamp output to valid RGB range [0, 1]
    return torch.clamp(x_clean, 0.0, 1.0)