import asyncio
import io
import torch
from fastapi import FastAPI, WebSocket
from PIL import Image

from tracer.renderer import render_1spp_kernel
from brain.flow import RectifiedFlowUnet

app = FastAPI()

model = RectifiedFlowUnet().cuda()
model.eval()

WIDTH, HEIGHT = 512, 512

shared_vram_buffer = torch.zeros((WIDTH, HEIGHT, 3), dtype=torch.float32, device='cuda')

@app.websocket("/ws/renderer")
async def render_loop(websocket: WebSocket):
  await websocket.accept()

  try:
    while True:
      # 1. Receive camera coordinates from Node.js transport
      data = await websocket.receive_json()
      cam_x, cam_y, cam_z = data.get("x", 0), data.get("y", 0), data.get("z", 0)
      
      # 2. Taichi Render (Writes directly to shared_vram_buffer)
      render_1spp_kernel(cam_x, cam_y, cam_z, shared_vram_buffer)
      
      # 3. AI Denoising (PyTorch)
      with torch.no_grad():
        # Reshape from Taichi (W, H, C) to PyTorch (B, C, H, W)
        tensor_bchw = shared_vram_buffer.permute(2, 1, 0).unsqueeze(0)
        
        # Run the Rectified Flow Euler step
        clean_tensor = model(tensor_bchw)
        
        # Convert back to (H, W, C) for JPEG compression
        final_image = clean_tensor.squeeze(0).permute(1, 2, 0)
        final_pixels = (final_image.cpu().numpy() * 255).astype('uint8')
      
      # 4. Compress to JPEG and send binary frame
      img = Image.fromarray(final_pixels)
      buf = io.BytesIO()
      img.save(buf, format='JPEG', quality=80)
      byte_frame = buf.getvalue()
      
      await websocket.send_bytes(byte_frame)
  except Exception as e:
    print(f"Connection Closed: {e}")