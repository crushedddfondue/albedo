import asyncio
import torch
import cv2
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Import our custom modules
from tracer.renderer import render_1spp_kernel
from brain.flow import RectifiedFlowUNet

app = FastAPI()

# Add CORS Middleware to allow WebSocket connections from the Node Gateway
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allow all origins (for local development)
  allow_credentials=False,
  allow_methods=["*"],
  allow_headers=["*"],
)

# 1. Initialize the Neural Network on the GPU
# We drop base_dim to 16 to ensure it runs fast enough for real-time inference.
# We also call .half() to utilize FP16 Tensor Cores on your RTX GPU.
model = RectifiedFlowUNet(base_dim=16).cuda()
model.eval()
model = model.half()

# 2. Pre-allocate the Shared VRAM Buffer (Zero-Copy)
WIDTH, HEIGHT = 512, 512
# Shape for Taichi: (Width, Height, 10 Channels)
shared_vram_buffer = torch.zeros((WIDTH, HEIGHT, 10), dtype=torch.float32, device='cuda')

@app.websocket("/ws/render")
async def render_loop(websocket: WebSocket):
  await websocket.accept()
  
  try:
    while True:
      # 1. Receive camera coordinates from Node.js transport
      data = await websocket.receive_json()
      cam_x, cam_y, cam_z = data.get("x", 0), data.get("y", 0), data.get("z", 0)
      
      # Yield briefly to the async event loop to prevent WebSocket lag
      await asyncio.sleep(0.001)
      
      # 2. Taichi Render (Writes directly to shared_vram_buffer)
      render_1spp_kernel(cam_x, cam_y, cam_z, shared_vram_buffer)
      
      # 3. AI Denoising (PyTorch)
      with torch.inference_mode():
        # Reshape from Taichi (W, H, C) to PyTorch (B, C, H, W)
        # We cast to .half() here so the input tensor matches the FP16 model weights
        tensor_bchw = shared_vram_buffer.permute(2, 1, 0).unsqueeze(0).half()
        
        # Run the Rectified Flow Euler step
        clean_tensor = model(tensor_bchw)
        
        # Convert back to (H, W, C) for JPEG compression (cast to float32 for safety)
        final_image = clean_tensor.squeeze(0).permute(1, 2, 0).float()
        final_pixels = (final_image.cpu().numpy() * 255).astype('uint8')
      
      # 4. Compress to JPEG and send binary frame
      # OpenCV expects BGR color space, but our model outputs RGB
      final_pixels_bgr = cv2.cvtColor(final_pixels, cv2.COLOR_RGB2BGR)
      
      # Use OpenCV's highly optimized C++ backend to encode the JPEG in ~2ms
      _, encoded_img = cv2.imencode('.jpg', final_pixels_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
      byte_frame = encoded_img.tobytes()
      
      await websocket.send_bytes(byte_frame)
      
  except Exception as e:
    print(f"Connection closed: {e}")