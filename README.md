# Project Albedo: Real-Time Flow-Matched Neural Rendering

Project Albedo is a full-stack, real-time neural rendering engine. It replaces standard high-sample Monte Carlo integration with a continuous-time neural ODE, resolving extremely noisy 1-Sample-Per-Pixel (1-SPP) ray tracing into photorealistic frames using a custom Rectified Flow model.

By leveraging Zero-Copy VRAM sharing between a Taichi-based hardware-accelerated ray tracer and a PyTorch inference engine, the system achieves real-time interactive framerates streamed over WebSockets to a React frontend.

---

## The Architecture

This project solves the fundamental bandwidth and latency bottlenecks of real-time AI rendering by unifying the graphics pipeline and the AI inference pipeline on a single GPU.

The Engine (Taichi / CUDA): A custom Bounding Volume Hierarchy (BVH) ray tracer shoots exactly 1 ray per pixel. It outputs a 10-channel tensor containing Noisy Radiance, Albedo, World Normals, and Depth directly into VRAM.

The AI (PyTorch / FastAPI): A U-Net, conditioned on the G-buffers, models a deterministic vector field using Rectified Flow. It solves the ODE $dx_t = v_\theta(x_t, t)dt$ in a single Euler step, taking the VRAM pointer from Taichi and resolving the noise instantly.

The Transport (WebSockets / Node.js): The clean image buffer is compressed and streamed dynamically to the client at 60 FPS.

The Client (React): An interactive web canvas where users can manipulate the 3D scene and adjust rendering parameters with zero local compute overhead.

---

## The Mathematics (Rectified Flow)

Unlike traditional Diffusion models (DDPM/DDIM) which rely on Stochastic Differential Equations (SDEs) and require dozens of iterative denoising steps, Albedo utilizes Rectified Flow.

By learning a deterministic vector field that draws straight, non-intersecting paths between the prior distribution $\pi_0$ (1-SPP noise) and the target distribution $\pi_1$ (clean image), the model minimizes the trajectory curvature.

The training objective simplifies to a continuous-time MSE loss:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[\Vert{} v_\theta(x_t, t) - (x_1 - x_0) \Vert{}^2\right]$$

Because the probability paths are straight, inference requires only a single Euler discretization step, dropping latency from seconds per frame to milliseconds per frame.

---

## Tech Stack & Key Flexes

Custom Compute Shaders: Written in Taichi for direct-to-tensor GPU ray tracing. No intermediate CPU copies.

Simulation-Free ODE Training: Built the Conditional Flow Matching math from first principles.

Zero-Copy Tensor Bridge: Overcame standard Python memory constraints by passing raw CUDA pointers between the renderer and the neural network.

Pixel Streaming Protocol: Low-latency WebSockets implementation capable of handling high-resolution buffers dynamically.

---

## References & Inspiration

This architecture was built upon the theoretical foundations established in the following papers:

Flow Matching for Generative Modeling (Lipman et al., ICLR 2023)

Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu et al., ICLR 2023)

RenderFlow: Single-Step Neural Rendering via Flow Matching (2026)
