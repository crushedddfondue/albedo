# Project Albedo: Development Roadmap

Strictly adhering to the principles outlined in "The Protocol.pdf"

---

## Phase-$\alpha$ : Ideation and Deep Dive (Plain English)

### Step-1: Ideation Core

Build a real-time, flow-matched neural rendering engine. We want to bypass the slow process of standard path tracing and standard diffusion, achieving 60 FPS interactive rendering in a browser.

### Step-2: Dig Deep (The Pipeline in English)

```The Canvas (React)```: A user looks at a 3D scene in their browser. They move their mouse to look around.

```The Transport (WebSockets)```: The browser sends those camera coordinates instantly to our backend server.

```The Engine (Taichi/Python)```: The backend receives the coordinates and shoots exactly one light ray per pixel into a virtual 3D scene. It figures out the color of the object, the direction it faces, how far away it is, and a very noisy estimate of shadows. It saves this as raw data in the GPU memory.

```The Brain (PyTorch)```: An AI model looks at this noisy data. Because we trained it to understand physics and straight-line paths, it instantly calculates how to "push" the noisy pixels into a perfectly clean, photorealistic image in one single step.

```The Return```: The server compresses this clean image and shoots it back to the browser. The user sees a perfect frame instantly.

### Step-3: Organizing and Phasing

The Crux (What we build first): The Zero-Copy VRAM bridge between the Taichi Ray Tracer and the PyTorch Model. If these two cannot share memory efficiently, real-time streaming is impossible.

**Build Outward**:

- Mathematics & Theory (Phase-b)

- Backend Infrastructure (Taichi to PyTorch bridge)

- AI Training (Overfitting a single scene)

- Frontend Transport (WebSockets + React)

---

## Phase-$\beta$: Bring the Math In

### Step-0: References and Research Papers

Flow Matching for Generative Modeling (Lipman et al.) -> Understand Vector Fields.

Flow Straight and Fast: Rectified Flow (Liu et al.) -> Understand the 1-Step ODE.

RenderFlow -> Understand how to apply the above to G-buffers.
(Prerequisite topics to master: ODEs, Euler Integration, Continuity Equation, Optimal Transport, Bounding Volume Hierarchies).

### Step-1: Mathematical Walkthrough

**The Input $X_0$**: A tensor of shape $[B, 10, H, W]$ containing Noisy Radiance (3), Depth (1), Normals (3), and Albedo (3).

**The Target $X_1$**: A tensor of shape $[B, 3, H, W]$ containing the 1000-SPP clean image.

**The Vector Field**: We define the straight path: $X_t = (1-t)X_0 + tX_1$.

**The Velocity**: The derivative of this path with respect to time is simply the difference: $dX_t/dt = X_1 - X_0$.

**Inference**: To get the clean image at runtime, we use a single Euler step: $X_1 = X_0 + 1.0 \cdot v_\theta(X_0)$.

### Step-2: Add Compute Cost (Hardware Aware)

**Hardware**: RTX 5070 Ti (12GB VRAM), 32GB System RAM.

**Constraints**: 12GB VRAM is tight for 3D rendering and a deep U-Net simultaneously.

**Adjustments**: We must restrict the training resolution to $256 \times 256$ or $512 \times 512$ initially. Batch sizes during training must be small ($B=2$ or $B=4$). We will use Mixed Precision (torch.float16) to cut memory usage in half.

### Step-3: Training Objective & Data

**Data Generation**: We will use the Taichi engine to render 5,000 paired frames of $X_0$ (noisy) and $X_1$ (clean) from a static 3D mesh.

**Loss Function**: Continuous-time Mean Squared Error (MSE) of the predicted velocity field against the true straight-line velocity.

$$\mathcal{L}(\theta) = \mathbb{E}_{t, X_0, X_1} \left[\Vert{} v_\theta(X_t, t) - (X_1 - X_0) \Vert{}^2\right]$$

---

## Phase-$\gamma$: Implementation

### Step-0: Creating Documentation

(To be completed prior to Step-1)

[x] 1. Finalized Roadmap (This document).

[ ] 2. System Architecture / Design Diagram (Mermaid graph mapping Node -> FastAPI -> Taichi -> PyTorch).

[ ] 3. Decisions File (Defending Taichi over C++/OptiX for Zero-Copy sharing).

[ ] 4. The Complete Mathematical Walkthrough (Expanding Phase-b into full LaTeX proofs).

### Step-1: Creating the Environment

[ ] Initialize Git repository.

[ ] Create Python virtual environment: python -m venv albedo_env

[ ] Create requirements.txt (torch, taichi, fastapi, uvicorn, numpy).

[ ] Create package.json for React frontend.

### Step-2: Begin Coding (Iterative Phasing)

1. **The Math Sandbox**: Code a 2D toy model of Rectified Flow in a Jupyter Notebook to verify the loss function converges.

2. **The Generator**: Write the Taichi Bounding Volume Hierarchy (BVH) and 1-SPP ray tracer. Check: Verify it outputs a correct 10-channel tensor.

3. **The AI**: Write the PyTorch U-Net with Time and Spatial conditioning. Train on the Taichi dataset.

4. **The Bridge**: Write the FastAPI server to handle memory pointers between Taichi and PyTorch in real-time.

5. **The Client**: Build the React UI and WebSocket loop to stream the final frames.
