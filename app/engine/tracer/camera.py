"""
app/engine/tracer/camera.py

Pinhole Camera:

The following is a Pinhole Camera (Based on the OpenGL Camera Model).

Defining Parameters:
- position: Current position of Camera in World Space
- pitch: Rotation around X-axis
- yaw: Rotation around the Y-axis
- fov: Field of View
- aspect_ratio: Aspect Ratio of the Camera
- near: Near Clipping Plane
- far: Far Clipping Plane

The camera will be able to:
- Update Orientation based on Mouse Movement
- Compute View and Projection Matrices
- Generate Primary Rays for Ray-Tracing

"""

# Imports

import math

import numpy as np
import taichi as ti
from taichi.math import vec3, normalize

"""
Global Constants
- WORLD_UP: Global up vector for world space (0.0, 1.0, 0.0)
- MAX_PITCH: Setting the maximum pitch angle < 90 degrees to avoid gimbal lock
"""
WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
MAX_PITCH = math.radians(89.5)

# Defining the Camera Class
@ti.data_oriented
class Camera:
  def __init__(self, position, yaw, pitch, fov, aspect_ratio, near, far): # type: ignore
    self.position = np.array(position, dtype=np.float32)
    self.yaw = np.float32(yaw)
    self.pitch = np.float32(pitch)
    self.fov = np.float32(fov)
    self.aspect_ratio = np.float32(aspect_ratio)
    self.near = np.float32(near)
    self.far = np.float32(far)

  # Method to update camera orientation based-on mouse input
  def update_orientation(self, dx: float, dy: float, sensitivity: float = 0.0025):
    """
    Assume the position of camera to be (x, y) in the world space.
    Let's say that the mouse is set to sensitivity σ and the user moves their mouse in world space
    by (dx, dy).

    Now, we update the rotation on the X and Y axes (i.e: pitch and yaw)

    Hence,

    x := x + dx * σ
    y := y + dy * σ

    the change in x, is represented in yaw (Rotation around Y-axis)
    and changes in y, in pitch (Rotation around X-axis)

    Therefore,:
    yaw += dx * σ
    pitch += dy * σ

    We also make sure to clip pitch so that it doesn't exceed MAX_PITCH (89.5 degrees).

    yaw has no such bound -- left unwrapped, += accumulates without limit over a
    long session. That's not just untidy: sin/cos of a very large accumulated angle
    lose precision from range-reduction error in float32 long before it's visibly
    wrong, so it's a session-length-dependent bug that's miserable to reproduce.
    Wrapping yaw into (-pi, pi] after every update keeps the value driving sin/cos
    always small, independent of how long the camera has been running.
    """
    self.yaw += dx * sensitivity
    self.pitch += dy * sensitivity
    self.pitch = np.clip(self.pitch, -MAX_PITCH, MAX_PITCH)
    self.yaw = ((self.yaw + math.pi) % (2.0 * math.pi)) - math.pi

  def update_position(self, keys: dict, right: np.ndarray, forward: np.ndarray, dt: float, speed: float = 3.0):
    """
    Called once per rendered frame -- NOT once per keypress event. `keys` is a
    snapshot of which movement keys are currently held (booleans), sampled at
    render time. `right`/`forward` are passed in rather than recomputed here
    because the frame loop is expected to call basis_from_yaw_pitch() once and
    hand the result to both this method and primary_ray_generation -- computing
    the basis twice per frame would be redundant work for the same answer.
  
    `dt` is the server's own measured elapsed time since the last frame
    (e.g. via time.perf_counter() in the frame loop), NOT a value the client
    reports. If the client stamped dt itself, network jitter in when input
    packets arrive would make movement speed stutter independent of anything
    the player did -- the server measuring its own frame time keeps movement
    speed tied to actual render rate, immune to transport timing noise.
  
    Free-fly: `forward` is used exactly as basis_from_yaw_pitch() returns it,
    unprojected. Looking up and holding "forward" flies you upward -- correct
    for a scene-inspection camera. (If FPS-style ground-locked movement is
    ever wanted instead -- e.g. once the Java terrain service is live and a
    ground plane actually exists to walk on -- the only change needed is
    swapping `forward` below for its horizontal projection,
    normalize((forward[0], 0, forward[2])). Nothing else in this function
    changes.)
    """
    move_fwd = (1.0 if keys.get("forward") else 0.0) - (1.0 if keys.get("back") else 0.0)
    move_right = (1.0 if keys.get("right") else 0.0) - (1.0 if keys.get("left") else 0.0)
    move_up = (1.0 if keys.get("up") else 0.0) - (1.0 if keys.get("down") else 0.0)
  
    # All three axes (forward/back, strafe, vertical) combined into one vector
    # before normalizing -- not just forward+strafe. Folding vertical in too
    # means holding forward+up together is capped to the same speed as any
    # single-key direction, same reasoning as the horizontal-only case, just
    # applied to all three axes uniformly rather than treating vertical as a
    # separate uncapped speed.
    direction = move_fwd * forward + move_right * right + move_up * WORLD_UP
    norm = np.linalg.norm(direction)
  
    # No keys held -> direction is the zero vector -> normalizing would be
    # 0/0. Guard first, same as every other near-zero-before-dividing case
    # this build (BRDF grazing angle, MIS weight denominators, RR clamp).
    if norm > 1e-8:
      self.position += (direction / norm) * speed * dt

  # Calculating the basis vectors we are going from (x, y, z) --> (r, u, f)
  def basis_from_yaw_pitch(self):
    """
    Here (r, u, f): (right, up, forward) --> the basis vectors

    here,
    f = (cos(pitch) * sin(yaw), sin(pitch), -cos(pitch) * cos(yaw))

    f = f / || f || (normalizing the f vector) --> we need direction not magnitude, unit vector in same direction will do

    r = f x WORLD_UP / || f x WORLD_UP || (cross product of f and global up vector)

    u = r x f / || r x f || (cross product of r and f)

    we return (r, u, f) --> basis vector

    Some things to note:
    - The forward vector is calculated based on the yaw and pitch angles.
    - We are using a right-handed coordinate system, hence the negative sign in the z-component of the forward vector.
    - (r, u, f) --> recomputed fresh from yaw/pitch every call, nothing cached on self
    """
    cp = math.cos(self.pitch)
    forward = np.array(
      [cp * math.sin(self.yaw), math.sin(self.pitch), -cp * math.cos(self.yaw)],
      dtype=np.float32,
    )
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, WORLD_UP)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)  # cheap insurance against float drift

    return right, up, forward

  # To compute view matrix, used to convert world space coordinates to camera coordinates
  def view_matrix(self):
    right, up, forward = self.basis_from_yaw_pitch()  # (r, u, f)
    view = np.identity(4, dtype=np.float32)

    """
    V = [
      [r.x, r.y, r.z, -dot(r, p)],
      [u.x, u.y, u.z, -dot(u, p)],
      [-f.x, -f.y, -f.z, dot(f, p)],
      [0, 0, 0, 1]
    ]
    """
    view[0, 0:3] = right
    view[1, 0:3] = up
    view[2, 0:3] = -forward

    view[0, 3] = -np.dot(right, self.position)
    view[1, 3] = -np.dot(up, self.position)
    view[2, 3] = np.dot(forward, self.position)

    return view

  # To compute projection matrix --> converts camera coordinates to clip space coordinates
  def projection_matrix(self):
    """
    let,
    t_fov = tan(fov/2)
    near = f_n
    far = f_f

    Therefore,

    P = [
      1/(a*t_fov), 0, 0, 0,
      0, 1/t_fov, 0, 0,
      0, 0, -(f_f + f_n)/(f_f - f_n), -(2 * f_f * f_n)/(f_f - f_n),
      0, 0, -1, 0
    ]
    """
    tan_half_fov = math.tan(self.fov / 2.0)
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] = 1.0 / (self.aspect_ratio * tan_half_fov)
    p[1, 1] = 1.0 / tan_half_fov
    p[2, 2] = -(self.far + self.near) / (self.far - self.near)
    p[2, 3] = -(2.0 * self.far * self.near) / (self.far - self.near)
    p[3, 2] = -1.0
    return p

  @ti.func
  def ray_direction_for_pixel(px: ti.i32, py: ti.i32, jitter_x: ti.f32, jitter_y: ti.f32, width: ti.i32, height: ti.i32, right: vec3, up: vec3, forward: vec3, fov: ti.f32, aspect_ratio: ti.f32) -> vec3:  # type: ignore
    """Single source of truth for the pixel -> ray mapping.

    Previously duplicated in Camera.primary_ray_generation and
    accumulate_kernel, and the two had already diverged on jitter. A G-buffer
    generated from a different ray than the radiance is misaligned by a
    sub-pixel offset, and every downstream filter inherits that error.

    ndc_y uses the lower-left origin convention, matching ti.GUI.set_image.
    """
    scale_phi = ti.tan(fov / 2.0)

    ndc_x = 2.0 * ((ti.cast(px, ti.f32) + jitter_x) / ti.cast(width, ti.f32)) - 1.0
    ndc_y = 2.0 * ((ti.cast(py, ti.f32) + jitter_y) / ti.cast(height, ti.f32)) - 1.0

    return normalize(
      ndc_x * aspect_ratio * scale_phi * right +
      ndc_y * scale_phi * up +
      forward
    )

  # To generate primary rays for ray tracing
  @ti.kernel
  def primary_ray_generation(self, right: vec3, up: vec3, forward: vec3, position: vec3, fov: ti.f32, aspect_ratio: ti.f32, ray_o: ti.template(), ray_d: ti.template(),): # type: ignore
    """
    here, we are using the OpenGL perspective projection matrix.
    pixel: (p_x, p_y)
    jitter: (j_x, j_y)  --> random number (i.e j_x, j_y in [0, 1))
    aspect_ratio = a
    (r, u, f) = (right, up, forward) --> basis vectors

    x_ndc = (2 * (p_x + j_x) / width)-1
    y_ndc = 1 - (2 * (p_y + j_y)/height)

    where width and height are the dimensions of the image plane.

    d_world = (x_ndc * a * tan(fov/2) * r + y_ndc * tan(fov/2) * u + f) / || (x_ndc * a * tan(fov/2) * r + y_ndc * tan(fov/2) * u + f) ||
    """
    tan_half_fov = math.tan(fov / 2.0)
    for px, py in ray_o:
      jx, jy = ti.random(ti.f32), ti.random(ti.f32)
      x_ndc = (2.0 * (px + jx) / ray_o.shape[0])-1
      y_ndc = 1.0 - (2.0 * (py + jy)/ray_o.shape[1])

      direction = normalize(
        x_ndc * aspect_ratio * tan_half_fov * right +
        y_ndc * tan_half_fov * up + forward
      )

      ray_o[px, py] = position
      ray_d[px, py] = direction