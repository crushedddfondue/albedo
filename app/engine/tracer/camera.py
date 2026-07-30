import math
import taichi as ti
from taichi.math import vec3

ti.init(arch=ti.cuda)

class PinholeCamera:
  def __init__(self, position, yaw_angle, pitch_angle):
    self.position = position
    
    # FIX: Use standard Python math for CPU-side initialization
    self.yaw_angle = math.radians(yaw_angle)
    self.pitch_angle = math.radians(pitch_angle)

  # This is to generate the right vector and up vector for our pinhole camera
  @ti.func
  def orthonormal_basis_generation(self):
    """
    We are trying to create the x, y, z components of our forward vector:

    f = [ cos(theta)*cos(phi) , sin(phi) , sin(theta)*cos(phi) ]

    where:
      f_x: cos(theta)*cos(phi)
      f_y: sin(phi)
      f_z: sin(theta)*cos(phi)
    """
    forward_x = ti.math.cos(self.yaw_angle) * ti.math.cos(self.pitch_angle)
    forward_y = ti.math.sin(self.pitch_angle)
    forward_z = ti.math.sin(self.yaw_angle) * ti.math.cos(self.pitch_angle)

    # Combining the 3 components in x-direction, y-direction and z-direction
    forward = vec3(forward_x, forward_y, forward_z)

    # this is out preset up_world vector
    u_world = vec3(0.0, 1.0, 0.0)

    """
    Here we create the right vector for the pinhole camera

    r = (f x u_world) / || f x u_world ||
    """
    right_vector = ti.math.normalize(ti.math.cross(forward, u_world))

    # up_vector: u = r x f
    up_vector = ti.math.cross(right_vector, forward)

    return forward, right_vector, up_vector

  # To generate a ray-vector
  @ti.func
  def ray_generation(self, x_ndc, y_ndc, fov_angle, aspect_ratio):
    forward, right, up = self.orthonormal_basis_generation()

    """
    This is to return field-of-view scale s:

    where s = tan(fov_angle / 2)

    here fov_angle is in radians.
    """
    # FIX: Use Taichi's math for GPU-side calculations
    fov_scale = ti.math.tan(ti.math.radians(fov_angle) / 2.0)

    """
    Horizontal Component:
      h = x_ndc * aspect ratio * s * r
    
    Vertical Component:
      v = y_ndc * s * u
    
    On Combining the components, we get a ray in a particular direction

    d = (h + v + f) / || h + v + f || = ((x_ndc * aspect ratio * s * r) + (y_ndc * s * u) + f) / || (x_ndc * aspect ratio * s * r) + (y_ndc * s * u) + f) ||
    """
    horizontal_component = x_ndc * aspect_ratio * fov_scale * right
    vertical_component = y_ndc * fov_scale * up

    ray_direction = ti.math.normalize(horizontal_component + vertical_component + forward)

    return ray_direction