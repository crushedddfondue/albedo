import taichi as ti
from taichi.math import vec3

# Defining the BRDF (Bi-directional Reflectance Distribution Function) class
@ti.data_oriented
class BRDF:
  def __init__(self):
    pass
  @ti.func
  def evaluate(self, albedo: vec3)-> vec3:  # type: ignore
    """
    f_r = rho / pi
    """
    f_r = albedo / ti.math.pi

    return f_r

  @ti.func
  def sample_cosine_weighted_bounce(self, albedo: vec3, cos_theta: ti.f32, l_i: vec3)-> vec3: # type: ignore
    result = vec3(0.0)

    if cos_theta > 0.0:
      result = albedo * l_i

    return result