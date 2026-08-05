import taichi as ti

from tracer.sampling.hemisphere import cosine_weighted_sample_hemisphere

"""
Method to convert direction sampled from cosine_weighted_sample_hemisphere(u1, u2)
into world space
"""
@ti.func
def rotate_to_world_space(t, b, n):
  """
  This is done by using the tangent space basis vectors:
  (t, b, n) --> (tangent, bitangent, normal)

  we randomly sample u1 and u2 in [0, 1)

  w_world = x*t + y*b + z*n / || x*t + y*b + z*n ||
  pdf = cos(phi) / pi = z / pi --> from cosine_weighted_sample_hemisphere(u1, u2)
  we return:
  w_world and pdf

  """
  u1, u2 = ti.random(ti.f32), ti.random(ti.f32)
  x, y, z, pdf = cosine_weighted_sample_hemisphere(u1, u2)

  w_world = x * t + y * b + z * n

  return w_world.normalized(), pdf