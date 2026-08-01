import taichi as ti

# Function to sample a direction in the hemisphere with cosine-weighted distribution
@ti.func
def cosine_weighted_sample_hemisphere(u1, u2):
  """
  r = sqrt(u1)
  phi = 2 * pi * u2

  (x, y, z) --> (r*cos(phi), r*sin(phi), sqrt(1 - u1))

  Probability Density Function for cosine-weighted hemisphere sampling:
  pdf = cos(phi) / pi = z / pi

  we return:
  (x, y, z) and pdf
  """
  r = ti.sqrt(u1)

  phi = 2 * ti.math.pi * u2

  x = r * ti.cos(phi)
  y = r * ti.sin(phi)
  z = ti.sqrt(1 - u1)

  pdf = z / ti.math.pi
  return x, y, z, pdf
