import math
from taichi.math import vec3


def static(frame, base_position=vec3(0.0, 1.2, 4.0), base_yaw=0.0):
  return base_position, base_yaw

def dolly(frame, speed=0.01, base_position=vec3(0.0, 1.2, 4.0), base_yaw=0.0):
  position = base_position + vec3(0, 0, -speed * frame)
  return position, base_yaw

def pan(frame, rate_deg=0.2, base_position=vec3(0.0, 1.2, 4.0), base_yaw=0.0):
  yaw = base_yaw + math.radians(rate_deg * frame)
  return base_position, yaw

def orbit(frame, radius=4.0, rate_deg=0.2, target=vec3(0.0, 1.0, 0.0), elevation=0.2):
  theta = math.radians(rate_deg * frame)
  x = target.x + radius * math.sin(theta)
  z = target.z + radius * math.cos(theta)

  position = vec3(x, target.y + elevation, z)

  yaw = -theta

  return position, yaw

def scripted(frame):
  base_pos = vec3(0.0, 1.2, 4.0)
  base_yaw = 0.0

  if frame < 60:
    return static(frame, base_position=base_pos, base_yaw=base_yaw)
  elif frame < 180:
    return dolly(frame - 60, speed=0.015, base_position=base_pos, base_yaw=base_yaw)
  elif frame < 240:
    dolly_end_pos, dolly_end_yaw = dolly(119, speed=0.015, base_position=base_pos, base_yaw=base_yaw)
    return pan(frame - 180, rate_deg=1.5, base_position=dolly_end_pos, base_yaw=dolly_end_yaw)
  else:
    return orbit(frame-240, radius=4.0, rate_deg=0.5)