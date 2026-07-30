import taichi as ti
from taichi.math import vec3

# Importing from your file paths
from app.engine.tracer.camera import PinholeCamera
from app.engine.tracer.reference_pass import render_target

def main():
  ti.init(arch=ti.cuda)

  width, height = 512, 512
  image_buffer = ti.math.vec3.field(shape=(width, height))

  cam = PinholeCamera(position=vec3(0.0, 0.0, -5.0), yaw_angle=90.0, pitch_angle=0.0)

  # Call the kernel
  render_target(image_buffer, cam, spp=256, width=width, height=height, fov=90.0)

  # Display the result using Taichi's built-in GUI
  gui = ti.GUI("Project Albedo", res=(width, height)) # type: ignore
  while gui.running:
    gui.set_image(image_buffer)
    gui.show()

if __name__ == "__main__":
    main()