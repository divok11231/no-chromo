from PIL import Image
import numpy as np

def get_complex_grid(width, height, center_x, center_y, zoom):
    # Calculate the span of the complex plane based on zoom
    # A zoom of 1.0 = a width of 3.0 units in the complex plane
    x_span = 3.0 / zoom
    y_span = x_span * (height / width)
    
    # Define the boundaries based on center and span
    x_min, x_max = center_x - x_span/2, center_x + x_span/2
    y_min, y_max = center_y - y_span/2, center_y + y_span/2
    
    # Generate 1D axes (y_max to y_min flips it for standard image orientation)
    x_axis = np.linspace(x_min, x_max, width)
    y_axis = np.linspace(y_max, y_min, height)
    
    # Create the 2D grid and combine into a complex matrix
    X, Y = np.meshgrid(x_axis, y_axis)
    return X + 1j * Y

dimen = 512
iterations = 80
scale_factor = 255.0 / iterations

# Setup the "camera" and generate the grid
c = get_complex_grid(dimen, dimen, center_x=-0.5, center_y=0, zoom=1.0)
z = np.zeros((dimen, dimen), dtype=np.complex128)

mask = np.full(c.shape, True, dtype=bool)
escape_counts = np.zeros(c.shape, dtype=float)

# do iterations of mandelbrot set
for i in range(iterations):
    z[mask] = z[mask]**2 + c[mask]
    diverged = np.abs(z) > 10.0
    escaping_now = diverged & mask
    
    if np.any(escaping_now):
        # i + 1 - log2(log2(|z|)) is jsut smoothening the curve by picking up individual point velocities
        z_abs = np.abs(z[escaping_now])
        escape_counts[escaping_now] = i + 1 - np.log2(np.log2(z_abs))
        
    mask &= ~diverged
    if not np.any(mask):
        break

# normalizing values between 0 and 1
norm = np.clip(escape_counts / iterations, 0, 1)
norm = np.power(norm, 0.6)

# dithering
noise = (np.random.random(norm.shape) - 0.5) / 255.0
norm = np.clip(norm + noise, 0, 1)

# sin gradient 
img = np.zeros((dimen, dimen, 3), dtype=np.uint8)

# Red
img[:, :, 0] = (np.sin(norm * np.pi / 2) * 255).astype(np.uint8)

# Blue
img[:, :, 2] = (np.sin(norm * np.pi) * 255).astype(np.uint8)

# Masking
img[escape_counts == 0] = [0, 0, 0]

display = Image.fromarray(img, 'RGB')
display.save('out.bmp')
display.show()



