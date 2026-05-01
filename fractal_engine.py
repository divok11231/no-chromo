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
# Track minimum distance to axes (Pickover Stalks)
trap_dist = np.full(c.shape, 1e10)

# do iterations of mandelbrot set
for i in range(iterations):
    z[mask] = z[mask]**2 + c[mask]
    
    # Update orbit trap distance for active points
    # Pickover Stalks: distance to X or Y axis
    dist_x = np.abs(z[mask].real)
    dist_y = np.abs(z[mask].imag)
    trap_dist[mask] = np.minimum(trap_dist[mask], np.minimum(dist_x, dist_y))
    
    diverged = np.abs(z) > 10.0
    escaping_now = diverged & mask
    
    if np.any(escaping_now):
        # i + 1 - log2(log2(|z|)) is jsut smoothening the curve by picking up individual point velocities
        z_abs = np.abs(z[escaping_now])
        escape_counts[escaping_now] = i + 1 - np.log2(np.log2(z_abs))
        
    mask &= ~diverged
    if not np.any(mask):
        break

# normalizing escape values
norm = np.clip(escape_counts / iterations, 0, 1)
norm = np.power(norm, 0.6)

# Normalize orbit trap distance (logarithmic scale for "smoky" feel)
# 1e-6 prevents log(0), 10.0 is a tuning factor for contrast
trap_norm = np.clip(-np.log(trap_dist + 1e-6) / 10.0, 0, 1)

combined_norm = np.clip(norm * 0.3 + trap_norm * 0.7, 0, 1)

# visible dithering for chromostereopsis
dither_noise = (np.random.random(combined_norm.shape) - 0.5) * 0.15
dithered_norm = np.clip(combined_norm + dither_noise, 0, 1)

# discrete red and blue coloring with black background
img = np.zeros((dimen, dimen, 3), dtype=np.uint8)

bg_threshold = 0.2
color_threshold = 0.6

# Blue channel: mid-range values
blue_mask = (dithered_norm >= bg_threshold) & (dithered_norm < color_threshold)
img[blue_mask, 2] = 255

# Red channel: high-range values
red_mask = (dithered_norm >= color_threshold)
img[red_mask, 0] = 255

img[escape_counts == 0] = [255, 0, 0]

display = Image.fromarray(img, 'RGB')
display.save('out.bmp')
display.show()



