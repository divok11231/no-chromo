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

img_data = np.zeros((dimen, dimen), dtype=np.float32)

# do iterations of mandelbrot set
for i in range(iterations):
	
	z = z**2 + c

	# mask for values that have not diverged
	not_diverged = (np.abs(z)<4).astype(np.float32)

	# add non-divergent spots to image data
	img_data += not_diverged

# set non-diverged values to 0
img_data[img_data==img_data.max()] = 0

# Convert to RGB image
img = np.zeros((dimen, dimen, 3), dtype=np.uint8)
img[:, :, 0] = (img_data*scale_factor).astype(np.uint8)

display = Image.fromarray(img, 'RGB')
display.save('out.bmp')
display.show()



