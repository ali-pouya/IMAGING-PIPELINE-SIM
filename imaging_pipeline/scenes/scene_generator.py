import numpy as np
import matplotlib.pyplot as plt

def generate_barcode_scene(width=512, height=128, stripe_width=8):
    """Generate a synthetic 1D barcode pattern."""
    x = np.arange(width)
    bars = ((x // stripe_width) % 2) * 255
    scene = np.tile(bars, (height, 1))
    return scene.astype(np.uint8)

def generate_gradient_scene(width=512, height=512):
    """Generate a horizontal grayscale gradient."""
    gradient = np.tile(np.linspace(0, 255, width), (height, 1))
    return gradient.astype(np.uint8)

def generate_slanted_edge(width=512, height=512, angle_deg=5):
    """Generate a slanted edge for MTF measurement."""
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    edge = (xv * np.cos(np.deg2rad(angle_deg)) + yv * np.sin(np.deg2rad(angle_deg)))
    #return edge
    return (edge > width // 2).astype(np.uint8) * 255

def generate_siemens_star(size=512, spokes=55):
    """
    Generate a binary Siemens-star resolution target.
    The Siemens star is often used to visualize optical blur
    and evaluate MTF or PSF symmetry. It consists of alternating
    black and white wedges radiating from the center.

    Parameters
    ----------
    size : int
        Width and height of the square output image in pixels.
    spokes : int
        Number of black–white wedge pairs (so 36 gives 72 total wedges).

    Returns
    -------
    pattern : (size, size) ndarray of float
        Binary image (0 = black, 1 = white) containing the Siemens star.
    """
    # ------------------------------------------------------------
    # 1. Create a coordinate grid.
    # np.indices returns two 2-D arrays of pixel indices:
    #   y[i,j] gives the row number (vertical coordinate)
    #   x[i,j] gives the column number (horizontal coordinate)
    # We subtract size/2 to center the coordinate system at the image center
    # instead of the top-left corner.
    # ------------------------------------------------------------
    y, x = np.indices((size, size)) - size/2
    # ------------------------------------------------------------
    # 2. Compute the polar angle for each pixel.
    # arctan2(y, x) returns the angle (in radians) between the positive x-axis
    # and the line from the origin to point (x, y).
    # Range is (-pi, +pi].
    # ------------------------------------------------------------
    angle = np.arctan2(y, x)
    # ------------------------------------------------------------
    # 3. Create the alternating wedge pattern.
    # Multiply the angle by the desired number of spokes to control how many
    # times the sine function flips sign around the circle.
    #
    #   sin(spokes * angle) > 0  → white
    #   sin(spokes * angle) <= 0 → black
    #
    # The resulting boolean array is converted to float (1.0 / 0.0).
    # ------------------------------------------------------------
    pattern = (np.sin(spokes * angle) > 0).astype(float)
    # apply a circular mask to remove the square edges:
    r = np.sqrt(x**2 + y**2)
    mask = (r <= size/2)
    pattern[~mask] = 1
    #pattern[r > size/2] = 0.5  # gray background outside circle
    return pattern

def show_scene(scene, title="Scene"):
    plt.figure(figsize=(6, 3))
    plt.imshow(scene, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()
