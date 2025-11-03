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
    return (edge > width // 2).astype(np.uint8) * 255

def show_scene(scene, title="Scene"):
    plt.figure(figsize=(6, 3))
    plt.imshow(scene, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()
