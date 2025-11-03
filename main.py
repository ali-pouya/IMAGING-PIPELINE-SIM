from imaging_pipeline.scenes.scene_generator import (
    generate_barcode_scene,
    generate_gradient_scene,
    generate_slanted_edge,
    show_scene,
)

def main():
    barcode = generate_barcode_scene()
    show_scene(barcode, "Synthetic Barcode Scene")

    gradient = generate_gradient_scene()
    show_scene(gradient, "Grayscale Gradient Scene")

    edge = generate_slanted_edge()
    show_scene(edge, "Slanted Edge Scene")

if __name__ == "__main__":
    main()
