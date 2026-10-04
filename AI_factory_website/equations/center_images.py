"""Center rendered equations on their fixed hero and card canvases."""

from pathlib import Path
import sys

from PIL import Image, ImageChops


def center_equation(path: Path) -> None:
    """Recenter visible pixels without changing their scale or canvas size."""

    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    content_box = ImageChops.difference(image, background).getbbox()
    if content_box is None:
        raise RuntimeError(f"Equation image is blank: {path}")

    content = image.crop(content_box)
    left = (image.width - content.width) // 2
    top = (image.height - content.height) // 2
    background.paste(content, (left, top))
    background.save(path)


def main() -> None:
    """Center every generated PNG in the provided output directories."""

    for directory_name in sys.argv[1:]:
        directory = Path(directory_name)
        for path in sorted(directory.glob("*.png")):
            center_equation(path)


if __name__ == "__main__":
    main()
