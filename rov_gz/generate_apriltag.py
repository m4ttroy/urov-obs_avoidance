
#!/usr/bin/env python3

import ctypes
import ctypes.util
from pathlib import Path
from PIL import Image

# AprilTag configuration
FAMILY = "tag36h11"
TAG_ID = 0

# Physical black tag dimensions
TAG_SIZE_M = 0.20

# tag36h11 uses:
# width_at_border = 8
# total_width = 10
#
# Therefore the image includes a one-cell
# white quiet margin on every side.

TOTAL_CELLS = 10
PIXELS_PER_CELL = 100
OUTPUT_SIZE = TOTAL_CELLS * PIXELS_PER_CELL

OUTPUT = Path(
    "models/apriltag_36h11/"
    "materials/textures/tag36h11_0.png"
)


class ImageU8(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("stride", ctypes.c_int32),
        ("buf", ctypes.POINTER(ctypes.c_uint8)),
    ]


# Find the official AprilTag shared library
library = ctypes.util.find_library("apriltag")

if library is None:
    raise RuntimeError(
        "AprilTag library not found. "
        "Install it using: sudo apt install libapriltag-dev"
    )

lib = ctypes.CDLL(library)

# Create the 36h11 family
lib.tag36h11_create.restype = ctypes.c_void_p
family = lib.tag36h11_create()

if not family:
    raise RuntimeError("Could not create tag36h11 family")

# Generate the specified tag
lib.apriltag_to_image.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int
]
lib.apriltag_to_image.restype = ctypes.POINTER(ImageU8)

tag_image = lib.apriltag_to_image(family, TAG_ID)

if not tag_image:
    raise RuntimeError("AprilTag image generation failed")

img = tag_image.contents

if img.width != TOTAL_CELLS or img.height != TOTAL_CELLS:
    raise RuntimeError(
        f"Unexpected image dimensions: {img.width}x{img.height}"
    )

# Read the generated grayscale pixels
raw = ctypes.string_at(
    img.buf,
    img.height * img.stride
)

image = Image.frombytes(
    "L",
    (img.width, img.height),
    raw,
    "raw",
    "L",
    img.stride
)

# Enlarge without interpolation.
# Nearest-neighbor preserves sharp black/white edges.
image = image.resize(
    (OUTPUT_SIZE, OUTPUT_SIZE),
    Image.Resampling.NEAREST
)

# Save PNG
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT)

print("AprilTag generated successfully!")
print(f"Family: {FAMILY}")
print(f"ID: {TAG_ID}")
print(f"Black tag size: {TAG_SIZE_M} m")
print(f"Image resolution: {OUTPUT_SIZE} x {OUTPUT_SIZE}")
print(f"Saved to: {OUTPUT.resolve()}")

# Release C resources
lib.image_u8_destroy.argtypes = [
    ctypes.POINTER(ImageU8)
]
lib.image_u8_destroy(tag_image)

lib.tag36h11_destroy.argtypes = [ctypes.c_void_p]
lib.tag36h11_destroy(family)
