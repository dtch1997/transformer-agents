from pathlib import Path

import numpy as np


def iterate_image_files(directory):
    image_files = []

    # Create a Path object for the given directory
    directory_path = Path(directory)

    # Iterate over all files and directories in the given directory and its subdirectories
    for path in directory_path.glob("**/*"):
        # Check if the path is a file and has an image file extension
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            image_files.append(str(path))

    return image_files


def boundary(inputs):
    col = inputs.shape[1]
    inputs = inputs.reshape(-1)
    lens = len(inputs)
    start = np.argmax(inputs)
    end = lens - 1 - np.argmax(np.flip(inputs))
    top = start // col
    bottom = end // col

    return top, bottom


def seg_to_box(seg_mask, size):
    top, bottom = boundary(seg_mask)
    left, right = boundary(seg_mask.T)
    left, top, right, bottom = (
        left / size,
        top / size,
        right / size,
        bottom / size,
    )  # we normalize the size of boundary to 0 ~ 1

    return [left, top, right, bottom]
