def colourblind_palette():
    palette = [
        [255, 255, 255, 0],
        [230, 159, 0, 255],
        [86, 180, 233, 255],
        [0, 158, 115, 255],
        [240, 228, 66, 255],
        [0, 114, 178, 255],
        [213, 94, 0, 255],
        [204, 121, 167, 255],
        [0, 0, 0, 255]
    ]
    palette.extend([[255 - t[0], 255 - t[1], 255 - t[2], 255] for t in palette[1:-1]])
    return palette

# TODO add more palettes