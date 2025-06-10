def colourblind_palette():
    alpha = 255
    palette = [
        [255, 255, 255, 0],
        [230, 159, 0, alpha],
        [86, 180, 233, alpha],
        [0, 158, 115, alpha],
        [240, 228, 66, alpha],
        [0, 114, 178, alpha],
        [213, 94, 0, alpha],
        [204, 121, 167, alpha],
        [0, 0, 0, alpha]
    ]
    palette.extend([[255 - t[0], 255 - t[1], 255 - t[2], alpha] for t in palette[1:-1]])
    return palette

# TODO add more palettes