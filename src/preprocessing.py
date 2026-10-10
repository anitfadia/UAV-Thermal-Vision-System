import cv2


# --------------------------------
# 1. NORMALIZATION
# --------------------------------

def normalize_image(image):

    normalized = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return normalized


# --------------------------------
# 2. CLAHE
# --------------------------------

def apply_clahe(image, clip_limit=2.0, tile_size=8):

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )

    enhanced = clahe.apply(image)

    return enhanced


# --------------------------------
# 3. BILATERAL FILTER
# --------------------------------

def apply_bilateral_filter(image):

    filtered = cv2.bilateralFilter(
        image,
        9,
        75,
        75
    )

    return filtered