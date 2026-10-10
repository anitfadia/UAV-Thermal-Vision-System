import cv2


def detect_candidates(image):

    # --------------------------------
    # STEP 1: THRESHOLDING
    # --------------------------------

    _, threshold = cv2.threshold(
        image,
        180,
        255,
        cv2.THRESH_BINARY
    )

    print("Thresholding completed!")


    # --------------------------------
    # STEP 2: MORPHOLOGICAL OPERATIONS
    # --------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    cleaned = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )

    print("Morphological operations completed!")


    # --------------------------------
    # STEP 3: FIND CONTOURS
    # --------------------------------

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print("Total contours:", len(contours))


    # --------------------------------
    # STEP 4: FILTER HUMAN CANDIDATES
    # --------------------------------

    detections = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small regions
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect_ratio = h / float(w)

        # Human-like shape filtering
        if (
            h > 40
            and w > 15
            and aspect_ratio > 1.0
            and aspect_ratio < 4.5
        ):

            detections.append(
                (x, y, w, h)
            )


    print(
        "Human candidates detected:",
        len(detections)
    )

    return detections