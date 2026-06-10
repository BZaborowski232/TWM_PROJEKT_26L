import cv2
import numpy as np


LABEL_PRESENT_THRESHOLD = 0.03


def get_label_roi(image, bottle_bbox):

    if bottle_bbox is None:

        h, w = image.shape[:2]

        return (
            int(w * 0.35),
            int(h * 0.35),
            int(w * 0.65),
            int(h * 0.65)
        )

    bx, by, bw, bh = bottle_bbox

    y1 = by + int(bh * 0.29)
    y2 = by + int(bh * 0.45)

    x1 = bx + int(bw * 0.10)
    x2 = bx + int(bw * 0.90)

    return (
        x1,
        y1,
        x2,
        y2
    )


def detect_damaged_label(
    image,
    bottle_bbox=None
):

    x1, y1, x2, y2 = get_label_roi(
        image,
        bottle_bbox
    )

    roi = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------
    # Szukanie niebieskiego fragmentu etykiety
    # --------------------------------------------------

    lower_blue = np.array([
        95,
        50,
        20
    ])

    upper_blue = np.array([
        140,
        255,
        255
    ])

    blue_mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    blue_ratio = (
        np.count_nonzero(blue_mask)
        / blue_mask.size
    )

    # --------------------------------------------------
    # Największe skupisko niebieskiego
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest_blue_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > largest_blue_area:
            largest_blue_area = area

    largest_blue_ratio = (
        largest_blue_area
        / (roi.shape[0] * roi.shape[1])
    )

    # --------------------------------------------------
    # Decyzja
    # --------------------------------------------------

    label_present = (
        largest_blue_ratio >
        LABEL_PRESENT_THRESHOLD
    )

    damaged = not label_present

    confidence = min(
        largest_blue_ratio
        / LABEL_PRESENT_THRESHOLD,
        1.0
    )

    return {

        "label":
            "good"
            if label_present
            else "damaged_label",

        "damaged":
            damaged,

        "confidence":
            float(confidence),

        "blue_ratio":
            float(blue_ratio),

        "largest_blue_ratio":
            float(largest_blue_ratio),

        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2
    }


def draw_damaged_label_result(
    image,
    result
):

    annotated = image.copy()

    x1 = result["roi_x1"]
    y1 = result["roi_y1"]
    x2 = result["roi_x2"]
    y2 = result["roi_y2"]

    color = (
        (0, 0, 255)
        if result["damaged"]
        else (0, 255, 0)
    )

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    text = (
        f"{result['label']} "
        f"blue={result['largest_blue_ratio']:.3f} "
        f"conf={result['confidence']:.2f}"
    )

    cv2.putText(
        annotated,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    return annotated