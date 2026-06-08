import cv2
import numpy as np

SATURATION_THRESHOLD = 60
EDGE_THRESHOLD = 0.06
STD_GRAY_THRESHOLD = 29

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

    x1 = bx
    x2 = bx + bw

    y1 = by + int(bh * 0.25)
    y2 = by + int(bh * 0.60)

    return (
        x1,
        y1,
        x2,
        y2
    )


def detect_damaged_label(image,bottle_bbox=None):

    x1, y1, x2, y2 = get_label_roi(image,bottle_bbox)

    roi = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]

    mean_saturation = float(
        np.mean(saturation)
    )

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    std_gray = float(
        np.std(gray)
    )   

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_ratio = (
        np.count_nonzero(edges)
        / edges.size
    )

    std_saturation = np.std(
    hsv[:, :, 1]
    )

    damaged = (
        std_gray < STD_GRAY_THRESHOLD
        and
        mean_saturation < SATURATION_THRESHOLD
    )

    score = max(
        max(
            0,
            SATURATION_THRESHOLD - mean_saturation
        ) / SATURATION_THRESHOLD,

        max(
            0,
            EDGE_THRESHOLD - edge_ratio
        ) / EDGE_THRESHOLD
    )

    confidence = min(
        score,
        1.0
    )

    return {

        "label":
            "damaged_label"
            if damaged
            else "good",

        "damaged":
            damaged,

        "confidence":
            float(confidence),

        "mean_saturation":
            float(mean_saturation),

        "edge_ratio":
            float(edge_ratio),

        "std_saturation":
            float(std_saturation),

        "std_gray":
            float(std_gray),

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