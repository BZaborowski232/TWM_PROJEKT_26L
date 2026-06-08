import cv2
import numpy as np

BROWN_THRESHOLD = 0.08
DARK_THRESHOLD = 0.20
SATURATION_THRESHOLD = 65

def get_debris_roi(image):

    h, w = image.shape[:2]

    x1 = int(w * 0.40)
    x2 = int(w * 0.60)

    y1 = int(h * 0.35)
    y2 = int(h * 0.82)

    return x1, y1, x2, y2


def detect_debris(image):

    x1, y1, x2, y2 = get_debris_roi(image)

    roi = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    dark_mask = v_ch < 85

    brown_mask = (
        (h_ch >= 5)
        &
        (h_ch <= 35)
        &
        (s_ch > 35)
        &
        (v_ch < 230)
    )

    dark_ratio = (
        np.count_nonzero(dark_mask)
        / dark_mask.size
    )

    brown_ratio = (
        np.count_nonzero(brown_mask)
        / brown_mask.size
    )

    mean_saturation = float(
        np.mean(s_ch)
    )

    mean_value = float(
        np.mean(v_ch)
    )

    debris = (
        brown_ratio > BROWN_THRESHOLD
        or dark_ratio > DARK_THRESHOLD
        or mean_saturation > SATURATION_THRESHOLD
    )

    score = max(
        brown_ratio / BROWN_THRESHOLD,
        dark_ratio / DARK_THRESHOLD,
        mean_saturation / SATURATION_THRESHOLD
    )

    if debris:
        confidence = min(score, 1.0)
    else:
        confidence = max(0.0, 1.0 - score)

    return {
        "label": (
            "debris"
            if debris
            else "good"
        ),

        "debris": debris,

        "dark_ratio": float(
            dark_ratio
        ),

        "brown_ratio": float(
            brown_ratio
        ),

        "mean_saturation": float(
            mean_saturation
        ),

        "mean_value": float(
            mean_value
        ),

        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2,

        "confidence": float(confidence)
    }


def draw_debris_result(image, result):

    annotated = image.copy()

    x1 = result["roi_x1"]
    y1 = result["roi_y1"]
    x2 = result["roi_x2"]
    y2 = result["roi_y2"]

    color = (
        (0, 0, 255)
        if result["debris"]
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