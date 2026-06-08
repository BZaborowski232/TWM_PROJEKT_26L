import cv2
import numpy as np

FILL_THRESHOLD = 0.80


def get_fill_roi(image):

    h, w = image.shape[:2]

    x1 = int(w * 0.42)
    x2 = int(w * 0.58)

    y1 = int(h * 0.20)
    y2 = int(h * 0.85)

    return x1, y1, x2, y2


def detect_fill_level(image):

    x1, y1, x2, y2 = get_fill_roi(image)

    roi = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    liquid_mask = (
        (s_ch > 25)
        &
        (v_ch < 240)
    )

    vertical_profile = np.mean(
        liquid_mask.astype(np.uint8),
        axis=1
    )

    liquid_rows = np.where(
        vertical_profile > 0.15
    )[0]

    if len(liquid_rows) == 0:

        fill_ratio = 0.0

        underfilled = True

        score = 0.0

        confidence = 1.0

        return {

            "label":
                "underfilled",

            "underfilled":
                underfilled,

            "fill_ratio":
                fill_ratio,

            "score":
                score,

            "confidence":
                confidence,

            "fill_y":
                None,

            "roi_x1": x1,
            "roi_y1": y1,
            "roi_x2": x2,
            "roi_y2": y2
        }

    fill_y = liquid_rows[0]

    roi_height = roi.shape[0]

    fill_ratio = (
        roi_height - fill_y
    ) / roi_height

    underfilled = (
        fill_ratio
        < FILL_THRESHOLD
    )

    score = (
        fill_ratio
        / FILL_THRESHOLD
    )

    if underfilled:
        confidence = max(
            0.0,
            1.0 - score
        )
    else:
        confidence = min(
            score,
            1.0
        )

    return {

        "label":
            "underfilled"
            if underfilled
            else "normal_fill",

        "underfilled":
            underfilled,

        "fill_ratio":
            float(fill_ratio),

        "score":
            float(score),

        "confidence":
            float(confidence),

        "fill_y":
            int(fill_y),

        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2
    }


def draw_fill_result(image, result):

    annotated = image.copy()

    x1 = result["roi_x1"]
    y1 = result["roi_y1"]
    x2 = result["roi_x2"]
    y2 = result["roi_y2"]

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    if result["fill_y"] is not None:

        level_y = (
            y1
            + result["fill_y"]
        )

        cv2.line(
            annotated,
            (x1, level_y),
            (x2, level_y),
            (0, 255, 255),
            2
        )

    color = (
        (0, 0, 255)
        if result["underfilled"]
        else (0, 255, 0)
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