import cv2
import numpy as np

CAP_HEIGHT_THRESHOLD = 0.25


def get_cap_roi(image, bottle_bbox):

    if bottle_bbox is None:

        h, w = image.shape[:2]

        return (
            int(w * 0.30),
            int(h * 0.03),
            int(w * 0.70),
            int(h * 0.25)
        )

    bx, by, bw, bh = bottle_bbox

    x1 = bx + int(bw * 0.20)
    x2 = bx + int(bw * 0.80)

    y1 = by
    y2 = by + int(bh * 0.18)

    return x1, y1, x2, y2


def detect_cap(image, bottle_bbox=None):

    x1, y1, x2, y2 = get_cap_roi(
        image,
        bottle_bbox
    )

    roi = image[y1:y2, x1:x2]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    # MATLAB Color Thresholder
    lower_cap = np.array([
        104,
        143,
        169
    ])

    upper_cap = np.array([
        112,
        255,
        255
    ])

    cap_mask = cv2.inRange(
        hsv,
        lower_cap,
        upper_cap
    )

    contours, _ = cv2.findContours(
        cap_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour = None
    largest_area = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area > largest_area:

            largest_area = area
            largest_contour = cnt

    cap_height_ratio = 0.0
    cap_width_ratio = 0.0
    cap_area_ratio = 0.0

    if largest_contour is not None:

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        cap_height_ratio = (
            h / roi.shape[0]
        )

        cap_width_ratio = (
            w / roi.shape[1]
        )

        cap_area_ratio = (
            largest_area
            / (roi.shape[0] * roi.shape[1])
        )

    cap_present = (
        cap_height_ratio >
        CAP_HEIGHT_THRESHOLD
    )

    confidence = min(
        cap_height_ratio
        / CAP_HEIGHT_THRESHOLD,
        1.0
    )

    return {

        "label":
            "cap_present"
            if cap_present
            else "no_cap",

        "cap_present":
            cap_present,

        "confidence":
            float(confidence),

        "cap_height_ratio":
            float(cap_height_ratio),

        "cap_width_ratio":
            float(cap_width_ratio),

        "cap_area_ratio":
            float(cap_area_ratio),

        "largest_area":
            float(largest_area),

        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2,

        "mask":
            cap_mask,

        "bottle_bbox": bottle_bbox
    }


def draw_cap_result(
    image,
    result
):

    annotated = image.copy()

    if result["bottle_bbox"] is not None:

        bx, by, bw, bh = result["bottle_bbox"]

        cv2.rectangle(
            annotated,
            (bx, by),
            (bx + bw, by + bh),
            (255, 0, 0),
            2
        )

    x1 = result["roi_x1"]
    y1 = result["roi_y1"]
    x2 = result["roi_x2"]
    y2 = result["roi_y2"]

    color = (
        (0, 255, 0)
        if result["cap_present"]
        else (0, 0, 255)
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
        f"h={result['cap_height_ratio']:.3f} "
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