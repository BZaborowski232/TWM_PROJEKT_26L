def get_bottle_bbox(label_path, image_w, image_h):

    if not label_path.exists():
        return None

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls_id = int(parts[0])

            if cls_id != 0:
                continue

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            bw = int(width * image_w)
            bh = int(height * image_h)

            bx = int((x_center * image_w) - bw / 2)
            by = int((y_center * image_h) - bh / 2)

            return (bx, by, bw, bh)

    return None