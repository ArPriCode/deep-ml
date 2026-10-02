import numpy as np

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign each anchor a training label via IoU matching.
    """

    anchors = np.asarray(anchors, dtype=float)
    gt_boxes = np.asarray(gt_boxes, dtype=float)

    n = len(anchors)
    m = len(gt_boxes)

    # No anchors
    if n == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    # No ground-truth boxes
    if m == 0:
        return np.zeros(n, dtype=int), np.full(n, -1, dtype=int)

    # Anchor dimensions
    ax1 = anchors[:, 0][:, None]
    ay1 = anchors[:, 1][:, None]
    ax2 = anchors[:, 2][:, None]
    ay2 = anchors[:, 3][:, None]

    # GT dimensions
    gx1 = gt_boxes[:, 0][None, :]
    gy1 = gt_boxes[:, 1][None, :]
    gx2 = gt_boxes[:, 2][None, :]
    gy2 = gt_boxes[:, 3][None, :]

    # Intersection
    ix1 = np.maximum(ax1, gx1)
    iy1 = np.maximum(ay1, gy1)
    ix2 = np.minimum(ax2, gx2)
    iy2 = np.minimum(ay2, gy2)

    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)

    intersection = iw * ih

    # Areas
    anchor_area = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    gt_area = np.maximum(0, gx2 - gx1) * np.maximum(0, gy2 - gy1)

    union = anchor_area + gt_area - intersection

    # IoU
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0
    )

    # Best GT for every anchor
    best_gt = np.argmax(iou, axis=1)
    best_iou = iou[np.arange(n), best_gt]

    # Initially ignore everything
    labels = np.full(n, -1, dtype=int)
    matched_gt = np.full(n, -1, dtype=int)

    # Negative
    labels[best_iou < neg_threshold] = 0

    # Positive
    positive = best_iou >= pos_threshold
    labels[positive] = 1
    matched_gt[positive] = best_gt[positive]

    # Force every GT to have a positive anchor
    for j in range(m):
        anchor_idx = np.argmax(iou[:, j])

        labels[anchor_idx] = 1
        matched_gt[anchor_idx] = j

    return labels, matched_gt