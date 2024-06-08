"""Anchor box utilities."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from PIL import Image

class AnchorUtils:
    @staticmethod
    def norm(bb, shape):
        h, w = shape
        return np.array([bb[0]/w, bb[1]/h, bb[2]/w, bb[3]/h])

    @staticmethod
    def unnorm(bb, shape):
        h, w = shape
        return np.array([bb[0]*w, bb[1]*h, bb[2]*w, bb[3]*h])

    @staticmethod
    def xyxy2xywh(bb):
        return torch.stack([bb[:,0], bb[:,1], bb[:,2]-bb[:,0], bb[:,3]-bb[:,1]], axis=1)

    @staticmethod
    def generate_anchors(scales, centers, sizes):
        anchors = []
        grid_size = []
        for s in scales:
            for (x, y) in centers:
                for (w, h) in sizes:
                    for i in range(s):
                        for j in range(s):
                            anchors.append(np.array([x + i - w / 2, y + j - h / 2, x + i + w / 2, y + j + h / 2]) / s)
                            grid_size.append(np.array([1. / s, 1. / s]))
        anchors = np.array(anchors)  # Convert list of ndarrays to a single ndarray
        grid_size = np.array(grid_size)  # Convert list of ndarrays to a single ndarray
        return torch.tensor(anchors).float(), torch.tensor(grid_size).float()

    @staticmethod
    def plot_anchors(img, anns, anchors, classes, ax=None, overlap=False):
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        labels, bbs = anns

        anchors = AnchorUtils.xyxy2xywh(anchors)
        _anchors = np.array([AnchorUtils.unnorm(a, img.shape[:2]) for a in anchors])
        for a in _anchors:
            x, y, w, h = a
            rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
            ax.add_patch(rect)

        for lab, bb in zip(labels, bbs):
            x, y, xm, ym = bb
            w, h = xm - x, ym - y
            rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            text = ax.text(x, y - 10, classes[int(lab)], {'color': 'red'})  # Asegúrate de que `lab` sea un índice entero
            text.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            ax.add_patch(rect)
        
if __name__ == "__main__":
    bbox = [50, 30, 200, 150]
    shape = (400, 600)
    scales = [8, 16, 32]
    centers = [(50, 50), (100, 100)]
    sizes = [(20, 20), (40, 40)]

    norm_bbox = AnchorUtils.norm(bbox, shape)
    print("Normalized BBox:", norm_bbox)

    unnorm_bbox = AnchorUtils.unnorm(norm_bbox, shape)
    print("Unnormalized BBox:", unnorm_bbox)

    bb_torch = torch.tensor([bbox])
    xywh_bbox = AnchorUtils.xyxy2xywh(bb_torch)
    print("XYWH BBox:", xywh_bbox)

    k, anchors, grid_size = AnchorUtils.generate_anchors(scales, centers, sizes)
    print("Anchors:", anchors)
    print("Grid Size:", grid_size)

    # Example usage of plot_anchors
    def get_sample():
        img = Image.open("data/datasets/agaveHD-1/valid/DSC00637_geotag_4_jpg.rf.999a9a2d393f7c22ee71b94ff72f8b8b.jpg")
        img_np = np.array(img)
        anns = (np.array([0]), np.array([[50, 30, 200, 150]]))
        return img_np, anns

    img_np, anns = get_sample()

    scales = [6]
    centers = [(0.5, 0.5)]
    size_scales = [0.5]
    aspect_ratios = [(1., 1.), (1.5, 0.8)]
    sizes = [(s*a[0], s*a[1]) for s in size_scales for a in aspect_ratios]
    k, anchors, grid_size = AnchorUtils.generate_anchors(scales, centers, sizes)

    classes = ["agave"]

    AnchorUtils.plot_anchors(img_np, anns, anchors, classes)
    plt.show()
