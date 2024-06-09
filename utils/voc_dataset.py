#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import torchvision
import random

train = torchvision.datasets.VOCDetection('./data', download=False)
classes = ["background","aeroplane","bicycle","bird","boat",
    "bottle","bus","car","cat","chair","cow","diningtable","dog",
    "horse","motorbike","person","pottedplant","sheep","sofa",
    "train","tvmonitor"
    ]

def get_sample(ix):
    img, label = train[ix]
    img_np = np.array(img)
    anns = label['annotation']['object']
    if type(anns) is not list:
        anns = [anns]
    labels = np.array([classes.index(ann['name']) for ann in anns])
    bbs = [ann['bndbox'] for ann in anns]
    bbs = np.array([[int(bb['xmin']), int(bb['ymin']),int(bb['xmax']),int(bb['ymax'])] for bb in bbs])
    anns = (labels, bbs)
    return img_np, anns

def plot_anns(img, anns, ax=None, bg=-1):
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    labels, bbs = anns
    for lab, bb in zip(labels, bbs):
        if bg == -1 or lab != bg:
            x, y, xm, ym = bb
            w, h = xm - x, ym - y
            rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            text = ax.text(x, y - 10, classes[lab], {'color': 'red'})
            text.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            ax.add_patch(rect)