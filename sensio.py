#%% Load dataset and annotations
#from utils.agave_dataset import AgaveDataset, get_transform, get_sample, show_image
from utils.voc_dataset import get_sample, plot_anns
from utils.voc_dataset import classes
from utils.anchors import AnchorUtils
from utils.ssd_loss import actn_to_bb
from utils.ssd_train import fit
import albumentations as A
from utils.ssd_model import SSD
from utils.ssd_loss import SSDLoss
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A

# Dataset and annotations
idx = 4445
img_np, anns = get_sample(idx)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#  Anchors
scales = [6, 3, 1]
centers = [(0.5, 0.5)] 
size_scales = [0.5]
aspect_ratios = [(1., 1.), (1.5, 0.8), (1.8, 0.4)]

sizes = [(s * a[0], s * a[1]) for s in size_scales for a in aspect_ratios]

k, anchors, grid_size = AnchorUtils.generate_anchors(scales, centers, sizes)
anchors = anchors.to(device)
grid_size = grid_size.to(device)

AnchorUtils.plot_anchors(img_np, anns, anchors, classes)
plt.show()
print(f"Anchors: {len(anchors)}, k: {k}")


# Model
n_classes = len(classes)
k_values = [3, 3, 3]


net = SSD(n_classes=n_classes, k=k_values).to(device)
input_tensor = torch.rand((64, 3, 100, 100)).to(device)

output = net(input_tensor)
print(output[0].shape, output[1].shape) # -> target: torch.Size([64, 138, 4]) torch.Size([64, 138, 2])


# Loss (SSDLoss)
criterion = SSDLoss(
    anchors=anchors,
    grid_size=grid_size,
    threshold=0.4,
)

targets = (torch.rand((64, 5, 4)).to(device), torch.randint(0, n_classes, (64, 5)).to(device))
loss = criterion(output, targets)
print(f"Loss: {loss.item()}")

#%%
# TODO: Fix the plot on the bbs, seems to be not unnormalized correctly to 100 x 100 target
#trans = A.Compose([
#    A.Resize(100, 100)
#], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

#labels, bbs = anns
#augmented = trans(**{'image': img_np, 'bboxes': bbs, 'labels': labels})
#img, bbs, labels = augmented['image'], augmented['bboxes'], augmented['labels']
#bbs_normalized = [AnchorUtils.norm(bb, img_np.shape[:2]) for bb in bbs]
#augmented = trans(image=img_np, bboxes=bbs_normalized, labels=labels)
#img, bbs_aug, labels_aug = augmented['image'], augmented['bboxes'], augmented['labels']
#bbs = [AnchorUtils.unnorm(bb, img_np.shape[:2]) for bb in bbs_aug]
#AnchorUtils.plot_anchors(img, (labels_aug, bbs), anchors, classes)
#plt.show()


#%% TODO: fix the tensors, are not in the the same device
trans = A.Compose([
    A.Resize(100, 100)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

labels, bbs = anns
augmented = trans(**{'image': img_np, 'bboxes': bbs, 'labels': labels})
img, bbs, labels = augmented['image'], augmented['bboxes'], augmented['labels']


img_tensor = torch.FloatTensor(img / 255.).permute(2,0,1).unsqueeze(0).to(device)
bb_norm = [AnchorUtils.norm(bb, img.shape[:2]) for bb in bbs]
bb_tensor = torch.FloatTensor(bb_norm).unsqueeze(0).to(device)
label_tensor = torch.tensor(labels).long().unsqueeze(0).to(device)

#%%
model = SSD(n_classes = len(classes), k=k)
fit(model, img_tensor, (bb_tensor, label_tensor), epochs=100)

#%%

def predict(model, X):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        bbs, labels = model(X)
        bbs = actn_to_bb(bbs[0], anchors, grid_size)
    return bbs, torch.max(torch.softmax(labels, axis=2)[0].cpu(), axis=1)  # Asegúrate de que las etiquetas estén en la CPU

bbs, (scores, labels) = predict(model, img_tensor)
bbs = [AnchorUtils.unnorm(bb.cpu().numpy(), img.shape[:2]) for bb in bbs] 
# %%
plot_anns(img, (labels, bbs))
plt.show()
#%%
plot_anns(img, (labels, bbs), bg=0)
plt.show()