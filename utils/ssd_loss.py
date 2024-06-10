# TODO: linting
"""
Utilities for SSD loss function.
"""
import torch
import torch.nn as nn
import torchvision

def actn_to_bb(actn, anchors, grid_size):
    actn_bbs = torch.tanh(actn)
    actn_p1 = anchors[:, :2] + actn_bbs[:, :2] * grid_size * 0.5
    actn_p2 = anchors[:, 2:] + actn_bbs[:, 2:] * grid_size * 0.5
    return torch.cat([actn_p1, actn_p2], dim=1)

def map_to_ground_truth(overlaps):
    prior_overlap, prior_idx = overlaps.max(1)
    gt_overlap, gt_idx = overlaps.max(0)
    gt_overlap[prior_idx] = 1.99
    for i, o in enumerate(prior_idx): gt_idx[o] = i
    return gt_overlap, gt_idx

class SSDLoss(torch.nn.Module):
    def __init__(self, anchors, grid_size, threshold=0.4):
        super().__init__()
        self.loc_loss = torch.nn.L1Loss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.anchors = anchors
        self.grid_size = grid_size
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, preds, target):
        pred_bbs, pred_cs = preds 
        tar_bbs, c_t = target
        loc_loss, clas_loss = 0, 0
        for pred_bb, pred_c, tar_bb, tar_c in zip(pred_bbs, pred_cs, tar_bbs, c_t):
            labels = torch.zeros(len(self.anchors)).long().to(self.device)
            if tar_bb.shape[0] is not 0:
                overlaps = torchvision.ops.box_iou(tar_bb, self.anchors)
                gt_overlap, gt_idx = map_to_ground_truth(overlaps)
                pos = gt_overlap > self.threshold
                pos_idx = torch.nonzero(pos)[:,0].to(self.device)
                tar_idx = gt_idx[pos_idx]
                pred_bb = actn_to_bb(pred_bb, self.anchors, self.grid_size)
                _anchors = pred_bb[pos_idx]
                tar_bb = tar_bb[tar_idx]
                loc_loss += self.loc_loss(_anchors, tar_bb)
                labels[pos_idx] = tar_c[tar_idx]
                clas_loss += self.class_loss(pred_c, labels)
        return clas_loss + loc_loss

if __name__ == "__main__":
    # Simulación de datos para prueba
    anchors = torch.rand((46, 4))  # Ejemplo de anclas
    grid_size = torch.tensor([0.1, 0.1])  # Ejemplo de tamaño de cuadrícula
    ssd_loss = SSDLoss(anchors, grid_size)

    # Predicciones de ejemplo
    preds = (
        torch.rand((2, 46, 4)),  # pred_bbs
        torch.rand((2, 46, 1))   # pred_cs
    )

    # Objetivos de ejemplo
    target = (
        torch.rand((2, 5, 4)),   # tar_bbs (suponiendo un máximo de 5 cajas por imagen)
        torch.randint(0, 1, (2, 5))  # c_t
    )

    # Calcular pérdida
    loss = ssd_loss(preds, target)
    print("Loss:", loss.item())
