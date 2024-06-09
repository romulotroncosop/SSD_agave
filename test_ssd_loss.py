# test_ssd_loss.py
import torch
from torch.utils.data import DataLoader
from utils.agave_dataset import AgaveDataset, get_transform
from utils.ssd_loss import SSDLoss
from utils.ssd_model import SSD
from utils.anchors import AnchorUtils

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    boxes = [t['boxes'] for t in targets]
    labels = [t['labels'] for t in targets]
    return images, boxes, labels

def main():
    # Configuraciones
    root_dir = 'data/datasets/agaveHD-1'  # Cambia esta ruta a donde tienes tu dataset
    batch_size = 2
    num_classes = 2  # Incluyendo la clase de fondo

    # Inicializar dataset y dataloader
    dataset = AgaveDataset(root=root_dir, image_set='train', transform=get_transform(train=True))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Definir anclas y tamaño de la cuadrícula
    scales = [6, 3, 1]
    centers = [(0.5, 0.5)]
    size_scales = [0.5]
    aspect_ratios = [(1., 1.), (1.5, 0.8), (1.8, 0.4)]
    sizes = [(s * a[0], s * a[1]) for s in size_scales for a in aspect_ratios]
    k, anchors, grid_size = AnchorUtils.generate_anchors(scales, centers, sizes)

    # Inicializar el cálculo de la pérdida
    ssd_loss = SSDLoss(anchors, grid_size)

    # Inicializar el modelo SSD
    net = SSD(n_classes=num_classes, k=k)

    # Asegurar que el número de anclas y predicciones coinciden
    dummy_input = torch.rand((2, 3, 100, 100))
    pred_bbs, pred_cs = net(dummy_input)
    assert pred_bbs.size(1) == anchors.size(0), f"El número de predicciones ({pred_bbs.size(1)}) no coincide con el número de anclas ({anchors.size(0)})"
    print("El número de predicciones coincide con el número de anclas.")

    # Iterar sobre el dataloader
    for data in dataloader:
        images, boxes, labels = data

        # Asegúrate de que todas las listas tengan el mismo tamaño
        max_boxes = max([b.size(0) for b in boxes])
        padded_boxes = [torch.cat([b, torch.zeros(max_boxes - b.size(0), 4)], dim=0) for b in boxes]
        padded_labels = [torch.cat([l, torch.zeros(max_boxes - l.size(0), dtype=torch.int64)], dim=0) for l in labels]

        # Convertir a tensores
        boxes_tensor = torch.stack(padded_boxes)
        labels_tensor = torch.stack(padded_labels)

        # Obtener predicciones del modelo
        preds = net(images)

        # Preparar el target
        target = (boxes_tensor, labels_tensor)

        # Calcular la pérdida
        loss = ssd_loss(preds, target)
        print("Loss:", loss.item())

if __name__ == "__main__":
    main()
