#%%
import torch
import torchvision
from torch.utils.data import DataLoader
from utils.agave_dataset import AgaveDataset, get_transform
from tqdm import tqdm

# Definir la función collate_fn fuera del cuerpo principal
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Definir el dispositivo
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Dataset
    dataset = AgaveDataset('data/datasets/agaveHD-1', image_set='train', transform=get_transform(train=True))
    dataset_test = AgaveDataset('data/datasets/agaveHD-1', image_set='valid', transform=get_transform(train=False))

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Modelo
    weights = torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT
    model = torchvision.models.detection.ssd300_vgg16(weights=weights)

    # Ajustar el número de clases
    num_classes = 2  # 'agave' + background
    num_anchors = sum(model.anchor_generator.num_anchors_per_location())

    # Confirmar el valor correcto de in_channels
    in_channels = 364

    # Crear una nueva cabeza de clasificación con el número correcto de clases
    new_classification_head = torch.nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
    model.head.classification_head.module_list[0].conv = new_classification_head
    model.to(device)

    # Optimizador
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Entrenamiento
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for images, targets in data_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                pbar.set_postfix(loss=losses.item())
                pbar.update(1)

        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
