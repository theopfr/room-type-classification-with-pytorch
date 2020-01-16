
from roomDataset import RoomDataset
from termcolor import colored
import torch
from torchvision import transforms


# converts image matrix to tensor and normalize
def transform():
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transform

# creates dataloader
def create_dataloader(dataset_path: str, batch_size=64):
    dataset = RoomDataset(dataset_path, transform())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return dataloader

# prints out training progress
def show_progress(epochs, epoch, loss, val_accuracy, val_loss):
    epochs = colored(epoch, "cyan", attrs=['bold']) + colored("/", "cyan", attrs=['bold']) + colored(epochs, "cyan", attrs=['bold'])
    loss = colored(round(loss, 6), "cyan", attrs=['bold'])
    accuracy = colored(round(val_accuracy, 4), "cyan", attrs=['bold']) + colored("%", "cyan", attrs=['bold'])
    val_loss = colored(round(val_loss, 6), "cyan", attrs=['bold'])

    print(" ")
    print("\nepoch {} - loss: {} - val_acc: {} - val_loss: {}".format(epochs, loss, accuracy, val_loss))
    print("\n..........................................................................................\n")

# freezes
def freeze_layers(model):
    for child in model.children():
        for i, param in enumerate(child.parameters()):
            if i < 9:
                param.requires_grad = False
    
    return model









