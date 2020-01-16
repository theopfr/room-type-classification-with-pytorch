
from utils import transform, freeze_layers, show_progress, invert_norm
from model import Model
from roomDataset import RoomDataset

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import lr_scheduler


class RunModel:
    def __init__(self, Model, train_set="", test_set="", validation_set="", batch_size=64, epochs=100, lr=1e2, dropout_chance=0.5, lr_decay=0.1):
        self.Model = Model

        self.train_set = train_set
        self.test_set = test_set
        self.validation_set = validation_set

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dropout_chance = dropout_chance
        self.lr_decay = lr_decay

    """ creates dataloader """
    def _create_dataloader(self, dataset_path: str):
        dataset = RoomDataset(dataset_path, transform())
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True
        )
        return dataloader

    """ calculates accuracy """
    def _validate(self, model):
        validation_dataset = self._create_dataloader(self.validation_set)
        model = model.eval()

        accuracy, total, correct, total_loss = 0, 0, 0, []
        for images, targets in validation_dataset:

            images = images.float().cuda()
            targets = targets.float().cuda()

            predictions = model(images)

            correct_in_batch = 0
            for target in predictions.round().eq(targets):
                
                c = 0
                for element in target:
                    if element == True:
                        c += 1
                if c == 4:
                    correct_in_batch += 1

            total += targets.size(0)
            correct += correct_in_batch

            criterion = nn.MSELoss()
            batch_loss = criterion(predictions, targets)
            total_loss.append(batch_loss.item())

        accuracy = 100 * correct/total
        loss = np.mean(total_loss)

        return accuracy, loss

    """ trains model """
    def train(self):            
        training_dataset = self._create_dataloader(self.train_set)
        model = self.Model(dropout_chance=self.dropout_chance).cuda()
        model = freeze_layers(model)

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

        loss_data, validation_accuracy_data, validation_loss_data = [], [], []

        for epoch in tqdm(range(self.epochs), ncols=90, desc="progress"):
            epoch_loss = []

            for images, targets in training_dataset:
                optimizer.zero_grad()

                images = images.float().cuda()
                targets = targets.float().cuda()

                predictions = model.train()(images)

                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

            scheduler.step()
            # print("learning rate:", optimizer.param_groups[0]["lr"])

            current_loss = np.mean(epoch_loss)
            current_val_accuracy, current_val_loss = self._validate(model)

            show_progress(self.epochs, epoch, current_loss, current_val_accuracy, current_val_loss)

            loss_data.append(current_loss)
            validation_accuracy_data.append(current_val_accuracy)
            validation_loss_data.append(current_val_loss)

            if epoch % 5:
                torch.save(model.state_dict(), "models/model_1.pt")

        print("\n finished training")

    """ tests dataset """
    def test(self):
        testing_dataset = self._create_dataloader(self.test_set)

        model = self.Model(dropout_chance=0.0)
        model.load_state_dict(torch.load("models/model_1.pt"))
        model.cuda().eval()
        
        # calculate accuracy of the trained model
        acc, _ = self._validate(model)
        print("testing accuracy: ", round(acc, 4), "%")

        # get examples
        test_elements = []
        for images, targets in testing_dataset:
            images = images.float().cuda()
            targets = targets.float().cuda()
            test_elements.append([images, targets])

        # invert normalization for plotting
        denorm = invert_norm()
        
        # show examples
        for element in test_elements:
            outputs = model(element[0])
            print("\nexpected: ", element[1].cpu().detach().numpy()[0], "\ngot:      ", np.around(outputs.cpu().detach().numpy()[0]))
            print("\n____________________\n")
            plt.matshow(denorm(element[0][0]).cpu().detach().numpy().reshape(350, 475, 3))
            plt.title("got: " + str(outputs.cpu().detach().numpy()[0]))
            plt.show()


if __name__ == "__main__":
    runModel = RunModel(
        Model, 
        train_set="dataset/data/train_set.json",
        test_set="dataset/data/test_set.json",
        validation_set="dataset/data/val_set.json", 
        batch_size=16,
        epochs=50,
        lr=0.01,
        dropout_chance=0.45,
        lr_decay=0.1)

    # runModel.train()
    runModel.test()
        
        