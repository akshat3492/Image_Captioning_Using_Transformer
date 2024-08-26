import torch
import time

def save_rotation_model(net):
    """
    Saves the state dictionary of the given neural network model to a specified file path.

    Arguments:
        net (torch.nn.Module): The neural network model to be saved.

    Returns:
        None
    """
    path = "save_model/self_supervised_rotation_model.pth"
    torch.save(net.state_dict(), path)


class Rotation_Predictor():
    """
    A class to train and evaluate a rotation prediction model using self-supervised learning.

    This class encapsulates the training and validation processes for a neural network that
    predicts the rotation angle applied to an image. It includes methods for adjusting the 
    learning rate, training the model for one epoch, and running evaluation on a validation set.

    Attributes:
        net (torch.nn.Module): The neural network model to be trained.
        criterion (torch.nn.Module): The loss function to be used during training.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
        num_epochs (int): Number of epochs to train the model.
        decay_epochs (int): Number of epochs after which the learning rate is decayed.
        init_lr (float): Initial learning rate for training.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training set.
        valloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): The device (CPU or GPU) on which the model is trained.
    """
    
    def __init__(self, net, criterion, optimizer, num_epochs, decay_epochs, init_lr, trainloader, valloader):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.decay_epochs = decay_epochs
        self.init_lr = init_lr
        self.trainloader = trainloader
        self.valloader = valloader

        self.device = next(self.net.parameters()).device

    def adjust_learning_rate(self):
        """
        Adjusts the learning rate according to a predefined decay schedule.
        
        This method reduces the learning rate by a factor of 0.1 every `decay_epochs` epochs.
        """
        lr = self.init_lr * (0.1 ** (self.num_epochs // self.decay_epochs))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr # Update learning rate for each parameter group in the optimizer

    def train_epoch(self):
        """
        Trains the model for a specified number of epochs and evaluates it on the validation set.
        
        This method includes logic for adjusting the learning rate, saving the best model, and
        performing early stopping if the model achieves an accuracy above a certain threshold.

        Returns:
            None
        """
        
        best_accuracy = 0.0

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_correct = 0.0
            running_total = 0.0
            start_time = time.time()

            self.net.train()

            for i, sample in enumerate(self.trainloader, 0):
                

                img_rotated = sample["Rotated Image"].to(self.device)
                rot_label = sample["Rotation Label"].to(self.device)
                
                self.optimizer.zero_grad()

                outputs = self.net(img_rotated)
                loss = self.criterion(outputs, rot_label)
                loss.backward()
                self.optimizer.step()

                # Get predicted results
                predicted = torch.argmax(outputs, dim = 1)
    
                # print statistics
                print_freq = 100
                running_loss += loss.item()
    
                # calc acc
                running_total += rot_label.size(0)
                running_correct += (predicted == rot_label).sum().item()
    
                if i % print_freq == (print_freq - 1):  
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_freq:.3f} acc: {100*running_correct / running_total:.2f} time: {time.time() - start_time:.2f}')
                    running_loss, running_correct, running_total = 0.0, 0.0, 0.0
                    start_time = time.time()

            self.adjust_learning_rate()
            
            self.net.eval()
            accuracy = self.run_test()
            

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_rotation_model(self.net)

            if accuracy > 70:
                print("Early Stopping!")
                break

        print("Training Finished!")

    def run_test(self):
        """
        Evaluates the model on the validation set and computes the accuracy and average loss.
        
        This method performs inference on the validation set without tracking gradients,
        calculates the accuracy of the model's predictions, and computes the average loss.

        Returns:
            float: The accuracy of the model on the validation set.
        """
        
        correct = 0
        total = 0
        avg_test_loss = 0.0

        with torch.no_grad(): # Disable gradient calculation for efficiency during inference
            for i, sample in enumerate(self.valloader, 0):
                images = sample["Rotated Image"].to(self.device)
                labels = sample["Rotation Label"].to(self.device)
                
                outputs = self.net(images)
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
                # loss
                avg_test_loss += self.criterion(outputs, labels)  / len(self.valloader)
        print('TESTING:')
        print(f'Accuracy of the network on the test images: {100 * correct / total:.2f} %')
        print(f'Average loss on the test images: {avg_test_loss:.3f}')
    
        return 100 * (correct / total)

