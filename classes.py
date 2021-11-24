import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor









BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001


class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256), # input, output
            nn.ReLU(), 
            nn.Linear(256, 10) 
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):   # how to manipulte data
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root='data_down', 
        download=True, 
        train=True, 
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root='data_down', 
        download=True, 
        train=False, 
        transform=ToTensor()
    )
    return train_data, validation_data


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, target in data_loader: 
        inputs, target = inputs.to(device), target.to(device)


        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, target)

        # backpropagate loss and updateweights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")
  


def train(model, data_loader, loss_fn, optimiser, device, EPOCHS):

    for i in range(EPOCHS):
        print(f"Epoch {i+1}") 
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("-------------------------")
    print("Training")



  

if __name__ == '__main__':
    # download MNIST dataset 
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # create data loader for the train set 
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)


    # build model
    if torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)

    # instantiate loss function + optimiser 
    loss_fn = nn.CrossEntropyLoss
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr = LEARNING_RATE
    )

    # train model 
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)


torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
print('Model trained and stored at feedforwathdp')