import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from six.moves import urllib
import logging
from tech_renaissance import import_tsr, export_tsr


BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_WORKERS = 0
LEARNING_RATE = 0.1
INTERVAL = 100
LOG_FILE = 'log.txt'

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(message)s')
best_accuracy = 0.0


def print_and_record(*args):
    print(*args)
    text = ' '.join(map(str, args))
    logging.info(text)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.name = 'MLP'
        self.classifier = None
        self.make_layers()

    def forward(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        x = self.classifier(x)
        return x

    def make_layers(self):
        classifier_layers = list()
        classifier_layers += [nn.Linear(784, 512, bias=False)]
        classifier_layers += [nn.Tanh()]
        classifier_layers += [nn.Linear(512, 256, bias=False)]
        classifier_layers += [nn.Tanh()]
        classifier_layers += [nn.Linear(256, 10, bias=False)]
        self.classifier = nn.Sequential(*classifier_layers)


def train_one_epoch(model, criterion, optimizer, train_loader, device, epoch):
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(torch.float32), target.to(torch.int32)
            export_tsr(data, 'train_images.tsr')
            export_tsr(target, 'train_labels.tsr')
            break


def test(model, criterion, test_loader, device):
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(torch.float32), target.to(torch.int32)
            export_tsr(data, 'test_images.tsr')
            export_tsr(target, 'test_labels.tsr')
            break


def main():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../../..', train=True, download=True, transform=train_transform),
        batch_size=60000, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../../..', train=False, transform=test_transform),
        batch_size=10000, shuffle=False, num_workers=0)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    os.makedirs('models', exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print_and_record(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        test(model, criterion, test_loader, device)
        break

    print_and_record(f'\nTraining completed. Best Accuracy: {best_accuracy:.2f}%')


if __name__ == '__main__':
    main()
