import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from six.moves import urllib
import logging


BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_WORKERS = 0
LEARNING_RATE = 0.001
INTERVAL = 100
LOG_FILE = 'log_adamw.txt'

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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f'Train Epoch: {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')


def test(model, criterion, test_loader, device):
    global best_accuracy
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset), test_acc))

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        # torch.save(model.state_dict(), f'models/best_model.pth')
        # print(f'New best model saved with accuracy: {test_acc:.2f}%')


def main():
    # 记录开始时间
    start_time = time.time()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../../..', train=True, download=True, transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../../..', train=False, transform=test_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    os.makedirs('models', exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        test(model, criterion, test_loader, device)

    # print(f'\nTraining completed. Best Accuracy: {best_accuracy:.2f}%')
    # print(f'Total training time: {int(time.time() - start_time)} seconds')
    print_and_record(f'\n{best_accuracy:.2f}%')
    print_and_record(f'{int(time.time() - start_time)}')


if __name__ == '__main__':
    main()
