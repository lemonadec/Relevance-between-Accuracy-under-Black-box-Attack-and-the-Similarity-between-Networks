# testing resnet_20 on CIFAR10 dataset
import torch
import torchvision as tv
import resnet


batchsize = 128
lr = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet.Resnet_18_CIFAR10().to(device)
num_epochs = 160
# Load CIFAR10 dataset
preprocess = tv.transforms.Compose([
    tv.transforms.Pad(4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomCrop(32),
    tv.transforms.ToTensor()
])
trainingset = tv.datasets.CIFAR10("data/", train=True,
                                  transform=preprocess, download=True)
testset = tv.datasets.CIFAR10("data/", train=False,
                              transform=tv.transforms.ToTensor(), download=True)
trainloader = torch.utils.data.DataLoader(trainingset, batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)
# Training of model


def test_accuracy(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            _, predict = torch.max(output.data, 1)
            correct += (predict == labels).sum().item()
            total += labels.size[0]
    model.train()
    return correct / total


crit = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0001)
for epoch in range(num_epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(trainloader):
        images.to(device)
        labels.to(device)
        out = model(images)
        loss = crit(out, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch + 1 == 80 or epoch + 1 == 120:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    print("Epoch {}: training loss:{:.4f} test accuracy:{:.3f}".format(epoch + 1,
                                                                       total_loss.item()))
