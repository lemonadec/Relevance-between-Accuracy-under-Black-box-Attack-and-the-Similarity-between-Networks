import os
import torch
import torchvision as tv
import resnet


if not os.path.exists("models"):
    os.mkdir("models")
os.chdir("models")
training_model_num = 5
start_index = len(os.listdir()) + 1
batchsize = 128
lr = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 160
crit = torch.nn.CrossEntropyLoss()
# Load CIFAR10 dataset
preprocess = tv.transforms.Compose([
    tv.transforms.Pad(4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomCrop(32),
    tv.transforms.ToTensor()
])
trainingset = tv.datasets.CIFAR10("../data/", train=True,
                                  transform=preprocess, download=True)
testset = tv.datasets.CIFAR10("../data/", train=False,
                              transform=tv.transforms.ToTensor(), download=True)
trainloader = torch.utils.data.DataLoader(trainingset, batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)

for model_index in range(training_model_num):
    crit = torch.nn.CrossEntropyLoss()
    model = resnet.Resnet_18_CIFAR10().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=0.0001)
    model_save_path = "model{}.pt".format(model_index + start_index)
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            loss = crit(out, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch + 1 == 80 or epoch + 1 == 120:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
    torch.save(model.state_dict(), model_save_path)
