# test PGD attack on a test set of CIFAR10
import advattack
import resnet
import torch
import torchvision as tv
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_accuracy(dataloader, model):
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        _, predict = torch.max(output.data, 1)
        correct += (predict == y).sum().item()
        total += y.size(0)
    return correct / total


batchsize = 500
testset = tv.datasets.CIFAR10("data/", train=False,
                              transform=tv.transforms.ToTensor(), download=True)
testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)
testloader2 = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)
model = resnet.Resnet_20_CIFAR10().to(device)
# Using a test Resnet model. If the file not exist, please run test_resnet_cifar10 to train
# a test model
model.load_state_dict(torch.load("testmodel_resnet_cifar.ckpt", map_location=device))
model.eval()
correct, total = 0, 0
adv_examples_list = []
label_list = []
for i, (images, labels) in enumerate(testloader):
    images = images.to(device)
    labels = labels.to(device)
    adv_examples = advattack.PGD(images, labels, model, 7, 1 / 32, 1 / 128)
    adv_examples_list.append(adv_examples)
    label_list.append(labels)
    output = model(adv_examples)
    _, predict = torch.max(output.data, 1)
    correct += (predict == labels).sum().item()
    total += labels.size(0)
    print("Correct rate for white box attack{}".format(correct / total))
adv_set = torch.cat(adv_examples_list)
test_labels = torch.cat(label_list)
torch.save(adv_set, "Adv_examples.pt")


def test_accuracy_adv(model):
    model.eval()
    with torch.no_grad():
        output = model(adv_set)
        _, predict = torch.max(output.data, 1)
        correct = (predict == test_labels).sum().item()
        total = test_labels.size(0)
    model.train()
    return correct / total


os.chdir("models")
for path in os.listdir():
    model.load_state_dict(torch.load(path, map_location=device))
    print(path+": adversarial accuracy" + "{:.3f}".format(test_accuracy_adv(model)))
