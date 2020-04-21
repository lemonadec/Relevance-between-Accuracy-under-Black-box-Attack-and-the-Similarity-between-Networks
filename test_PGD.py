# test PGD attack on a test set of CIFAR10
import advattack
import resnet
import torch
import torchvision as tv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 50
testset = tv.datasets.CIFAR10("data/", train=False,
                              transform=tv.transforms.ToTensor(), download=True)
testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)
model = resnet.Resnet_18_CIFAR10().to(device)
# Using a test Resnet model. If the file not exist, please run test_resnet_cifar10 to train
# a test model
model.load_state_dict(torch.load("testmodel_resnet_cifar.ckpt", map_location=device))
correct, total = 0, 0
for i, (images, labels) in enumerate(testloader):
    images = images.to(device)
    labels = labels.to(device)
    adv_examples = advattack.PGD(images, labels, model, 10, 0.03, 0.001)
    output = model(adv_examples)
    _, predict = torch.max(output.data, 1)
    correct += (predict == labels).sum().item()
    total += labels.size(0)
    print(correct / total)
