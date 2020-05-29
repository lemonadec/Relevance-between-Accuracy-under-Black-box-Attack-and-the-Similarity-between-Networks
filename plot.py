import similarity
import resnet
import numpy as np
import torchvision as tv
import torch
from matplotlib import pyplot as plt
import advattack as adv
import os
plt.switch_backend('agg')


def plot_similarity_vs_acc(index_func, std_model, model_list, save_path=None):
    """Plot network similarity vs. accuracy. Test data is 500 images in
    CIFAR10 test set"""
    # Load test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batchsize = 1000
    testset = tv.datasets.CIFAR10("data/", train=False, transform=tv.transforms.ToTensor(), download=True)
    testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)
    testimage, testlabel = iter(testloader).next()
    testimage, testlabel = testimage.to(device), testlabel.to(device)
    # Generate adv. examples
    adv_images = adv.PGD(testimage, testlabel, std_model, iternum=10, eps=1/32, stepsize=1/128)
    # Calculate black box attack accuracy
    acc_list = []
    for model in model_list:
        output = model(adv_images)
        _, predict = torch.max(output.data, 1)
        correct = (predict == testlabel).sum().item()
        acc_list.append(correct/batchsize)
    # Calculate simlarity index after last layer

    def preprocess(feature):
        feature = feature.view(batchsize, -1)
        feature -= torch.mean(feature, 0)
        return feature

    sim_list = []
    model_feature = resnet.Resnet_20_CIFAR10_feature().to(device)
    model_feature.load_state_dict(std_model.state_dict())
    output = model_feature(testimage)
    std_feature = preprocess(model_feature.get_feature(8))  # Get the feature after last block
    for model in model_list:
        model_feature.load_state_dict(model.state_dict())
        output = model_feature(testimage)
        feature = preprocess(model_feature.get_feature(8))
        sim_list.append(index_func(std_feature, feature).item())
    # Ploting
    accuracy, similarity = np.array(acc_list), np.array(sim_list)
    plt.scatter(accuracy, similarity)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    std_model = resnet.Resnet_20_CIFAR10().to(device)
    model_list = []
    os.chdir("models")
    for index, path in enumerate(os.listdir()):
        if index:
            model = resnet.Resnet_20_CIFAR10().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model_list.append(model)
        else:
            std_model.load_state_dict(torch.load(path, map_location=device))
    os.chdir("..")
    plot_similarity_vs_acc(similarity.LR, std_model, model_list, save_path="fig1.PNG")
