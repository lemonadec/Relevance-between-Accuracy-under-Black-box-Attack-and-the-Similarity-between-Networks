import similarity
import resnet
import numpy as np
import torchvision as tv
import torch
import matplotlib
from matplotlib import pyplot as plt
import advattack as adv
import os
matplotlib.use('Qt5Agg')

def get_func():
    yield similarity.LR
    yield similarity.CKA
    yield similarity.CCA
    yield similarity.CCA_rou
    yield similarity.SVCCA
    yield similarity.SVCCA_rou
    yield similarity.PWCCA
    yield similarity.HSIC

def plot_similarity_vs_acc(std_model, model_list):
    """Plot network similarity vs. accuracy. Test data is 500 images in
    CIFAR10 test set"""
    print("get started")
    # Load test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batchsize = 5000
    testset = tv.datasets.CIFAR10("data/", train=False, transform=tv.transforms.ToTensor(), download=True)
    testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)
    testimage, testlabel = iter(testloader).next()
    testimage, testlabel = testimage.to(device), testlabel.to(device)
    # Generate adv. examples
    # the result is saved in "save_x.npy" as a numpy array
    
    print("generating...")
    adv_images = adv.PGD(testimage, testlabel, std_model, iternum=10, eps=1/32, stepsize=1/128)
    np.save('save_x', adv_images.numpy())
    
    #adv_images = torch.from_numpy(np.load('save_x.npy'))
    print("done")
    # Calculate black box attack accuracy
    # the result is saved in "save_acc.npy" as a numpy array
    
    acc_list = []
    for model in model_list:
        output = model(adv_images)
        _, predict = torch.max(output.data, 1)
        correct = (predict == testlabel).sum().item()
        acc_list.append(correct/batchsize)
    np.save('save_acc', np.array(acc_list))
    
    # Calculate simlarity index after last layer
    #acc_list = np.load('save_acc.npy').tolist()
    def preprocess(feature):
        feature = feature.view(batchsize, -1)
        feature -= torch.mean(feature, 0)
        return feature
    
    print("everything ok here")
    

    #used to test maxmatch
    model_feature = resnet.Resnet_20_CIFAR10_feature().to(device)
    model_feature.load_state_dict(std_model.state_dict())
    output = model_feature(testimage)
    std_feature = []
    for j in range(3):
        std_feature.append(model_feature.get_feature(j+6))
    for j in range(3): # 8 block features
        print("block ",j+6)
        sim_list = []
        for model in model_list:
            model_feature.load_state_dict(model.state_dict())
            output = model_feature(testimage)
            feature = model_feature.get_feature(j+6)  #get_feature
            sim_list.append(similarity.maxmatch(std_feature[j], feature,0.8))#try 0.1,0.2,...,0.5
        # Ploting
        accuracy, similar = np.array(acc_list), np.array(sim_list)
        plt.cla()
        plt.scatter(accuracy, similar)
        plt.savefig("maxmatch "+"epsilon=08 "+str(j+6))
    
    #used to test the first 8 indices
    std_feature = []
    model_feature = resnet.Resnet_20_CIFAR10_feature().to(device)
    model_feature.load_state_dict(std_model.state_dict())
    output = model_feature(testimage)
    for j in range(3): # 8 block features
        std_feature.append(preprocess(model_feature.get_feature(j+6)))  # Get the feature after last block
    print("std_model done")
    for j in range(3): # 8 block features
        for i, index_func in enumerate(get_func()): # 8 indexfunc
            print("block ",j+6, " index func ", i)
            sim_list = []
            for model in model_list:
                model_feature.load_state_dict(model.state_dict())
                output = model_feature(testimage)
                feature = preprocess(model_feature.get_feature(j+6))
                sim_list.append(index_func(std_feature[j], feature).item())
            # Ploting
            accuracy, similar = np.array(acc_list), np.array(sim_list)
            plt.cla()
            plt.scatter(accuracy, similar)
            plt.savefig(str(j+6)+' '+str(i))
 

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
    plot_similarity_vs_acc(std_model, model_list)
