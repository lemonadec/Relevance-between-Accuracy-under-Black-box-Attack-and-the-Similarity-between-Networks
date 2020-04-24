# implement different adversarial attack method
import torch
from torch.autograd import Variable


def PGD(clean_images, labels, model, iternum, eps, stepsize):
    model.eval()
    crit = torch.nn.CrossEntropyLoss()
    adv_images = torch.normal(clean_images, std=0.1 * eps)
    adv_images = Variable(adv_images, requires_grad=True)
    lowerbound = torch.zeros(clean_images.size()).type_as(clean_images)
    upperbound = torch.ones(clean_images.size()).type_as(clean_images)
    for i in range(iternum):
        adv_images.requires_grad = True
        out = model(adv_images)
        loss = crit(out, labels)
        loss.backward()
        step = stepsize * torch.sign(adv_images.grad)
        adv_images.detach()
        adv_images.requires_grad = False
        adv_images += step
        adv_images = torch.max(adv_images, torch.max(clean_images - eps, lowerbound)[0])
        adv_images = torch.min(adv_images, torch.min(clean_images + eps, upperbound)[0])
    return adv_images.data
