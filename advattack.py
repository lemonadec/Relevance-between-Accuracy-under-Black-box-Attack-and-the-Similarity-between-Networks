# implement different adversarial attack method
import torch
from torch.autograd import Variable


def PGD(clean_images, labels, model, iternum, eps, stepsize):
    model.eval()
    adv_images = clean_images
    crit = torch.nn.CrossEntropyLoss()
    for i in range(iternum):
        clean_images = Variable(clean_images, requires_grad=True)
        out = model(clean_images)
        loss = crit(out, labels)
        loss.backward()
        step = stepsize * torch.sign(clean_images.grad)
        adv_images += step
        adv_images = torch.max(adv_images, clean_images + eps)
        adv_images = torch.min(adv_images, clean_images - eps)
        adv_images = torch.min(adv_images, 255 * torch.ones(adv_images.size()))
        adv_images = torch.max(adv_images, torch.zeros(adv_images.size()))

    return adv_images
