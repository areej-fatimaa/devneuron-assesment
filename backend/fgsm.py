import torch
import torch.nn.functional as F

class FGSM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def attack(self, image, label, epsilon=0.1):
        image.requires_grad = True

        output = self.model(image)
        loss = F.nll_loss(output, label)
        self.model.zero_grad()
        loss.backward()

        data_grad = image.grad.data
        perturbed_image = image + epsilon * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image
