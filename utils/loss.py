import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        # assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = abs(predict.view(num, -1))
        tar = abs(target.view(num, -1))

        intersection = (pre * tar).sum()
        union = pre.sum() + tar.sum()

        score = 1 - (2 * intersection + self.epsilon) / (union + self.epsilon)

        return score