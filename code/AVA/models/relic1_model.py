import torch.nn as nn
import torch
from models.mv2 import cat_net


def SelfAttentionMap(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

class NIMA(nn.Module):
    def __init__(self):
        super(NIMA, self).__init__()
        base_model = cat_net()
        self.base_model = base_model
        # 对于在代码中之前定义的所有参数，这些参数的 requires_grad 属性将被设置为 False，
        # 并且在后续代码中如果尝试修改这些参数的 requires_grad 属性，将不会起作用，因为这些参数已经被标记为不需要计算梯度。
        # 但是，对于在之后定义的参数，这些参数的 requires_grad 属性将默认为 True，并且可以通过 p.requires_grad = True 等语句来显式地指示需要计算梯度。
        for p in self.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(3681, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        x1, x2 = self.base_model(x)
        x = torch.cat([x1,x2],1)
        x = self.head(x)

        return x

#
# if __name__=='__main__':
#     model = NIMA()
#     x = torch.rand((16,3,224,224))
#     out = model(x)