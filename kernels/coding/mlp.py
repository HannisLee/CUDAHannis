import torch


class selfAttention(torch.Module):
    def __init__(self):
        super().__init__()
        self.Q_proj = nn.Linear(4096,4096)
        self.W_proj = nn.Linear(4096,4096)
        self.V_proj = nn.Linear(4096,4096)
        self.O_proj

    def forward(x):

        q = self.Q_proj
        k = self.Q_proj
        v = self.Q_proj


        x_1 = Q.mutmul(V.transpose(-2,-1))/x_1.size(-2)
        x_2 =F.softmax(x_1)

        x_3 = x_2.mutmul(V.transpose(-2,-1))

        O = self.o_proj(x_3)


M,K K N   M N

naive

