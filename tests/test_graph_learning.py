import torch.nn.functional as F
import torch.nn as nn
import torch

"""
    这个是把PICK对应论文中的GraphModel中的Graph Learning的代码，
    看起来忒费劲，我就笨方法，一行行肢解后，打印出来看、理解：
    我理解就是
    - 降维
    - 做softmax算每个节点和彼此之间的关系权重
    - 然后计算loss：
            '''
                \mathcal{L}_{GL}=\frac{1}{N}\sum_{i,j=1}^N exp(A_{ij}+\eta \Vert v_i - v_j \Vert^2_2) + \gamma \Vert A \Vert_F^2 
            '''
    
"""

N = 8  # 有多少个bbox，一张图里
in_dim = 10  # 10：每个bbox的隐层表达的维度
learning_dim = 5  # 应该是降维后的维度
eta = 1
gamma = 1
box_num = torch.tensor([1,8])


x = torch.randn(1, N, in_dim)
print("我是bbox的feature： x = ", x.shape)

learn_w = nn.Parameter(torch.empty(learning_dim))
projection = nn.Linear(in_dim, learning_dim, bias=False)
adj = torch.ones((1, N, N))
print("我是整个图的邻接矩阵：A = ", adj)
print("我是整个图的邻接矩阵：A.shape ", adj.shape)

B, N, D = x.shape  # todo 难道是N，T，D？

# (B, N, D)
x_hat = projection(x)  # todo 这个是做了一个降维么？3维度=>1维，还是最后1维降维了？
print("我被线性变换了一下: x -> hat_x %r=>%r" % (x.shape, x_hat.shape))

_, _, learning_dim = x_hat.shape  # 紧接着看这个，应该是最后一维降维了 ： self.projection = nn.Linear(in_dim, learning_dim, bias=False)
# (B, N, N, learning_dim)
x_i = x_hat.unsqueeze(2).expand(B, N, N, learning_dim)# todo <--- ???
x_j = x_hat.unsqueeze(1).expand(B, N, N, learning_dim)
print("骚操作之后的x_i:", x_i.shape)
print("骚操作之后的x_j:", x_j.shape)

distance = torch.abs(x_i - x_j)  # <--- 感觉是两两做差
print("| x_i - x_j |: ", distance.shape)

distance = torch.einsum('bijd, d->bij', distance, learn_w) # todo <--- ???
print("| x_i - x_j | 啥爱因斯坦sum后 : ", distance.shape)

out = F.leaky_relu(distance)
print(" Relu(| x_i - x_j |) : ", distance.shape)

# for numerical stability, due to softmax operation mable produce large value
max_out_v, _ = out.max(dim=-1, keepdim=True) # todo <--- ???
print(" Max(Relu(| x_i - x_j |)) : ", max_out_v.shape)

out = out - max_out_v

soft_adj = torch.exp(out)
print(" Exp(Max(Relu(| x_i - x_j |))) : soft邻接矩阵 ", soft_adj.shape)

soft_adj = adj * soft_adj  # <-- ??? 为何要用新的邻接矩阵，和就的邻接矩阵相乘
print("两个邻接矩阵相乘：adj * soft_adj：adj.shape=%r, soft_adj.shape=%r" % (adj.shape, soft_adj.shape))

sum_out = soft_adj.sum(dim=-1, keepdim=True)
soft_adj = soft_adj / sum_out + 1e-10

print("终于算出了新的邻接矩阵了：")
print(soft_adj)
print("新的邻接矩阵的shape：", soft_adj.shape)

print("=======================================================")
print("开始计算loss了！")

B, N, D = x_hat.shape
x_i = x_hat.unsqueeze(2).expand(B, N, N, D)
x_j = x_hat.unsqueeze(1).expand(B, N, N, D)
box_num_div = 1 / torch.pow(box_num.float(), 2)
# (B, N, N)
dist_loss = adj + eta * torch.norm(x_i - x_j, dim=3)  # remove square operation duo to it can cause nan loss.
dist_loss = torch.exp(dist_loss)
# (B,)
dist_loss = torch.sum(dist_loss, dim=(1, 2)) * box_num_div.squeeze(-1)
# (B,)
f_norm = torch.norm(adj, dim=(1, 2))  # remove square operation duo to it can cause nan loss.
gl_loss = dist_loss + gamma * f_norm

print("loss结果为：", gl_loss)
