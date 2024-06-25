import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# Transformer中的Norm层
class Norm(nn.Module):
    """
        input_dim: 输入token的维度
    """
    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.norm(x)
        return x


# Transformer中的FFN(FeedForward Network)层
class MLP(nn.Module):
    """
        input_dim:输入token的维度
        hidden_dim:中间隐层的维度
        dropout:防止过拟合的概率参数
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.ffn(x)
        return x


# Transformer中的Attention层
class Attention(nn.Module):
    """
        input_dim:输入token的维度
        head_dim:每个head的维度，head_dim*head_cnt为Q,K,V的维度
        head_cnt:head的数量
        dropout:防止过拟合的概率参数
    """
    def __init__(self, input_dim, head_dim, head_cnt, dropout):
        super().__init__()
        inner_dim = head_cnt * head_dim
        project_out = not (head_cnt == 1 and head_dim == input_dim)

        self.head_cnt = head_cnt    # head数量
        self.scale = head_dim ** -0.5    # 尺度缩放因子
        self.softmax = nn.Softmax(dim=-1)    # softmax
        self.w_qkv = nn.Linear(input_dim, inner_dim*3, bias=False)    # 计算Q,K,V的权重W

        self.output = nn.Sequential(
            nn.Linear(inner_dim, input_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        batch, token_num, _ = x.shape
        qkv = self.w_qkv(x).chunk(3, dim=-1)    # 拿到Q,K,V矩阵  b,n,3*inner_dim --> 3*(b,n,inner_dim)  元组
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.head_cnt), qkv)    # 拿到每个head的Q,K,V
        # Attention操作
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale    # (q*k^T) / (inner_dim**-0.5)
        att = self.softmax(dots)
        att = einsum('b h i j, b h j d -> b h i d', att, v)
        x = rearrange(att, 'b h n d -> b n (h d)')
        x = self.output(x)
        return x


# 空时Transformer块 -- spatial-temporal information
class SpatialTemporalTransformerBlock(nn.Module):
    def __init__(self, input_dim, head_dim, head_cnt, mlp_dim, dropout):
        super().__init__()
        self.norm1 = Norm(input_dim=input_dim)
        self.attention = Attention(input_dim=input_dim, head_dim=head_dim, head_cnt=head_cnt, dropout=dropout)
        self.norm2 = Norm(input_dim=input_dim)
        self.ffn = MLP(input_dim=input_dim, hidden_dim=mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attention(self.norm1(x)) + x   # 残差结构
        x = self.ffn(self.norm2(x)) + x
        return x


# 空间Transformer块 -- spatial information
class SpatialTransformerBlock(nn.Module):
    def __init__(self, input_dim, head_dim, head_cnt, mlp_dim, dropout):
        super().__init__()
        self.norm1 = Norm(input_dim=input_dim)
        self.attention = Attention(input_dim=input_dim, head_dim=head_dim, head_cnt=head_cnt, dropout=dropout)
        self.norm2 = Norm(input_dim=input_dim)
        self.ffn = MLP(input_dim=input_dim, hidden_dim=mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attention(self.norm1(x)) + x  # 残差结构
        x = self.ffn(self.norm2(x)) + x
        return x


# 时间Transformer块 -- temporal information
class TemporalTransformerBlock(nn.Module):
    def __init__(self, input_dim, head_dim, head_cnt, mlp_dim, dropout):
        super().__init__()
        self.norm1 = Norm(input_dim=input_dim)
        self.attention = Attention(input_dim=input_dim, head_dim=head_dim, head_cnt=head_cnt, dropout=dropout)
        self.norm2 = Norm(input_dim=input_dim)
        self.ffn = MLP(input_dim=input_dim, hidden_dim=mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.attention(self.norm1(x)) + x  # 残差结构
        x = self.ffn(self.norm2(x)) + x
        return x


# 预处理 -- 分割spatial和temporal的token
class TokenProcess(nn.Module):
    def __init__(self, time, height, width, channel):
        super().__init__()

    def forward(self, x):
        batch, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> b t h w c')
        st_token = rearrange(x, 'b t h w c -> b (c t) (h w)')
        spatial_token = rearrange(x, 'b t h w c -> b c (t h w)')    # spatial 信息
        temporal_token = rearrange(x, 'b t h w c -> b t (h w c)')    # temporal 信息

        return st_token, spatial_token, temporal_token


# 特征维度处理
class DimensionReduction(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        batch, c, t, h, w = x.size()
        x = self.conv3d(x)
        # print(x.size())
        return x


# 模型整体架构
class Model(nn.Module):
    def __init__(self, time, height, width, channel,
                 out_time, out_channel, kernel_size, stride, padding,
                 head_spatial_dim, head_spatial_cnt, scale_spatial_dim,
                 head_temporal_dim, head_temporal_cnt, scale_temporal_dim,
                 st_dim, head_st_dim, head_st_cnt, scale_st_dim,
                 num_class, dropout):
        super().__init__()
        self.dimension_reduction = DimensionReduction(in_channel=channel, out_channel=out_channel,kernel_size=kernel_size, stride=stride, padding=padding)
        self.token_process = TokenProcess(out_time, height, width, out_channel)
        self.spatial_transformer = SpatialTransformerBlock(input_dim=(out_time*height*width), head_dim=head_spatial_dim, head_cnt=head_spatial_cnt,
                                                           mlp_dim=(out_time*height*width)*scale_spatial_dim, dropout=dropout)
        self.temporal_transformer = TemporalTransformerBlock(input_dim=(out_channel*height*width), head_dim=head_temporal_dim, head_cnt=head_temporal_cnt,
                                                             mlp_dim=(out_channel*height*width)*scale_temporal_dim, dropout=dropout)
        # fusion
        self.w_spatial = nn.Linear((out_time*height*width), st_dim)
        self.w_temporal = nn.Linear((out_channel*height*width), st_dim)
        # self.st_norm = nn.LayerNorm(st_dim)
        # spatial-temporal Transformer
        # self.cls_spatial_temporal = nn.Parameter(torch.randn(1, 1, st_dim))  # class token
        # self.spatial_temporal_transformer = SpatialTemporalTransformerBlock(input_dim=st_dim, head_dim=head_st_dim, head_cnt=head_st_cnt,
        #                                                                     mlp_dim=st_dim*scale_st_dim, dropout=dropout)
        # classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(st_dim),
            nn.Linear(st_dim, 256),
            nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        # 对通道进行降维
        x = self.dimension_reduction(x)
        # 预处理操作 -- 添加cls_token和pos_embedding
        _, spatial_token, temporal_token = self.token_process(x)
        # spatial Transformer
        spatial_token = self.spatial_transformer(spatial_token)
        # temporal Transformer
        temporal_token = self.temporal_transformer(temporal_token)
        # 融合操作
        spatial_token = self.w_spatial(spatial_token)
        temporal_token = self.w_temporal(temporal_token)
        spatial_temporal_token = torch.cat((spatial_token, temporal_token), dim=1)
        spatial_temporal_token = torch.mean(spatial_temporal_token, dim=1)
        # spatial_temporal_token = self.st_norm(spatial_temporal_token)
        # spatial-temporal Transformer
        # batch, nst, dst = spatial_temporal_token.size()
        # cls_spatial_temporal_tokens = repeat(self.cls_spatial_temporal, '() n d -> b n d', b=batch)    # 添加cls_token
        # spatial_temporal_token = torch.cat((cls_spatial_temporal_tokens, spatial_temporal_token), dim=1)
        # spatial_temporal_token = self.spatial_temporal_transformer(spatial_temporal_token)
        # 取出cls token
        # cls_token = spatial_temporal_token[:,0]
        output = self.classifier(spatial_temporal_token)
        return output



if __name__ == "__main__":
    x = torch.rand(1,256,32,8,8)
    # T C H W : 32 256 8 8
    # 1. 32 16 8 8  [out_time=32,out_channel=16, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),]
    # 2. 32 8 8 8  [out_time=32,out_channel=8, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),]
    # 3. 16 16 8 8   [out_time=16,out_channel=16, kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0),]
    # 4. 16 8 8 8  [out_time=16,out_channel=8, kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0),]
    model = Model(time=32, height=8, width=8, channel=256,
                 out_time=32, out_channel=16, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0),
                 head_spatial_dim=256, head_spatial_cnt=4, scale_spatial_dim=2,
                 head_temporal_dim=64, head_temporal_cnt=4, scale_temporal_dim=2,
                 st_dim=256, head_st_dim=64, head_st_cnt=4, scale_st_dim=2,
                 num_class=4, dropout=0)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    preds = model(x)
    print(preds.shape)









