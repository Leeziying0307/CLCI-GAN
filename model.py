import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from sync_batchnorm import SynchronizedBatchNorm2d
from GlobalAttention import VisionTransformer, Transformer, LayerNorm, Block
from typing import Tuple, Union
import clip
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
BatchNorm = SynchronizedBatchNorm2d

device = "cuda" if torch.cuda.is_available() else "cpu"

class C(nn.Module):
    def __init__(self, input_dim=512, num_clusters=10, output_dim=100):
        self.num_clusters = num_clusters
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        # self.prototype = nn.Linear(input_dim, output_dim, bias=False)

    def train(self, documents):
        # 文本表示
        X = self.vectorizer.fit_transform(documents)
        # 训练聚类模型
        self.kmeans.fit(X)

    def forward(self, input_data, is_inference=False):
        if is_inference:
            # 推断时输入为文本特征
            cluster_centers = input_data
        else:
            # 训练时输入为文本
            # 文本表示
            X = self.vectorizer.transform(input_data)
            # 训练聚类模型
            self.kmeans.fit(X)
            # 获取对应聚类中心
            cluster_centers = self.kmeans.cluster_centers_

        # 将每个聚类中心进行线性变换
        # outputs = self.prototype(cluster_centers)
        return cluster_centers

    def get_cluster_centers(self):
        # 获取聚类中心
        return self.kmeans.cluster_centers_


def sinkhorn_knopp_normalization(matrix, epsilon=1e-10, max_iter=1000):
    # 将 matrix 移动到 CPU 上
    matrix_cpu = matrix.cpu()

    # 计算 row_sum 和 col_sum
    row_sum = torch.sum(matrix_cpu, dim=1, keepdim=True).numpy()
    col_sum = torch.sum(matrix_cpu, dim=0, keepdim=True).numpy()

    # 初始化两个正数矩阵
    u = np.ones_like(row_sum)
    v = np.ones_like(col_sum)

    # 迭代更新
    for _ in range(max_iter):
        u = row_sum / (np.dot(matrix_cpu.numpy(), v.reshape(8, 1, 196, 768)) + epsilon)

        v = col_sum / (np.dot(matrix_cpu.numpy().transpose(0, 2, 1), u.reshape(8, 1, 196, 1)).reshape(8, 1, 1,
                                                                                                      768) + epsilon)

    # 最终的归一化结果
    normalized_matrix = np.dot(np.dot(np.diag(u.flatten()), matrix_cpu.numpy()), np.diag(v.flatten()))

    return normalized_matrix

class NetG(nn.Module):
    def __init__(self):
        super().__init__()

        embed_dim = 512
        image_resolution = 224
        vision_layers = 8
        vision_width = 768
        vision_patch_size = 16
        context_length = 77
        transformer_width = 512
        transformer_heads = 8
        transformer_layers = 12

        self.context_length = context_length
        self.linear = nn.Linear(1408, 588, bias=True)
        self.c1 = nn.Conv2d(3, 16, 5, 1, 2)
        self.c2 = nn.Conv2d(16, 16, 5, 1, 2)
        self.c3 = nn.Conv2d(8, 16, 5, 1, 2)
        self.c4 = nn.Conv2d(1, 16, 5, 1, 2)
        self.c5 = nn.Conv2d(1, 8, 1)
        vision_heads = vision_width // 96
        self.visual2 = VisionTransformer(image_size=28, patch_size=16, num_classes=1000, embed_dim=768, num_heads=8,
                                         num_layers=12)
        self.visual3 = VisionTransformer(image_size=56, patch_size=16, num_classes=1000, embed_dim=768, num_heads=12,
                                         num_layers=12)
        self.visual4 = VisionTransformer(image_size=112, patch_size=16, num_classes=1000, embed_dim=768, num_heads=12,
                                         num_layers=12)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.model, self.preprocess = clip.load('ViT-B/32', device=device)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 512))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.decoder_pred = nn.Linear(512, 16**2 * 3, bias=True)
        self.channel_transform = nn.Conv2d(in_channels=8, out_channels=512, kernel_size=1)
        self.conv_img = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 3, 3, 1, 1),
            nn.Tanh(),
        )
        self.L = nn.Linear(11, 224, bias=True)

        self.sinkhorn_knopp_normalization = sinkhorn_knopp_normalization
        self.decoder_blocks = nn.ModuleList([Block()for i in range(8)]) #decoder_depth
        self.decoder_norm = LayerNorm(512)
        self.decoder_embed = nn.Linear(512, 512, bias=True)
        self.transposed_conv = nn.ConvTranspose2d(in_channels=1, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, vision_patch_size + 1, embed_dim), requires_grad=False)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def z_score_normalization(self, data):
        mean = data.mean()
        std_dev = data.std()
        normalized_data = (data - mean) / std_dev
        return normalized_data

    def feature_mask(self, logits_per_image):
        logits_per_image = torch.where(logits_per_image < 0.5, torch.tensor(0.0).to(logits_per_image), logits_per_image)
        return logits_per_image

    def forward(self, noise, text_features):

        cluster_centers = C.forward(self, input_data=text_features, is_inference=True)

        image_features = torch.cat((noise, cluster_centers), dim=1)
        image_features = image_features.view(8, 1408)
        image_features = self.linear(image_features)
        image_features = image_features.view(8, 3, 14, 14)

        image_features = nn.ReLU(inplace=True)(image_features)
        image_features = self.c1(image_features)
        image_features = F.interpolate(image_features, size=(28, 28), mode='bilinear', align_corners=False)    #28

        image_features = nn.ReLU(inplace=True)(image_features)
        image_features = self.c2(image_features)
        image_features = self.visual2(image_features)
        image_features = image_features.unsqueeze(0)
        image_features = F.interpolate(image_features, size=(56, 56), mode='bilinear', align_corners=False)     #56

        image_features = nn.ReLU(inplace=True)(image_features)
        image_features = self.c3(image_features)
        image_features = self.visual3(image_features)
        image_features = image_features.unsqueeze(0)
        image_features = F.interpolate(image_features, size=(112, 112), mode='bilinear', align_corners=False)    #112

        image_features = nn.ReLU(inplace=True)(image_features)
        image_features = self.c4(image_features)
        image_features = self.visual4(image_features)
        image_features = image_features.unsqueeze(0)
        image_features = F.interpolate(image_features, size=(224, 224), mode='bilinear', align_corners=False)      #224
        image_features = self.c5(image_features)
        image_features = image_features.squeeze(0)

        # normalized features
        image_features = torch.tensor(image_features)
        text_features = torch.tensor(text_features)
        image_features = self.z_score_normalization(image_features)
        text_features = self.z_score_normalization(text_features)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        text_features = text_features.t()
        text_features = self.L(text_features.to(torch.float32)).to(device)

        logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        #mask
        logits_per_image = logits_per_image.cpu()
        logits_per_image = self.feature_mask(logits_per_image)

        #decoder
        logits_per_image = logits_per_image.to(device)
        x = self.decoder_embed(logits_per_image)

        noise_1 = torch.randn(8, 512)
        ids_shuffle = torch.argsort(noise_1, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        ids_restore = ids_restore.to(x_.device)
        ids_restore = ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        x_ = torch.gather(x_, dim=1, index=ids_restore)

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        # x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        # print(x.shape) #torch.Size([8, 512, 768])
        x = self.channel_transform(x)
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        fake_image = self.conv_img(x)

        return fake_image, logits_per_image, cluster_centers


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)  # 128
        self.block0 = resD(ndf * 1, ndf * 2)  # 64
        self.block1 = resD(ndf * 2, ndf * 4)  # 32
        self.block2 = resD(ndf * 4, ndf * 8)  # 16
        self.block3 = resD(ndf * 8, ndf * 16)  # 8
        self.block4 = resD(ndf * 16, ndf * 16)  # 4
        self.block5 = resD(ndf * 16, ndf * 16)  # 4

        self.COND_DNET = D_GET_LOGITS(ndf)

    def forward(self, x):

        out = self.conv_img(x)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        return out


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x) + self.gamma * self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.joint_conv = nn.Sequential(
            nn.Conv2d(512, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )
        self.conv = nn.Conv2d(1024, 256, 3, 1, 1, bias=False)

    def forward(self, out, y):

        y = y.view(-1, 256, 1, 1)
        y = y.repeat(1, 1, 4, 4)
        out = self.conv(out)
        indices = torch.randperm(y.size(0))[:1]
        y = y[indices, :, :, :]
        # print(out.shape, y.shape)
        h_c_code = torch.cat((out, y), 1)
        out = self.joint_conv(h_c_code)
        return out