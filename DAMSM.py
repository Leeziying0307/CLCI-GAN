# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from main import model, preprocess
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):

    def forward(self, captions, cap_lens, hidden, mask=None):
        text_features = model.encoder_text(captions)
        return text_features


class CNN_ENCODER(nn.Module):

    def forward(self, x):
        image_input = preprocess(x).unsqueeze(0)
        image_features = model.encode_image(image_input)
        return image_features
