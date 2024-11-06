# -*- encoding: utf-8 -*-
from __future__ import print_function
import multiprocessing

import os
import io
import sys
import time
import errno
import random
import pprint
import datetime
import dateutil.tz
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.utils.data as data
from miscc.utils import mkdir_p
from miscc.utils import imagenet_deprocess_batch
from miscc.config import cfg, cfg_from_file
from miscc.losses import DAMSM_loss#, discriminator_loss
from sync_batchnorm import DataParallelWithCallback
from datasets import TextDataset
from datasets import prepare_data
from transformers import CLIPProcessor, CLIPModel
from model import NetG, NetD, C
import clip
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# 离线状态下可运行
# TRANSFORMERS_OFFLINE = 1

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
multiprocessing.set_start_method('spawn', True)
UPDATE_INTERVAL = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def sampling(netG, dataloader, ixtoword, device):
    model_dir = cfg.TRAIN.NET_G
    istart = cfg.TRAIN.NET_G.rfind('_') + 1
    iend = cfg.TRAIN.NET_G.rfind('.')
    start_epoch = int(cfg.TRAIN.NET_G[istart:iend])
    # hard debug by setting the index of trained epoch, adjust it as your need

    split_dir = 'valid'
    #split_dir = 'test_every'
    # Build and load the generator
    netG.load_state_dict(torch.load(model_dir))
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    #s_tmp = model_dir
    s_tmp = model_dir[:model_dir.rfind('.pth')]
    s_tmp_dir = s_tmp
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(fake_img_save_dir)

    real_img_save_dir = '%s/%s' % (s_tmp, 'real')
    mkdir_p(real_img_save_dir)
    cap_save_dir = '%s/%s' % (s_tmp, 'caps')
    mkdir_p(cap_save_dir)

    idx = 0
    cnt = 0
    # hard debug by specifyng the number of synthezied images from caption
    for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            text_encoder = model.encoder_text(captions)
            real_imgs = imags[0].to(device)
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            # if step > 50:
            #     break
            # hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            # words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            # words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            # code for generating captions
            #cap_imgs = cap2img_new(ixtoword, captions, cap_lens, s_tmp_dir)

            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
                noise = torch.randn(batch_size, 100)
                noise = noise.to(device)
                text_feature = text_encoder(captions)
                fake_imgs, stage_masks, _ = netG(noise, text_feature)
                stage_mask = stage_masks[-1]
            for j in range(batch_size):
                # save generated image
                s_tmp = '%s/img' % (fake_img_save_dir)
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)

                #fullpath = '%s_%3d.png' % (s_tmp,i)
                fullpath = '%s_s%d.png' % (s_tmp, idx)
                im.save(fullpath)

                # save the last fusion mask
                s_tmp = '%s/fm' % fake_img_save_dir
                im = stage_mask[j].data.cpu().numpy()
                # [0, 1] --> [0, 255]
                # im = 1-im # only for better visualization
                im = im * 255.0
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = np.squeeze(im, axis=2)
                im = Image.fromarray(im)
                fullpath = '%s_%d.png' % (s_tmp, idx)
                im.save(fullpath)

                idx += 1


def gen_sample(netG, device, wordtoix):
    """
    generate sample according to user defined captions.

    caption should be in the form of a list, and each element of the list is a description of the image in form of string.
    caption length should be no longer than 18 words.
    example captions see below
    """
    captions = ['A colorful blue bird has wings with dark stripes and small eyes',
                'A colorful green bird has wings with dark stripes and small eyes',
                'A colorful white bird has wings with dark stripes and small eyes',
                'A colorful black bird has wings with dark stripes and small eyes',
                'A colorful pink bird has wings with dark stripes and small eyes',
                'A colorful orange bird has wings with dark stripes and small eyes',
                'A colorful brown bird has wings with dark stripes and small eyes',
                'A colorful red bird has wings with dark stripes and small eyes',
                'A colorful yellow bird has wings with dark stripes and small eyes',
                'A colorful purple bird has wings with dark stripes and small eyes']

    # captions = ['A herd of black and white cattle standing on a field',
    #  'A herd of black cattle standing on a field',
    #  'A herd of white cattle standing on a field',
    #  'A herd of brown cattle standing on a field',
    #  'A herd of black and white sheep standing on a field',
    #  'A herd of black sheep standing on a field',
    #  'A herd of white sheep standing on a field',
    #  'A herd of brown sheep standing on a field']

    # captions = ['some horses in a field of green grass with a sky in the background',
    #  'some horses in a field of yellow grass with a sky in the background',
    #  'some horses in a field of green grass with a sunset in the background',
    #  'some horses in a field of yellow grass with a sunset in the background']

    # caption to idx
    # split string to word
    for c, i in enumerate(captions):
        captions[c] = i.split()

    caps = torch.zeros((len(captions), 18), dtype=torch.int64)

    for cl, line in enumerate(captions):
        for cw, word in enumerate(line):
            caps[cl][cw] = wordtoix[word.lower()]
    caps = caps.to(device)
    cap_len = []
    for i in captions:
        cap_len.append(len(i))

    caps_lens = torch.tensor(cap_len, dtype=torch.int64).to(device)

    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    netG.load_state_dict(torch.load(model_dir))
    netG.eval()

    batch_size = len(captions)
    s_tmp = model_dir[:model_dir.rfind('.pth')]
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(fake_img_save_dir)

    for step in range(50):

        # hidden = text_encoder.init_hidden(batch_size)
        # words_embs, sent_emb = text_encoder(caps, caps_lens, hidden)
        # words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

        #######################################################
        # (2) Generate fake images
        ######################################################
        with torch.no_grad():
            # noise = torch.randn(1, 100) # using fixed noise
            # noise = noise.repeat(batch_size, 1)
            # use different noise
            noise = []
            for i in range(batch_size):
                noise.append(torch.randn(1, 100))
            noise = torch.cat(noise, 0)
            
            noise = noise.to(device)
            text_features = model.encoder_text(captions)
            fake_imgs, stage_masks, _ = netG(noise, text_features)
            stage_mask = stage_masks[-1]
        for j in range(batch_size):
            # save generated image
            s_tmp = '%s/img' % fake_img_save_dir
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            # fullpath = '%s_%3d.png' % (s_tmp,i)
            fullpath = '%s_%d.png' % (s_tmp, step)
            im.save(fullpath)

            # save fusion mask
            s_tmp = '%s/fm' % fake_img_save_dir
            im = stage_mask[j].data.cpu().numpy()
            # im = 1-im # only for better visualization
            # [0, 1] --> [0, 255]
            im = im * 255.0
            im = im.astype(np.uint8)

            im = np.transpose(im, (1, 2, 0))
            im = np.squeeze(im, axis=2)
            im = Image.fromarray(im)
            fullpath = '%s_%d.png' % (s_tmp, step)
            im.save(fullpath)


def cap2img(ixtoword, caps, cap_lens):
    imgs = []
    for cap, cap_len in zip(caps, cap_lens):
        idx = cap[:cap_len].cpu().numpy()
        caption = []
        for i, index in enumerate(idx, start=1):
            caption.append(ixtoword[index])
            if i % 4 == 0 and i > 0:
                caption.append("\n")
        caption = " ".join(caption)
        fig = plt.figure(figsize=(2.5, 1.5))
        plt.axis("off")
        plt.text(0.5, 0.5, caption)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = transforms.ToTensor()(img)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)
    assert imgs.dim() == 4, "image dimension must be 4D"
    return imgs


def write_images_losses(writer, imgs, fake_imgs, errD, d_loss, errG, DAMSM, epoch):
    index = epoch
    writer.add_scalar('errD/d_loss', errD, index)
    writer.add_scalar('errD/MAGP', d_loss, index)
    writer.add_scalar('errG/g_loss', errG, index)
    writer.add_scalar('errG/DAMSM', DAMSM, index)
    imgs_print = imagenet_deprocess_batch(imgs)
    imgs_256_print = imagenet_deprocess_batch(fake_imgs)
    writer.add_image('images/img1_pred', torchvision.utils.make_grid(imgs_256_print, normalize=True, scale_each=True), index)
    #writer.add_image('images/img2_caption', torchvision.utils.make_grid(cap_imgs, normalize=True, scale_each=True), index)
    writer.add_image('images/img3_real', torchvision.utils.make_grid(imgs_print, normalize=True, scale_each=True), index)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def prepare_labels(batch_size):
    # Kai: real_labels and fake_labels have data type: torch.float32
    # match_labels has data type: torch.int64
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if cfg.CUDA:
        real_labels = real_labels.to(device)
        fake_labels = fake_labels.to(device)
        match_labels = match_labels.to(device)
    return real_labels, fake_labels, match_labels


def train(dataloader, captions, netG, netD, optimizerG, optimizerD, state_epoch, batch_size, device):

    base_dir = os.path.join('tmp', cfg.CONFIG_NAME, str(cfg.TRAIN.NF))

    if not cfg.RESTORE:
        writer = SummaryWriter(os.path.join(base_dir, 'writer'))
    else:
        writer = SummaryWriter(os.path.join(base_dir, 'writer_new'))

    mkdir_p('%s/models' % base_dir)
    real_labels, fake_labels, match_labels = prepare_labels(batch_size)

    # Build and load the generator and discriminator
    if cfg.RESTORE:
        model_dir = cfg.TRAIN.NET_G
        netG.load_state_dict(torch.load(model_dir))
        model_dir_D = model_dir.replace('netG', 'netD')
        netD.load_state_dict(torch.load(model_dir_D))
        netG.train()
        netD.train()
        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        state_epoch = int(cfg.TRAIN.NET_G[istart:iend])

    for epoch in tqdm(range(state_epoch + 1, cfg.TRAIN.MAX_EPOCH + 1)):
        data_iter = iter(dataloader)
        # for step, data in enumerate(dataloader, 0):
        for step in range(len(data_iter)):
            data = next(data_iter)
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            # captions = [item for tpl in captions for item in tpl]
            # captions = clip.tokenize(captions).to(device)
            # with torch.no_grad():
            #     text_features = model.encode_text(captions)
            text_features = captions

            # synthesize fake images
            noise = torch.randn(11, 512)
            noise = noise.to(device)
            fake_imgs, stage_masks, code = netG(noise, text_features)

            imgs = imags[0].to(device)
            real_features = netD(imgs)
            output = netD.module.COND_DNET(real_features, text_features)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.module.COND_DNET(real_features[:(batch_size - 1)], text_features[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()


            # G does not need update with D

            fake_features = netD(fake_imgs.detach())
            errD_fake = netD.module.COND_DNET(fake_features, text_features)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

            errD = errD_real + errD_fake# + discriminator_loss

            optimizerD.zero_grad()
            errD.backward()
            optimizerD.step()

            # MA-GP
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (text_features.data).requires_grad_()
            features = netD(interpolated)
            out = netD.module.COND_DNET(features, sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated, sent_inter),
                                        grad_outputs=torch.ones(out.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            # 假设 grad0 是你需要调整的张量
            grad1 = grad1.view(1, -1)

            grad = torch.cat((grad0, grad1), dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # update G
            features = netD(fake_imgs)
            output = netD.module.COND_DNET(features, text_features)
            errG = - output.mean()
            DAMSM = 0.05 * DAMSM_loss(preprocess, fake_imgs, real_labels, text_features, match_labels, cap_lens, class_ids)
            errG_total = errG + DAMSM
            optimizerG.zero_grad()  # Make sure to call this before backward()
            errG_total.backward()
            optimizerG.step()

            # caption can be converted to image and shown in tensorboard
            #cap_imgs = cap2img(ixtoword, captions, cap_lens)

            write_images_losses(writer, imgs, fake_imgs, errD, d_loss, errG, DAMSM, epoch)

        if (epoch >= cfg.TRAIN.WARMUP_EPOCHS) and (epoch % cfg.TRAIN.GSAVE_INTERVAL == 0):
            torch.save(netG.state_dict(), '%s/models/netG_%03d.pth' % (base_dir, epoch))
        if (epoch >= cfg.TRAIN.WARMUP_EPOCHS) and (epoch % cfg.TRAIN.DSAVE_INTERVAL == 0):
            torch.save(netD.state_dict(), '%s/models/netD_%03d.pth' % (base_dir, epoch))


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    # Kai: i don't want to specify a gpu id
    # torch.cuda.set_device(cfg.GPU_ID)

    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        ixtoword = dataset.ixtoword
        wordtoix = dataset.wordtoix
        caption = dataset.captions
        # print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        ixtoword = dataset.ixtoword
        caption = dataset.captions
        # print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # validation data #

    for step, data in enumerate(dataloader, 0):
        imags, captions, cap_lens, class_ids, keys = data

    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    documents = [item for tpl in captions for item in tpl]

    # 创建并训练模型
    text_clustering_model = C(input_dim=512, num_clusters=10, output_dim=100)
    text_clustering_model.train(documents)

    state_epoch = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG().to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    netG = DataParallelWithCallback(netG)
    # netD = DataParallelWithCallback(netD)
    netD = nn.DataParallel(netD)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))


    if cfg.B_VALIDATION:
        sampling(netG, dataloader, ixtoword, device)  # generate images for the whole valid dataset
        # gen_sample(text_encoder, netG, device, wordtoix) # generate images with description from user
    else:
        # print("device", device)
        train(dataloader, captions, netG, netD, optimizerG, optimizerD, state_epoch, batch_size, device)
