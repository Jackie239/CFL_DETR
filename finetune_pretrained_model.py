import torch
pretrained_weights  = torch.load('./model_pretrain/detr-r101-2c7b67e5.pth')

#NWPU数据集，10类
num_class = 2    #类别数+1，1为背景
pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1, 256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_class+1)
torch.save(pretrained_weights, "./model_pretrain/detr-r50_%d.pth"%num_class)
