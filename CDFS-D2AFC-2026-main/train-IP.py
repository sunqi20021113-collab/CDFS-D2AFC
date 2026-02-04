import numpy as np
import os
import argparse
import pickle
import time
import imp
import random
# import importlib.util
import logging
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model.mapping import Mapping
from model.encoder import Encoder, PrototypeGenerator
from model.Network import Former, CSSTFormer, SSFE_net
from utils.dataloader import get_HBKC_data_loader, Task, get_target_dataset, tagetSSLDataset,get_allpixel_loader
from utils import utils, loss_function, data_augment
from model.SCFormer_model_copy import MultiScaleMaskedTransformerDecoder as mynet
from model.CITM_fsl import *

from model import loss
from model.loss import coral_loss
from model.loss import LFCN
from model.loss import ConTeXLoss


# import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument('--config', type=str, default=os.path.join('./config', 'IP.py'))
args = parser.parse_args()

# load hyperparameters
# spec = importlib.util.spec_from_file_location('module_name', 'path/to/module.py')
# config = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(config)
config = imp.load_source("", args.config).config
train_opt = config['train_config']
data_path = config['data_path']
source_data = config['source_data']
target_data = config['target_data']
target_data_gt = config['target_data_gt']
log_dir = config['log_dir']
patch_size = train_opt['patch_size']
batch_task = train_opt['batch_task']
emb_size = train_opt['d_emb']
SRC_INPUT_DIMENSION = train_opt['src_input_dim']
TAR_INPUT_DIMENSION = train_opt['tar_input_dim']
N_DIMENSION = train_opt['n_dim']                       # 目前还不清楚作用
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class']
EPISODE = train_opt['episode']
LEARNING_RATE = train_opt['lr']
GPU = config['gpu']
TAR_CLASS_NUM = train_opt['tar_class_num']  # the number of class
TAR_LSAMPLE_NUM_PER_CLASS = train_opt['tar_lsample_num_per_class']  # 每个类别的标签样本数
SCR_CLASS_NUM = train_opt['scr_class_num']   # 源域样本数
WEIGHT_DECAY = train_opt['weight_decay']

sk_ratio = train_opt['sk_ratio']  # 负值表示不使用比例
tk_ratio = train_opt['tk_ratio']
sk = train_opt['sk']
tk = train_opt['tk']

utils.same_seeds(0)

# get src/tar class number -> label semantic vector
# labels_src = ["Breeding Area", "Salt Pan Crystallization Pond", "Field", "Reed", "Salt Field",
#               "Natural Willow Forest", "Other Forest Land", "Reed and Tamarisk Mixed Vegetation", "Tamarisk and Saltbush Mixed Vegetation", "Saline Land Saltbush Growth Area",
#               "Pit Pond", "Oil Well Platform", "Saline-Alkaline Flats", "Bare Beach", "Yellow River", "Reed and Saltbush Mixed Vegetation",
#               "Reed, Saltbush, and Tamarisk Mixed Vegetation", "Sea", "Smooth Cordgrass Management Area"]
#
# # GF5-1105
# labels_tar = ["Sea", "Yellow River", "Oil Well Platform", "Tamarisk and Saltbush Mixed Vegetation", "Saline-Alkaline Flats", "Bare Beach", "Pit Pond",
#               "Reed and Saltbush Mixed Vegetation", "Reed, Saltbush, and Tamarisk Mixed Vegetation", "Reed", "Reed and Tamarisk Mixed Vegetation", "Natural Willow Forest",
#               "Saline Land Saltbush Growth Area", "Smooth Cordgrass Management Area"]
#
#
# from transformers import BertModel, BertTokenizer
#
# # 标签特征提取器
# model = BertModel.from_pretrained('pretrain-model/bert-base-uncased')    # 加载预训练模型
# model.eval()              # 切换成评估模式
# tokenizer = BertTokenizer.from_pretrained('pretrain-model/bert-base-uncased')   # 加载与模型配套的分词器，确保分词方式与模型预训练时一致
#
# encoded_inputs_src = tokenizer(labels_src, padding=True, truncation=True, return_tensors='pt')  # 对源域标签进行分词  注意使用的是映射前的源域还是映射后的
# with torch.no_grad():
#     outputs_src = model(**encoded_inputs_src)     # 获取模型输出
#
#  # 提取每个隐藏层输出的第一个 token（[CLS]）的向量，形状为 (num_classes, 768)，用于表示类别的语义向量   # [CLS]向量通常用于表示整个序列的语义
# semantic_mapping_src = outputs_src.last_hidden_state[:, 0, :]  # (num_classess, 768)
#
# encoded_inputs_tar = tokenizer(labels_tar, padding=True, truncation=True, return_tensors='pt')
# with torch.no_grad():
#     outputs_tar = model(**encoded_inputs_tar)
# semantic_mapping_tar = outputs_tar.last_hidden_state[:, 0, :]  # (num_classess, 768)
#
# semantic_mapping_src = semantic_mapping_src.cpu().numpy()    # 将pytorch张量转化为numpy数组
# semantic_mapping_tar = semantic_mapping_tar.cpu().numpy()

# load source domain data
print('----------------------------load source domain data----------------------------')
with open(os.path.join(data_path, source_data), 'rb') as handle:    # 加载源域数据
    source_imdb = pickle.load(handle)

data_train = source_imdb['data']
labels_train = source_imdb['Labels']


keys_all_train = sorted(list(set(labels_train)))     # 去重标签，对类别进行排序
label_encoder_train = {}                      # 创建一个字典，将每个类别映射为唯一的索引
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
train_dict = {}                                        # 构建训练字典
for class_, path in zip(labels_train, data_train):    # 将标签列表labels_train和数据路径列表data_train逐项配对，形成(class_, path)的元组
    if label_encoder_train[class_] not in train_dict:     # 将类别标签 class_ 转换为对应的类别索引
        train_dict[label_encoder_train[class_]] = []
    train_dict[label_encoder_train[class_]].append(path)
del keys_all_train
del label_encoder_train

metatrain_data = utils.sanity_check(train_dict)

# 遍历metatrain_data中的每个类别的数据，将其维度从(H, W, C)变为(C, H, W)
for class_ in metatrain_data:
    for i in range(len(metatrain_data[class_])):
        metatrain_data[class_][i] = np.transpose(metatrain_data[class_][i], (2, 0, 1))

# source domain adaptation data
print(source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape)
print(source_imdb['Labels'].shape)
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
del source_dataset, source_imdb

# load target data
print('--------------------------------load target data-------------------------------')
test_data = os.path.join(data_path, target_data)
test_label = os.path.join(data_path, target_data_gt)
Data_Band_Scaler, GroundTruth = utils.load_data_IP(test_data, test_label)  # 加载测试数据和标签

# loss init
crossEntropy = nn.CrossEntropyLoss().to(GPU)
# cos_criterion = nn.CosineSimilarity(dim=1).to(GPU)    # 余弦相似度函数
# infoNCE_Loss = loss_function.ContrastiveLoss(batch_size=TAR_CLASS_NUM).to(GPU)   # 自定义的对比损失函数
# infoNCE_Loss_SSL = loss_function.ContrastiveLoss(batch_size=128).to(GPU)    # 对比损失函数
SupConLoss_t = loss_function.SupConLoss(temperature=0.1).to(GPU)            # 监督对比损失函数，拉近文本与图像原型

# lmmd = loss.stMMD_loss(class_num=TAR_CLASS_NUM)     # 定义子域对齐损失函数
wd_loss = loss.SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')   # Wasserstein距离损失  用于全局分布域对齐
# mmd_loss = loss.MMD_loss(kernel_type='linear')   # 最大均值差异  用于全局分布域对齐
# domain_criterion = nn.BCEWithLogitsLoss().to(GPU)    # 二进制交叉熵损失 域判别损失
domain_criterion_2 = nn.BCEWithLogitsLoss().to(GPU)    # 二进制交叉熵损失 域判别损失 D2

contex_loss_fn = ConTeXLoss(temperature=0.1, lambda_weight=0.5).to(GPU)

# awd_da_loss = loss.AWD_DA_Loss(epsilon=0.05)  # 自适应沃瑟斯坦距离损失


# experimental result index
nDataSet = 5    # 定义运行次数
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TAR_CLASS_NUM])   # 存储每次实验中每个类别的结果
k = np.zeros([nDataSet, 1])
f1_scores = np.zeros([nDataSet, 1])  # 存储F1-score
best_predict_all = []   # 用于存储所有实验中的最佳预测结果
best_predict_full = []
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None   # 存储实验过程中的最佳参数
best_full_G, best_Row_all, best_Column_all, best_nTrain_all = None, None, None, None


# seeds = [1236, 1237, 1226, 1227, 1211, 1212, 1216, 1240, 1222, 1223]
seeds = [1228, 1237, 1226, 1227, 1211, 1212, 1241, 1240, 1222, 1223]

# log setting
experimentSetting = '{}way_{}shot_{}'.format(TAR_CLASS_NUM, TAR_LSAMPLE_NUM_PER_CLASS, target_data.split('/')[0])
utils.set_logging_config(os.path.join(log_dir, experimentSetting), nDataSet)   # 存储实验日志
logger = logging.getLogger('main')    # logging.getLogger('main')：获取名为 main 的日志记录器；该记录器用于输出日志信息，例如调试、错误、警告等
logger.info('seeds_list:{}'.format(seeds))    # 记录种子列表


for iDataSet in range(nDataSet):
    logger.info('emb_size:{}'.format(emb_size))
    logger.info('patch_size:{}'.format(patch_size))
    logger.info('seeds:{}'.format(seeds[iDataSet]))

    utils.same_seeds(seeds[iDataSet])    # 确保在训练过程中使用相同的随机种子，从而使得实验结果可复现

    # 加载包括目标域训练数据、测试数据、目标域元训练（增强之后）数据、填充后的图像大小、数据位置索引、行、列、训练样本数、增强数据平铺列表数据、增强数据标签平铺列表数据
    train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain, target_aug_data_ssl, target_aug_label_ssl = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=TAR_CLASS_NUM,
        tar_lsample_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        shot_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        patch_size=patch_size)


    # 单独调用你新写的全像素 loader
    # all_test_loader, G_pad, Row_all, Col_all, nTrain_all = get_allpixel_loader(
    #     Data_Band_Scaler=Data_Band_Scaler,
    #     GroundTruth=GroundTruth,
    #     HalfWidth=patch_size // 2

    # )

    target_ssl_dataset = tagetSSLDataset(target_aug_data_ssl, target_aug_label_ssl)      # 目标域数据加载器
    target_ssl_dataloader = torch.utils.data.DataLoader(target_ssl_dataset, batch_size=64, shuffle=True, drop_last=True)

    # 生成小样本学习任务所需的掩码和参数  一个小样本任务的支持集大小、总样本大小、查询集掩码
    num_supports, num_samples, query_edge_mask, evaluation_mask = utils.preprocess(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS,
                                                                                   QUERY_NUM_PER_CLASS, batch_task, GPU)
    # 定义源域和目标域的映射网络（将原始特征映射到公共空间）与特征提取网络Encoder
    mapping_src = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    mapping_tar = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    # domain_classifier = DomainClassifier().to(GPU)
    # random_layer = RandomLayer([emb_size, TAR_CLASS_NUM], 1024).to(GPU)
    cross_transformer = CrossTransformer(embed_dim=emb_size, num_heads=8) # 初始化 Cross-Transformer
    domain_classifier_2 = DomainDiscriminator_2().to(GPU)
    # lfcn = LFCN(input_dim=emb_size)   # 计算自适应代价矩阵的轻量化网络


    encoder = Net(inp_channels=N_DIMENSION, dim=emb_size, depths=[1], num_heads_spa=[8], num_heads_spe=[7], dropout=0.3).to(GPU)


    mapping_src_optim = torch.optim.SGD(mapping_src.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                        weight_decay=WEIGHT_DECAY)
    mapping_tar_optim = torch.optim.SGD(mapping_tar.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                        weight_decay=WEIGHT_DECAY)
    encoder_optim = torch.optim.SGD(encoder.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    # domain_classifier_optim = torch.optim.SGD(domain_classifier.parameters(), lr=LEARNING_RATE, momentum=0.9,
    #                                           weight_decay=WEIGHT_DECAY)

    cross_transformer_optim = torch.optim.SGD(cross_transformer.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                              weight_decay=WEIGHT_DECAY)
    domain_classifier_2_optim = torch.optim.SGD(domain_classifier_2.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                              weight_decay=WEIGHT_DECAY)
    # lfcn_optim = torch.optim.SGD(domain_classifier_2.parameters(), lr=LEARNING_RATE, momentum=0.9,
    #                                           weight_decay=WEIGHT_DECAY)


    # 应用自定义权重初始化
    mapping_src.apply(utils.weights_init)
    mapping_tar.apply(utils.weights_init)
    # domain_classifier.apply(utils.weights_init)
    domain_classifier_2.apply(utils.weights_init)
    # encoder.apply(utils.weights_init)
    # lfcn.apply(utils.weights_init)

    mapping_src.to(GPU)
    mapping_tar.to(GPU)
    # domain_classifier.to(GPU)
    domain_classifier_2.to(GPU)
    cross_transformer.to(GPU)
    encoder.to(GPU)
    # lfcn.to(GPU)

    # 设置为训练模式（启用Dropout等训练专用层）
    mapping_src.train()
    mapping_tar.train()
    # domain_classifier.train()
    domain_classifier_2.train()
    encoder.train()
    cross_transformer.train()
    # lfcn.train()

    logger.info("Training...")
    last_accuracy = 0.0                # 记录上一次准确率
    best_episode = 0                   # 最佳训练轮次
    # 源域正确预测数/总样本数  目标域相同统计  当前准确率
    total_hit_src, total_num_src, total_hit_tar, total_num_tar, acc_src, acc_tar = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    train_start = time.time()
    writer = SummaryWriter()       # 创建可视化日志写入器

    # 获取目标域数据迭代器（用于masked监督对比学习）
    target_ssl_iter = iter(target_ssl_dataloader)       # 可无限迭代的数据加载器

    # 获取域对齐数据迭代器
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)


    for episode in range(EPISODE):                # 遍历所有训练轮次（元学习中的episode概念）
        # 源域任务构建
        task_src = Task(metatrain_data, TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        # 支持集数据加载器构建
        support_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                      shuffle=False)
        # 查询集数据加载器构建
        query_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                    shuffle=False)

        # 测试阶段只在目标域的 TAR_CLASS_NUM 个类别上进行，因此训练阶段必须模拟相同类别数的任务结构
        # 目标域任务构建
        task_tar = Task(target_da_metatrain_data, TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_tar = get_HBKC_data_loader(task_tar, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                      shuffle=False)
        query_dataloader_tar = get_HBKC_data_loader(task_tar, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                    shuffle=False)

        # 加载源域数据
        support_src, support_label_src = support_dataloader_src.__iter__().next()
        query_src, query_label_src = query_dataloader_src.__iter__().next()

        support_real_labels_src = task_src.support_real_labels
        support_real_labels_tar = task_tar.support_real_labels

        # 目标域同理
        support_tar, support_label_tar = support_dataloader_tar.__iter__().next()
        query_tar, query_label_tar = query_dataloader_tar.__iter__().next()

        # print('support_src', support_src.shape)   # [14, 76, 9, 9]
        # print('query_src', query_src.shape)       # [266, 76, 9, 9]
        # print('support_tar', support_tar.shape)   # [14, 150, 7, 7]
        # print('query_tar', query_tar.shape)       # [266, 150, 7, 7]

        # 源域特征提取
        support_features_src, support_output_src = encoder(mapping_src(support_src.to(GPU)))  # (14, 128)
        query_features_src, query_output_src = encoder(mapping_src(query_src.to(GPU)))            # (266, 128)

        # 目标域特征提取（同理）
        support_features_tar, support_output_tar = encoder(mapping_tar(support_tar.to(GPU)))  # (14, 128)    包含一个语义特征的映射过程
        query_features_tar, query_output_tar = encoder(mapping_tar(query_tar.to(GPU)))   # torch.Size([266, 128])

        # 自适应沃瑟斯坦距离域对齐
        # src_feature = torch.cat([support_features_src, query_features_src], dim=0)
        # tar_feature = torch.cat([support_features_tar, query_features_tar], dim=0)
        # C_A = lfcn(src_feature, tar_feature)
        # loss_wd, _ = awd_da_loss(C_A)

        # 原型计算（Prototypical Networks核心）
        if SHOT_NUM_PER_CLASS > 1:          # 多样本时取平均
            support_proto_src = support_features_src.reshape(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            support_proto_tar = support_features_tar.reshape(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)

        else:                               # 单样本直接使用
            support_proto_src = support_features_src
            support_proto_tar = support_features_tar

        # 分类损失（交叉熵）
        logits_src = utils.euclidean_metric(query_features_src, support_proto_src)     # 欧氏距离计算相似度
        f_loss_src = crossEntropy(logits_src, query_label_src.long().to(GPU))          # 源域分类损失

        logits_tar = utils.euclidean_metric(query_features_tar, support_proto_tar)
        f_loss_tar = crossEntropy(logits_tar, query_label_tar.long().to(GPU))          # 目标域分类损失

        f_loss = f_loss_src + f_loss_tar  # 总分类损失


        # 目标域监督对比学习
        try:       # 获取增强数据
            target_ssl_data, target_ssl_label = target_ssl_iter.next()
        except Exception as err:       # 重新初始化迭代器
            target_ssl_iter = iter(target_ssl_dataloader)
            target_ssl_data, target_ssl_label = target_ssl_iter.next()

        # 数据增强（随机遮挡）
        augment1_target_ssl_data = torch.FloatTensor(
            data_augment.random_mask_batch_spatial(target_ssl_data.data.cpu(), 0.6))  # (64, 150, 7, 7)
        augment2_target_ssl_data = torch.FloatTensor(
            data_augment.random_mask_batch_spatial(target_ssl_data.data.cpu(), 0.6))  # (64, 150, 7, 7)
        augment_target_ssl_data = torch.cat((augment1_target_ssl_data, augment2_target_ssl_data),
                                            dim=0)  #````````````````````````````````````````` ([128, 150, 7, 7])
        features_augment, _ = encoder(mapping_tar(augment_target_ssl_data.to(GPU)))  # (128, 128)   特征提取

        augment1_target_ssl_feature = F.normalize(features_augment[:len(target_ssl_data), :], dim=1)  # (64, 128)   标准化
        augment2_target_ssl_feature = F.normalize(features_augment[len(target_ssl_data):, :], dim=1)  # (64, 128)
        augment_target_ssl_feature = torch.cat(
            [augment1_target_ssl_feature.unsqueeze(1), augment2_target_ssl_feature.unsqueeze(1)],
            dim=1)  # (128, 2, 128)
        # scl_loss_tar = SupConLoss_t(augment_target_ssl_feature, target_ssl_label, None, GPU)   # 对比损失计算
        # CTx_loss_tar = SupConLoss_t(augment_target_ssl_feature, target_ssl_label, None, GPU)  # 对比损失计算

        # Contex loss
        B, dim_ssl, _, _ = augment1_target_ssl_data.shape
        label_ssl = torch.cat([target_ssl_label, target_ssl_label], dim=0).to(GPU)  # (128, 128)
        feature_ssl = features_augment
        # 3. 生成 sample_ids
        ids = torch.arange(B, device=GPU)
        sample_ids = torch.cat([ids, ids], dim=0)  # [128]
        # 假设已实例化
        CTx_loss_tar = contex_loss_fn(feature_ssl, label_ssl, sample_ids)


        # get domain adaptation data from  source domain and target domain  获取域对齐数据（进行整体分布域对齐）
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()
        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()
        # 源域和目标域特征掩码增强
        # augment1_src_data = torch.FloatTensor(
        #     data_augment.random_mask_batch_spatial(source_data.data.cpu(), 0.8))  # (64, 150, 7, 7)
        # augment1_tar_data = torch.FloatTensor(
        #     data_augment.random_mask_batch_spatial(target_data.data.cpu(), 0.8))  # (64, 150, 7, 7)
        # augment1_src_data = mapping_src(augment1_src_data.to(GPU))
        # augment1_tar_data = mapping_tar(augment1_tar_data.to(GPU))
        # augment_data = torch.cat((augment1_src_data, augment1_tar_data),
        #                                     dim=0)
        # # 生成多源聚合特征与对应标签
        # features_augment_src_tar, _ = encoder((augment_data.to(GPU)))  # (256, 128)   特征提取
        # augment_src_feature = F.normalize(features_augment_src_tar[:len(source_data), :], dim=1)  # (128, 128)   标准化
        # augment_tar_feature = F.normalize(features_augment_src_tar[len(target_data):, :], dim=1)  # (128, 128)
        # augment_tar_src_feature = torch.cat(
        #     [augment_src_feature.unsqueeze(1), augment_tar_feature.unsqueeze(1)],)
        # labels_src_tar = torch.cat((source_label, target_label),
        #                                     dim=0)  # ([256, ])
        # # 监督对比损失
        # scl_loss_tar = SupConLoss_t(augment_tar_src_feature, labels_src_tar, None, GPU)   # 对比损失计算

        # 特征提取
        wd_s, _ = encoder(mapping_src(source_data.to(GPU)))  # (64, 128)   特征提取
        wd_t, _ = encoder(mapping_tar(target_data.to(GPU)))  # (64, 128)   特征提取
        loss_wd, _, _ = wd_loss(wd_s, wd_t)
        # loss_mmd = mmd_loss(wd_s, wd_t)
        # loss_coral = coral_loss(wd_s, wd_t)   # 计算 CORAL 损失，输入源域和目标域的特征矩阵

        # 利用域判别器进行域对齐
        # features = torch.cat([support_features_src, query_features_src, support_features_tar, query_features_tar], dim=0)
        # outputs = torch.cat([support_output_src, query_output_src, support_output_tar, query_output_tar], dim=0)
        # softmax_output = nn.Softmax(dim=1)(outputs)
        # labels = torch.cat([support_label_src, query_label_src, support_label_tar, query_label_tar], dim=0)
        # labels_hot = one_hot(labels, TAR_CLASS_NUM).to(GPU)
        # combined = torch.cat([features, labels_hot], dim=1)
        # # 给每个样本创建域标签
        # domain_label = torch.zeros([support_features_src.shape[0] + query_features_src.shape[0]
        #                             + support_features_tar.shape[0] + query_features_tar.shape[0], 1]).to(GPU)
        # domain_label[:support_features_src.shape[0] + query_features_src.shape[0]] = 1  # 将源域支持样本和查询样本的域标签设置为1，目标域样本设为0
        # randomlayer_out = random_layer.forward([features, softmax_output])
        #
        # domain_logits = domain_classifier(randomlayer_out, episode)
        # domain_loss = domain_criterion(domain_logits, domain_label)  # 计算域损失

        # 分布级域适应对齐
        features_src = torch.cat([support_features_src, query_features_src], dim=0)
        features_tar = torch.cat([support_features_tar, query_features_tar], dim=0)
        # 得到分布对齐后的特征 f_s_d, f_t_d (各 (280, 128))
        f_s_d, f_t_d = cross_transformer(features_src, features_tar)  # torch.Size([140, 128])
        # 将对齐后的特征拼接，作为判别器输入： (20, 512)
        F_d = torch.cat([f_s_d, f_t_d], dim=0)
        # 得到域分类 logits (20, 1)
        domain_logits_2 = domain_classifier_2(F_d).to(GPU)
        # 构造真实域标签：前 10 行为 1（源域），后 10 行为 0（目标域）
        domain_label_2 = torch.cat([torch.ones(f_s_d.size(0), 1), torch.zeros(f_t_d.size(0), 1)], dim=0).to(GPU)
        # 计算对抗损失（判别器方向）
        loss_disc = domain_criterion_2(domain_logits_2, domain_label_2)


        # 总损失
        # loss = f_loss + 2.0 * scl_loss_tar + 2*loss_lmmd
        # loss = f_loss + 2.0*scl_loss_tar + 0.001 * loss_wd
        # loss = f_loss + 2.0 * scl_loss_tar + 0.1 * loss_mmd
        # loss = f_loss + 2.0 * scl_loss_tar + 0.1*loss_wd
        # loss = f_loss + 2.0 * scl_loss_tar
        # loss = f_loss
        # loss = f_loss + 2.0 * scl_loss_tar + domain_loss
        loss = f_loss + 2.0 * CTx_loss_tar + 0.001 * loss_wd + 1 * loss_disc
        # loss = f_loss + 2.0 * scl_loss_tar + 0.001 * loss_wd + loss_disc
        # loss = f_loss + 2.0 * scl_loss_tar + loss_coral + loss_disc


        # scl_loss_tar [0.5, 1, 1.5, 2, 2.5,3.0]
        # loss_wd [100, 10, 1, 0.1, 0.01, 0.001]  0.1、1、10、100都不行

        mapping_src.zero_grad()
        mapping_tar.zero_grad()
        encoder.zero_grad()
        # domain_classifier_optim.zero_grad()
        domain_classifier_2_optim.zero_grad()
        cross_transformer_optim.zero_grad()
        # lfcn_optim.zero_grad()

        loss.backward()   # 反向传播

        # 参数更新
        mapping_src_optim.step()
        mapping_tar_optim.step()
        encoder_optim.step()
        # domain_classifier_optim.step()   # 域判别器
        domain_classifier_2_optim.step()
        cross_transformer_optim.step()
        # lfcn_optim.step()

        # 准确率统计
        total_hit_src += torch.sum(torch.argmax(logits_src, dim=1).cpu() == query_label_src).item()
        total_num_src += query_src.shape[0]
        acc_src = total_hit_src / total_num_src    # 源域准确率

        # 目标域同理...
        total_hit_tar += torch.sum(torch.argmax(logits_tar, dim=1).cpu() == query_label_tar).item()
        total_num_tar += query_tar.shape[0]
        acc_tar = total_hit_tar / total_num_tar

        if (episode + 1) % 100 == 0:        # 定期日志记录
            logger.info(
                'episode: {:>3d}, f_loss: {:6.4f}, loss_wd: {:6.4f}, domain_loss_2: {:6.4f}, CTx_loss_tar: {:6.4f}, loss: {:6.4f}, acc_src: {:6.4f}, acc_tar: {:6.4f}'.format(
                    episode + 1,
                    f_loss.item(),
                    # loss_coral.item(),   loss_coral: {:6.4f},
                    loss_wd.item(),
                    # domain_loss.item(),domain_loss: {:6.4f},
                    loss_disc.item(),
                    # scl_loss_tar.item(), scl_loss_tar: {:6.4f},
                    CTx_loss_tar.item(),
                    # loss_lmmd.item(), loss_lmmd: {:6.4f}，
                    loss.item(),
                    acc_src,
                    acc_tar))

            writer.add_scalar('Loss/f_loss', f_loss.item(), episode + 1)

            # writer.add_scalar('Loss/loss_coral', loss_coral.item(), episode + 1)
            writer.add_scalar('Loss/loss_wd', loss_wd.item(), episode + 1)
            # writer.add_scalar('Loss/domain_loss', domain_loss.item(), episode + 1)
            writer.add_scalar('Loss/domain_loss_2', loss_disc.item(), episode + 1)
            # writer.add_scalar('Loss/text_align_loss', text_align_loss.item(), episode + 1)
            # writer.add_scalar('Loss/scl_loss_tar', scl_loss_tar.item(), episode + 1)
            writer.add_scalar('Loss/CTx_loss_tar', CTx_loss_tar.item(), episode + 1)
            writer.add_scalar('Loss/loss', loss.item(), episode + 1)

            writer.add_scalar('Acc/acc_src', acc_src, episode + 1)
            writer.add_scalar('Acc/acc_tar', acc_tar, episode + 1)

        if (episode + 1) % 500 == 0 or episode == 0:     # 定期测试评估
            with torch.no_grad():
                # 测试
                logger.info("Testing ...")
                train_end = time.time()
                # 切换到评估模式
                mapping_tar.eval()
                encoder.eval()
                total_rewards = 0         # 累计正确预测数
                counter = 0               # 累计测试样本数
                accuracies = []
                predict = np.array([], dtype=np.int64)
                predict_gnn = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)

                full_predict = np.array([], dtype=np.int64)

                train_datas, train_labels = train_loader.__iter__().next()

                support_real_labels = train_labels

                train_features, _ = encoder(mapping_tar(Variable(train_datas).to(GPU)))           # 提取支持集特征

                # 特征归一化处理
                max_value = train_features.max()
                min_value = train_features.min()
                print(max_value.item())
                print(min_value.item())
                train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

                #  KNN分类器构建
                KNN_classifier = KNeighborsClassifier(n_neighbors=1)
                KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_features, _ = encoder(mapping_tar((Variable(test_datas).to(GPU))))   # 提取测试特征
                    test_features = (test_features - min_value) * 1.0 / (max_value - min_value)     # 归一化
                    predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())   # KNN预测
                    test_labels = test_labels.numpy()
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter
                    accuracies.append(accuracy)

                # if (episode + 1) == 5000:
                #     # 生成全图预测结果
                #     for all_datas, all_labels in all_test_loader:
                #         all_data_features, _ = encoder(mapping_tar((Variable(all_datas).to(GPU))))  # 提取测试特征
                #         all_data_features = (all_data_features - min_value) * 1.0 / (max_value - min_value)  # 归一化
                #         predict_full_labels = KNN_classifier.predict(
                #             all_data_features.cpu().detach().numpy())  # KNN预测
                #
                #         full_predict = np.append(full_predict, predict_full_labels)
                #         best_predict_full = full_predict
                #         best_full_G, best_Row_all, best_Column_all, best_nTrain_all = G_pad, Row_all, Col_all, nTrain_all
                #
                #     #################classification map all pexil################################
                #     orig_nRow, orig_nCol, _ = Data_Band_Scaler.shape  # 原始图尺寸
                #     half = patch_size // 2
                #     # 确保 best_predict_full 长度正确
                #     assert len(best_predict_full) == orig_nRow * orig_nCol, (
                #         f"预测长度 {len(best_predict_full)} != {orig_nRow * orig_nCol}"
                #     )
                #
                #     # Row_all, Column_all: length = nRow*nCol
                #     for idx, pred in enumerate(best_predict_full):
                #         r = best_Row_all[idx]
                #         c = best_Column_all[idx]
                #         best_full_G[r, c] = pred + 1  # +1 将标签从 1…C，0 用作背景
                #
                #     print(best_predict_full.shape)
                #     print(best_predict_full)
                #
                #     # 生成分类地图
                #     hsi_pic_all = np.zeros((best_full_G.shape[0], best_full_G.shape[1], 3))
                #     for i in range(best_full_G.shape[0]):
                #         for j in range(best_full_G.shape[1]):
                #             if best_full_G[i][j] == 0:
                #                 hsi_pic_all[i, j, :] = [0, 0, 0]
                #             if best_full_G[i][j] == 1:
                #                 hsi_pic_all[i, j, :] = [0, 0, 1]
                #             if best_full_G[i][j] == 2:
                #                 hsi_pic_all[i, j, :] = [0, 1, 0]
                #             if best_full_G[i][j] == 3:
                #                 hsi_pic_all[i, j, :] = [0, 1, 1]
                #             if best_full_G[i][j] == 4:
                #                 hsi_pic_all[i, j, :] = [1, 0, 0]
                #             if best_full_G[i][j] == 5:
                #                 hsi_pic_all[i, j, :] = [1, 0, 1]
                #             if best_full_G[i][j] == 6:
                #                 hsi_pic_all[i, j, :] = [1, 1, 0]
                #             if best_full_G[i][j] == 7:
                #                 hsi_pic_all[i, j, :] = [0.5, 0.5, 1]
                #             if best_full_G[i][j] == 8:
                #                 hsi_pic_all[i, j, :] = [0.65, 0.35, 1]
                #             if best_full_G[i][j] == 9:
                #                 hsi_pic_all[i, j, :] = [0.75, 0.5, 0.75]
                #             if best_full_G[i][j] == 10:
                #                 hsi_pic_all[i, j, :] = [0.75, 1, 0.5]
                #             if best_full_G[i][j] == 11:
                #                 hsi_pic_all[i, j, :] = [0.5, 1, 0.65]
                #             if best_full_G[i][j] == 12:
                #                 hsi_pic_all[i, j, :] = [0.65, 0.65, 0]
                #             if best_full_G[i][j] == 13:
                #                 hsi_pic_all[i, j, :] = [0.75, 1, 0.65]
                #             if best_full_G[i][j] == 14:
                #                 hsi_pic_all[i, j, :] = [0, 0, 0.5]
                #             if best_full_G[i][j] == 15:
                #                 hsi_pic_all[i, j, :] = [0, 1, 0.75]
                #             if best_full_G[i][j] == 16:
                #                 hsi_pic_all[i, j, :] = [0.5, 0.75, 1]
                #
                #     best_full_G = best_full_G[
                #                   half: half + orig_nRow,
                #                   half: half + orig_nCol
                #                   ]  #
                #     halfwidth = patch_size // 2
                #     utils.classification_map(hsi_pic_all,
                #                              best_full_G, 24,
                #                              "classificationMap_full/IP_{}_{}shot.png".format(TAR_LSAMPLE_NUM_PER_CLASS, iDataSet))

                test_accuracy = 100. * total_rewards / len(test_loader.dataset)
                writer.add_scalar('Acc/acc_test', test_accuracy, episode + 1)

                logger.info('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                                     100. * total_rewards / len(test_loader.dataset)))
                test_end = time.time()

                mapping_tar.train()
                encoder.train()
                if test_accuracy > last_accuracy:
                    last_accuracy = test_accuracy
                    best_episode = episode
                    acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                    OA = acc
                    C = metrics.confusion_matrix(labels, predict)
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
                    best_predict_all = predict
                    best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
                    # 计算F1-score (weighted average)
                    f1_scores[iDataSet] = metrics.f1_score(labels, predict, average='weighted')

                    # 保存最佳模型参数（用于后续可视化）
                    best_mapping_tar_state = mapping_tar.state_dict()
                    best_encoder_state = encoder.state_dict()
                    
                    # 保存模型权重到文件
                    # model_save_dir = "model_weights"
                    # os.makedirs(model_save_dir, exist_ok=True)
                    # model_save_path = os.path.join(model_save_dir, f"IP_dataset{iDataSet+1}_{TAR_LSAMPLE_NUM_PER_CLASS}shot.pth")
                    # torch.save({
                    #     'mapping_tar': mapping_tar.state_dict(),
                    #     'encoder': encoder.state_dict(),
                    #     'accuracy': test_accuracy,
                    #     'episode': episode,
                    #     'config': {
                    #         'TAR_INPUT_DIMENSION': TAR_INPUT_DIMENSION,
                    #         'N_DIMENSION': N_DIMENSION,
                    #         'emb_size': emb_size,
                    #         'TAR_CLASS_NUM': TAR_CLASS_NUM,
                    #         'patch_size': patch_size
                    #     }
                    # }, model_save_path)
                    # logger.info(f"Model saved to {model_save_path} with accuracy {test_accuracy:.2f}%")

                logger.info('best episode:[{}], best accuracy={}'.format(best_episode + 1, last_accuracy))

    # # ==================== t-SNE可视化 ====================
    # logger.info("Generating t-SNE visualization...")
    # logger.info(f"Using best model from episode {best_episode + 1} with accuracy {last_accuracy:.2f}%")
    # with torch.no_grad():
    #     # 加载最佳模型参数
    #     mapping_tar.load_state_dict(best_mapping_tar_state)
    #     encoder.load_state_dict(best_encoder_state)
    #     mapping_tar.eval()
    #     encoder.eval()

    #     # 1. 提取目标域训练集和测试集特征
    #     train_datas, train_labels = train_loader.__iter__().next()
    #     train_features_tsne, _ = encoder(mapping_tar(Variable(train_datas).to(GPU)))
    #     train_features_tsne = train_features_tsne.cpu().numpy()
    #     train_labels_np = train_labels.numpy()

    #     # 提取测试集特征
    #     test_features_list = []
    #     test_labels_list = []
    #     for test_datas, test_labels in test_loader:
    #         test_features, _ = encoder(mapping_tar(Variable(test_datas).to(GPU)))
    #         test_features_list.append(test_features.cpu().numpy())
    #         test_labels_list.append(test_labels.numpy())

    #     test_features_tsne = np.concatenate(test_features_list, axis=0)
    #     test_labels_np = np.concatenate(test_labels_list, axis=0)

    #     # 合并训练和测试特征
    #     all_features = np.concatenate([train_features_tsne, test_features_tsne], axis=0)
    #     all_labels = np.concatenate([train_labels_np, test_labels_np], axis=0)

    #     # t-SNE降维
    #     tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    #     features_2d = tsne.fit_transform(all_features)

    #     # 分离训练和测试的2D坐标
    #     train_2d = features_2d[:len(train_labels_np)]
    #     test_2d = features_2d[len(train_labels_np):]

    #     # 绘制t-SNE图（论文级别高质量）
    #     plt.figure(figsize=(14, 11))
    #     colors = plt.cm.tab20(np.linspace(0, 1, TAR_CLASS_NUM))

    #     # 绘制测试集（用圆圈，边框很细或无边框）
    #     for i in range(TAR_CLASS_NUM):
    #         mask = test_labels_np == i
    #         plt.scatter(test_2d[mask, 0], test_2d[mask, 1],
    #                     c=[colors[i]], label=f'Class {i}',
    #                     alpha=0.7, s=60, marker='o', edgecolors='none')

    #     # # 绘制训练集（用星号标记，已注释）
    #     # for i in range(TAR_CLASS_NUM):
    #     #     mask = train_labels_np == i
    #     #     plt.scatter(train_2d[mask, 0], train_2d[mask, 1],
    #     #                c=[colors[i]], s=400, marker='*',
    #     #                edgecolors='black', linewidths=0.8)

    #     plt.title(f't-SNE Visualization (Dataset {iDataSet + 1}, Accuracy: {last_accuracy:.2f}%)',
    #               fontsize=20, fontweight='bold', pad=20)
    #     plt.xlabel('t-SNE Dimension 1', fontsize=18, fontweight='bold')
    #     plt.ylabel('t-SNE Dimension 2', fontsize=18, fontweight='bold')
    #     # NOTE: matplotlib 的 Legend 不支持 `linewidth` 参数；需要在创建 legend 后再设置边框线宽
    #     legend = plt.legend(
    #         bbox_to_anchor=(1.05, 1),
    #         loc='upper left',
    #         fontsize=12,
    #         frameon=True,
    #         fancybox=True,
    #         shadow=True,
    #         framealpha=0.9,
    #     )
    #     try:
    #         legend.get_frame().set_edgecolor('black')
    #         legend.get_frame().set_linewidth(2)
    #     except Exception:
    #         # 兼容极老版本/特殊 backend：即使设置失败也不影响保存图像
    #         pass
    #     plt.grid(True, alpha=0.4, linewidth=1.2)
    #     plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)

    #     # 加粗坐标轴边框
    #     ax = plt.gca()
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(2)

    #     plt.tight_layout()

    #     # 保存t-SNE图（高分辨率）
    #     tsne_save_path = f"visualization/tsne_IP_{TAR_LSAMPLE_NUM_PER_CLASS}shot_dataset{iDataSet + 1}.png"
    #     os.makedirs(os.path.dirname(tsne_save_path), exist_ok=True)
    #     plt.savefig(tsne_save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    #     plt.close()
    #     logger.info(f"t-SNE visualization saved to {tsne_save_path}")

    # # ==================== 全像素分类图生成 ====================
    # logger.info("Generating full-pixel classification map...")
    # logger.info(f"Using best model from episode {best_episode + 1} with accuracy {last_accuracy:.2f}%")
    # with torch.no_grad():
    #     # 加载最佳模型参数
    #     mapping_tar.load_state_dict(best_mapping_tar_state)
    #     encoder.load_state_dict(best_encoder_state)
    #     mapping_tar.eval()
    #     encoder.eval()

    #     # 提取训练集特征用于KNN分类器
    #     train_datas, train_labels = train_loader.__iter__().next()
    #     train_features, _ = encoder(mapping_tar(Variable(train_datas).to(GPU)))
        
    #     # 特征归一化
    #     max_value = train_features.max()
    #     min_value = train_features.min()
    #     train_features = (train_features - min_value) * 1.0 / (max_value - min_value)
        
    #     # 构建KNN分类器
    #     KNN_classifier = KNeighborsClassifier(n_neighbors=1)
    #     KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels.numpy())
        
    #     # 对所有像素进行预测
    #     full_predict = np.array([], dtype=np.int64)
    #     logger.info("Predicting all pixels...")
    #     for all_datas, all_labels in all_test_loader:
    #         all_data_features, _ = encoder(mapping_tar(Variable(all_datas).to(GPU)))
    #         all_data_features = (all_data_features - min_value) * 1.0 / (max_value - min_value)
    #         predict_full_labels = KNN_classifier.predict(all_data_features.cpu().detach().numpy())
    #         full_predict = np.append(full_predict, predict_full_labels)
        
    #     logger.info(f"Total pixels predicted: {len(full_predict)}")
        
    #     # 生成全图分类结果
    #     orig_nRow, orig_nCol, _ = Data_Band_Scaler.shape
    #     halfwidth = patch_size // 2
        
    #     # 确保预测长度正确
    #     assert len(full_predict) == orig_nRow * orig_nCol, (
    #         f"预测长度 {len(full_predict)} != {orig_nRow * orig_nCol}"
    #     )
        
    #     # 创建分类图矩阵
    #     full_classification_map = np.copy(G_pad)
        
    #     # 将预测结果填充到分类图中
    #     for idx, pred in enumerate(full_predict):
    #         r = Row_all[idx]
    #         c = Col_all[idx]
    #         full_classification_map[r, c] = pred + 1  # +1 将标签从 1...C，0 用作背景
        
    #     # 裁剪掉padding部分
    #     full_classification_map = full_classification_map[
    #         halfwidth: halfwidth + orig_nRow,
    #         halfwidth: halfwidth + orig_nCol
    #     ]
        
    #     # 生成彩色分类地图
    #     hsi_pic_full = np.zeros((full_classification_map.shape[0], full_classification_map.shape[1], 3))
        
    #     # 定义颜色映射（根据IP数据集的类别数调整）
    #     color_map = {
    #         0: [0, 0, 0],           # 背景-黑色
    #         1: [0, 0, 1],           # Class 1-蓝色
    #         2: [0, 1, 0],           # Class 2-绿色
    #         3: [0, 1, 1],           # Class 3-青色
    #         4: [1, 0, 0],           # Class 4-红色
    #         5: [1, 0, 1],           # Class 5-品红
    #         6: [1, 1, 0],           # Class 6-黄色
    #         7: [0.5, 0.5, 1],       # Class 7-浅蓝
    #         8: [0.65, 0.35, 1],     # Class 8-紫色
    #         9: [0.75, 0.5, 0.75],   # Class 9-粉紫
    #         10: [0.75, 1, 0.5],     # Class 10-浅绿
    #         11: [0.5, 1, 0.65],     # Class 11-青绿
    #         12: [0.65, 0.65, 0],    # Class 12-橄榄绿
    #         13: [0.75, 1, 0.65],    # Class 13-浅黄绿
    #         14: [0, 0, 0.5],        # Class 14-深蓝
    #         15: [0, 1, 0.75],       # Class 15-浅青
    #         16: [0.5, 0.75, 1]      # Class 16-天蓝
    #     }
        
    #     for i in range(full_classification_map.shape[0]):
    #         for j in range(full_classification_map.shape[1]):
    #             label = int(full_classification_map[i][j])
    #             if label in color_map:
    #                 hsi_pic_full[i, j, :] = color_map[label]
        
    #     # 保存分类图
    #     classification_map_path = f"classificationMap/IP_{TAR_LSAMPLE_NUM_PER_CLASS}shot_dataset{iDataSet + 1}_full.png"
    #     os.makedirs(os.path.dirname(classification_map_path), exist_ok=True)
        
    #     # 使用高质量保存
    #     plt.figure(figsize=(12, 10))
    #     plt.imshow(hsi_pic_full)
    #     plt.title(f'Full Pixel Classification Map\n(Dataset {iDataSet + 1}, {TAR_LSAMPLE_NUM_PER_CLASS}-shot, Accuracy: {last_accuracy:.2f}%)',
    #               fontsize=16, fontweight='bold', pad=15)
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(classification_map_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    #     plt.close()
        
    #     logger.info(f"Full-pixel classification map saved to {classification_map_path}")

    # # ==================== 误差热力图可视化 ====================
    # logger.info("Generating error heatmap...")
    # # 创建误差矩阵（与原始图像尺寸相同）
    # error_map = np.zeros((best_G.shape[0], best_G.shape[1]))

    # # 标记预测错误的位置
    # for i in range(len(best_predict_all)):
    #     row_idx = best_Row[best_RandPerm[best_nTrain + i]]
    #     col_idx = best_Column[best_RandPerm[best_nTrain + i]]
    #     predicted_label = best_predict_all[i]
    #     true_label = labels[i]

    #     # 如果预测错误，标记为1（错误），否则为0（正确）
    #     if predicted_label != true_label:
    #         error_map[row_idx, col_idx] = 1
    #     else:
    #         error_map[row_idx, col_idx] = 0

    # # 绘制误差热力图（论文级别高质量）
    # plt.figure(figsize=(18, 15))

    # # 子图1：误差分布热力图
    # plt.subplot(2, 2, 1)
    # halfwidth = patch_size // 2
    # error_map_crop = error_map[halfwidth:-halfwidth, halfwidth:-halfwidth]
    # im1 = plt.imshow(error_map_crop, cmap='RdYlGn_r', interpolation='nearest', vmin=0, vmax=1)
    # plt.title(f'Error Distribution Map\n(Red: Wrong, Green: Correct)',
    #           fontsize=16, fontweight='bold', pad=15)
    # cbar1 = plt.colorbar(im1, label='Error (1=Wrong, 0=Correct)', fraction=0.046, pad=0.04)
    # cbar1.set_label('Error (1=Wrong, 0=Correct)', fontsize=14, fontweight='bold')
    # cbar1.ax.tick_params(labelsize=12, width=2, length=6)
    # cbar1.outline.set_linewidth(2)
    # plt.axis('off')

    # # 子图2：真实标签图
    # plt.subplot(2, 2, 2)
    # gt_map = best_G[halfwidth:-halfwidth, halfwidth:-halfwidth]
    # im2 = plt.imshow(gt_map, cmap='tab20', interpolation='nearest')
    # plt.title('Ground Truth Labels', fontsize=16, fontweight='bold', pad=15)
    # cbar2 = plt.colorbar(im2, label='Class ID', fraction=0.046, pad=0.04)
    # cbar2.set_label('Class ID', fontsize=14, fontweight='bold')
    # cbar2.ax.tick_params(labelsize=12, width=2, length=6)
    # cbar2.outline.set_linewidth(2)
    # plt.axis('off')

    # # 子图3：预测标签图
    # plt.subplot(2, 2, 3)
    # pred_map = np.copy(best_G[halfwidth:-halfwidth, halfwidth:-halfwidth])
    # for i in range(len(best_predict_all)):
    #     row_idx = best_Row[best_RandPerm[best_nTrain + i]]
    #     col_idx = best_Column[best_RandPerm[best_nTrain + i]]
    #     pred_map[row_idx - halfwidth, col_idx - halfwidth] = best_predict_all[i] + 1
    # im3 = plt.imshow(pred_map, cmap='tab20', interpolation='nearest')
    # plt.title('Predicted Labels', fontsize=16, fontweight='bold', pad=15)
    # cbar3 = plt.colorbar(im3, label='Class ID', fraction=0.046, pad=0.04)
    # cbar3.set_label('Class ID', fontsize=14, fontweight='bold')
    # cbar3.ax.tick_params(labelsize=12, width=2, length=6)
    # cbar3.outline.set_linewidth(2)
    # plt.axis('off')

    # # 子图4：错误率统计柱状图（按类别）
    # plt.subplot(2, 2, 4)
    # class_errors = np.zeros(TAR_CLASS_NUM)
    # class_totals = np.zeros(TAR_CLASS_NUM)
    # for i in range(len(labels)):
    #     true_label = labels[i]
    #     class_totals[true_label] += 1
    #     if best_predict_all[i] != true_label:
    #         class_errors[true_label] += 1

    # error_rates = (class_errors / (class_totals + 1e-10)) * 100
    # bars = plt.bar(range(TAR_CLASS_NUM), error_rates, color='coral',
    #                edgecolor='black', linewidth=2.5, alpha=0.85)
    # plt.xlabel('Class ID', fontsize=16, fontweight='bold')
    # plt.ylabel('Error Rate (%)', fontsize=16, fontweight='bold')
    # plt.title('Per-Class Error Rate', fontsize=16, fontweight='bold', pad=15)
    # plt.xticks(range(TAR_CLASS_NUM), fontsize=13, fontweight='bold')
    # plt.yticks(fontsize=13, fontweight='bold')
    # plt.grid(True, alpha=0.4, axis='y', linewidth=1.5)

    # # 加粗坐标轴边框
    # ax4 = plt.gca()
    # for spine in ax4.spines.values():
    #     spine.set_linewidth(2.5)
    # ax4.tick_params(axis='both', which='major', width=2.5, length=6)

    # # 在柱状图上标注数值
    # for i, (bar, rate) in enumerate(zip(bars, error_rates)):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height,
    #              f'{rate:.1f}%',
    #              ha='center', va='bottom', fontsize=11, fontweight='bold')

    # plt.suptitle(f'Error Analysis (Dataset {iDataSet + 1}, OA: {last_accuracy:.2f}%)',
    #              fontsize=20, fontweight='bold', y=0.98)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])

    # # 保存误差热力图（高分辨率）
    # heatmap_save_path = f"visualization/error_heatmap_IP_{TAR_LSAMPLE_NUM_PER_CLASS}shot_dataset{iDataSet + 1}.png"
    # os.makedirs(os.path.dirname(heatmap_save_path), exist_ok=True)
    # plt.savefig(heatmap_save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    # plt.close()
    # logger.info(f"Error heatmap saved to {heatmap_save_path}")

    logger.info('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episode + 1, last_accuracy))
    logger.info("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
    logger.info("accuracy list: {}".format(acc))
    logger.info('***********************************************************************************')

OAMean = np.mean(acc)
OAStd = np.std(acc)

AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

kMean = np.mean(k)
kStd = np.std(k)

f1Mean = np.mean(f1_scores)
f1Std = np.std(f1_scores)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

logger.info("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
logger.info("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
logger.info("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
logger.info("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
logger.info("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
logger.info("average F1-score: " + "{:.4f}".format(100 * f1Mean) + " +- " + "{:.4f}".format(100 * f1Std))
logger.info("accuracy list: {}".format(acc))
logger.info("accuracy for each class: ")
for i in range(TAR_CLASS_NUM):
    logger.info("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

# # ==================== 生成综合统计图 ====================
# logger.info("Generating summary visualization...")

# # 设置全局字体（使用Linux常见字体）
# plt.rcParams['font.family'] = 'DejaVu Sans'  # Linux系统自带字体
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'

# # 创建综合统计图（论文级别高质量）
# fig = plt.figure(figsize=(20, 12))

# # 子图1：每次实验的准确率折线图
# ax1 = plt.subplot(2, 3, 1)
# ax1.plot(range(1, nDataSet + 1), acc, marker='o', linewidth=3, markersize=12,
#         color='#2E86AB', markeredgecolor='black', markeredgewidth=2)
# ax1.axhline(y=OAMean, color='r', linestyle='--', linewidth=3, label=f'Mean: {OAMean:.2f}%')
# ax1.fill_between(range(1, nDataSet + 1), OAMean - OAStd, OAMean + OAStd,
#                   alpha=0.25, color='r', label=f'Std: ±{OAStd:.2f}%')
# ax1.set_xlabel('Dataset Run', fontsize=16, fontweight='bold')
# ax1.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
# ax1.set_title('Accuracy Across Runs', fontsize=18, fontweight='bold', pad=15)
# ax1.grid(True, alpha=0.4, linewidth=1.5)
# legend1 = ax1.legend(fontsize=13, frameon=True)
# try:
#     legend1.get_frame().set_edgecolor('black')
#     legend1.get_frame().set_linewidth(2)
# except Exception:
#     pass
# ax1.set_xticks(range(1, nDataSet + 1))
# ax1.tick_params(axis='both', labelsize=14, width=2, length=6)
# for spine in ax1.spines.values():
#     spine.set_linewidth(2)

# # 子图4：混淆矩阵热力图（使用最后一次的结果）
# ax4 = plt.subplot(2, 3, 4)
# C_normalized = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
# im = ax4.imshow(C_normalized, cmap='YlOrRd', aspect='auto', interpolation='nearest')

# # # 在每个格子中添加数值标注
# # for i in range(TAR_CLASS_NUM):
# #     for j in range(TAR_CLASS_NUM):
# #         value = C_normalized[i, j]
# #         # 根据背景颜色选择文字颜色（深色背景用白字，浅色背景用黑字）
# #         text_color = 'white' if value > 0.5 else 'black'
# #         text = ax4.text(j, i, f'{value:.2f}',
# #                        ha="center", va="center", color=text_color,
# #                        fontsize=11, fontweight='bold')

# ax4.set_xlabel('Predicted Label', fontsize=16, fontweight='bold')
# ax4.set_ylabel('True Label', fontsize=16, fontweight='bold')
# ax4.set_title('Normalized Confusion Matrix (Last Run)', fontsize=18, fontweight='bold',
#              pad=15)
# ax4.set_xticks(np.arange(TAR_CLASS_NUM))
# ax4.set_yticks(np.arange(TAR_CLASS_NUM))
# ax4.set_xticklabels(np.arange(TAR_CLASS_NUM), fontsize=13, fontweight='bold')
# ax4.set_yticklabels(np.arange(TAR_CLASS_NUM), fontsize=13, fontweight='bold')
# ax4.tick_params(axis='both', labelsize=13, width=2, length=6)

# # 美化colorbar
# cbar4 = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
# cbar4.set_label('Proportion', fontsize=14, fontweight='bold')
# cbar4.ax.tick_params(labelsize=12, width=2, length=6)
# cbar4.outline.set_linewidth(2)

# # 加粗边框
# for spine in ax4.spines.values():
#     spine.set_linewidth(2)

# # 子图5：AA vs OA 散点图
# ax5 = plt.subplot(2, 3, 5)
# scatter = ax5.scatter(acc, 100 * AA, s=250, c=range(nDataSet), cmap='viridis',
#            edgecolors='black', linewidth=2.5, alpha=0.85)
# ax5.plot([acc.min(), acc.max()], [acc.min(), acc.max()],
#         'r--', linewidth=3, alpha=0.6, label='OA=AA line')
# ax5.set_xlabel('Overall Accuracy (OA) %', fontsize=16, fontweight='bold')
# ax5.set_ylabel('Average Accuracy (AA) %', fontsize=16, fontweight='bold')
# ax5.set_title('OA vs AA Comparison', fontsize=18, fontweight='bold', pad=15)
# ax5.grid(True, alpha=0.4, linewidth=1.5)
# legend5 = ax5.legend(fontsize=13, frameon=True)
# try:
#     legend5.get_frame().set_edgecolor('black')
#     legend5.get_frame().set_linewidth(2)
# except Exception:
#     pass
# ax5.tick_params(axis='both', labelsize=14, width=2, length=6)
# for spine in ax5.spines.values():
#     spine.set_linewidth(2)
# # 标注点
# for i, (oa, aa) in enumerate(zip(acc, 100*AA)):
#     ax5.annotate(f'{i+1}', (oa, aa), fontsize=12, ha='center', va='center',
#                 fontweight='bold')

# # 添加总标题
# plt.suptitle(f'Experimental Results - IP ({TAR_LSAMPLE_NUM_PER_CLASS}-Shot Learning)',
#             fontsize=22, fontweight='bold', y=0.98)
# plt.tight_layout(rect=[0, 0, 1, 0.96])

# # 保存综合统计图（高分辨率）
# summary_save_path = f"visualization/summary_IP_{TAR_LSAMPLE_NUM_PER_CLASS}shot.png"
# os.makedirs(os.path.dirname(summary_save_path), exist_ok=True)
# plt.savefig(summary_save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
# plt.close()
# logger.info(f"Summary visualization saved to {summary_save_path}")
# logger.info("="*80)
# logger.info("All visualizations generated successfully!")

# # #################classification map################################
# # for i in range(len(best_predict_all)):     # 创建一个大小与 best_G 相同的图像（RGB图像），初始化为零（黑色）
# #     best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[
# #                                                                                                         i] + 1
# # # 生成分类地图
# # hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
# # for i in range(best_G.shape[0]):
# #     for j in range(best_G.shape[1]):
# #         if best_G[i][j] == 0:
# #             hsi_pic[i, j, :] = [0, 0, 0]
# #         if best_G[i][j] == 1:
# #             hsi_pic[i, j, :] = [0, 0, 1]
# #         if best_G[i][j] == 2:
# #             hsi_pic[i, j, :] = [0, 1, 0]
# #         if best_G[i][j] == 3:
# #             hsi_pic[i, j, :] = [0, 1, 1]
# #         if best_G[i][j] == 4:
# #             hsi_pic[i, j, :] = [1, 0, 0]
# #         if best_G[i][j] == 5:
# #             hsi_pic[i, j, :] = [1, 0, 1]
# #         if best_G[i][j] == 6:
# #             hsi_pic[i, j, :] = [1, 1, 0]
# #         if best_G[i][j] == 7:
# #             hsi_pic[i, j, :] = [0.5, 0.5, 1]
# #         if best_G[i][j] == 8:
# #             hsi_pic[i, j, :] = [0.65, 0.35, 1]
# #         if best_G[i][j] == 9:
# #             hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
# #         if best_G[i][j] == 10:
# #             hsi_pic[i, j, :] = [0.75, 1, 0.5]
# #         if best_G[i][j] == 11:
# #             hsi_pic[i, j, :] = [0.5, 1, 0.65]
# #         if best_G[i][j] == 12:
# #             hsi_pic[i, j, :] = [0.65, 0.65, 0]
# #         if best_G[i][j] == 13:
# #             hsi_pic[i, j, :] = [0.75, 1, 0.65]
# #         if best_G[i][j] == 14:
# #             hsi_pic[i, j, :] = [0, 0, 0.5]
# #         if best_G[i][j] == 15:
# #             hsi_pic[i, j, :] = [0, 1, 0.75]
# #         if best_G[i][j] == 16:
# #             hsi_pic[i, j, :] = [0.5, 0.75, 1]
# #
# # halfwidth = patch_size // 2
# # utils.classification_map(hsi_pic[halfwidth:-halfwidth, halfwidth:-halfwidth, :],
# #                          best_G[halfwidth:-halfwidth, halfwidth:-halfwidth], 24,
# #                          "classificationMap/IP_{}shot.png".format(TAR_LSAMPLE_NUM_PER_CLASS))









