# -*- coding:utf-8 -*-
# author: cangshui.lw
# CreateAt: '2021/1/4-11:21 上午'
import torch, os
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertTokenizer, BertModel
from transformers import ViTFeatureExtractor, ViTForImageClassification
import json
from io import BytesIO
import torch.nn.functional as F
import utils
from torch.nn import CrossEntropyLoss, SoftMarginLoss
import numpy as np
from align_model import build_model
from attention import AttentionSequencePoolingLayer

torch.autograd.set_detect_anomaly(True)


class Config(object):
    """配置参数"""

    def __init__(self, dataset, args):
        self.model_name = 'shoucai_siamese_train_concat_poi'
        # self.rel_dict_path = 'NYT10-HRL/data/rel2id.json'
        self.save_path = os.path.join(dataset, 'saved_dict/' + self.model_name + '.ckpt')  # 模型训练结果
        self.save_result_path = dataset + 'saved_result/test_result.json'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备


        self.require_improvement = 100000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = args.epoch  # epoch数
        self.batch_size = args.batch_size  # mini-batch大小 128
        self.bert_max_len = 200  # 每句话处理成的长度(短填长切)
        self.learning_rate = args.lr  # 学习率
        # self.bert_path = '/data/volume1/bert-base-chinese'
        self.bert_path = 'bert-base-chinese'
        # self.vit_path = '/data/volume2/weights_ViT'
        # 挂载路径为:
        # -Dvolumes对应的本地路径为 / data / volume1 - / data / volumeN
        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # self.vit_extractor = ViTFeatureExtractor.from_pretrained(self.vit_path)
        self.hidden_size = 768
        self.num_layers = 2
        self.dropout = 0.1
        self.rank_loss = torch.nn.MarginRankingLoss(margin=args.margin)
        self.log = args.log
        self.softmax = args.softmax
        self.freeze = args.freeze
        dict_load = np.load('temp_dataset_item_pois_relation_1129_poi_2_pic_num.npy', allow_pickle=True)
        self.dd_poi_2_pic_num = dict_load.item()  # POI对应的图片数，用来决定商品中多个POI优先级
        if args.use_norm_vector:
            print('use norm vector.npy')
            dict_load = np.load('temp_dataset_item_pois_relation_0110_imgs_avg10_vector.npy', allow_pickle=True)
        else:
            print('use wo norm vector.npy')
            dict_load = np.load('temp_dataset_item_pois_relation_0110_imgs_avg_10vector_wo_norm.npy', allow_pickle=True)
        self.use_norm = args.use_norm_vector
        self.dd_poi_2_mean_10_img_vectors = dict_load.item()  # POI对应的平均图像向量 512维度

        dict_load = np.load('temp_dataset_item_pois_relation_0110_poi_name_vector.npy', allow_pickle=True)
        self.dd_poi_name_2_vectors = dict_load.item()  # POI name 对应的文本向量 512维度

        # if not os.path.exists('align_model.pth'):
        #     utils.download_align_model_weights(args.align_pth, 'align_model.pth')
        # 初始化 align model
        config_align = {
            "model_path": "align_model.pth",
            "bert_path": self.bert_path}
        # key_word = '普陀山#船'
        # device = "cuda:0" if torch.cuda.is_available() else torch.device('cpu')
        opt = utils.Dict2Obj(config_align)
        self.align_model = build_model(opt).to(self.device)
        if self.freeze:
            self.align_model.eval()
            for param in self.align_model.parameters():
                param.requires_grad = False
        # self.align_model.eval()
        # for param in self.align_model.parameters():
        #     param.requires_grad = False

    def get_rank_loss(self):
        return self.rank_loss


class Multimodal_Model(nn.Module):
    def __init__(self, config):
        super(Multimodal_Model, self).__init__()
        # self.bert = BertModel.from_pretrained(config.bert_path)
        # feature_extractor = ViTFeatureExtractor.from_pretrained('weights_ViT')
        # self.vit_model = ViTForImageClassification.from_pretrained(config.vit_path)
        # self._num_labels = num_labels
        self.freeze = config.freeze
        # if self.freeze:
        #     for param in self.bert.parameters():
        #         param.requires_grad = False
        #     for param in self.vit_model.parameters():
        #         param.requires_grad = False
        # else:
        #     for param in self.bert.parameters():
        #         param.requires_grad = True
        #     for param in self.vit_model.parameters():
        #         param.requires_grad = True
        self.use_norm = config.use_norm
        self.linear1 = nn.Linear(1024, 512)
        # self.linear1_2 = nn.Linear(config.hidden_size * 2, 256)
        self.linear3 = nn.Linear(512 * 3, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = config.softmax
        self.softmax_fn = torch.nn.Softmax(dim=0)
        self.align_model = config.align_model
        self.attention_layer = AttentionSequencePoolingLayer(embedding_dim=512)

        # if self.freeze:
        #     for param in self.align_model.parameters():
        #         param.requires_grad = False

    # b_preds_1 = model(img_1, title_feature, pois_feature, mean_poi_img_vec1, mean_poi_img_vec2)
    def forward(self, img_1, pois_feature, mean_poi_img_vec, poi_text_vec, poi_len):
        """

        :param img_1:  不变
        :param title_feature: 不需要
        :param pois_feature: 不变
        :param mean_poi_img_vec1-5: b * 5 * 512
        :param poi_text_vec1-5: b * 5 * 512
        :param pois_input: b * 5 * 1024 (太长了要先降维 再送入Attention？）
        :param poi_len: b * 1 代表该商品有多少有效poi 1-5
        :return:
        """
        img_vec = self.align_model.encode_image(img_1, norm=self.use_norm)
        pois_input = torch.cat([mean_poi_img_vec, poi_text_vec], dim=-1)
        # poi 序列降维

        pois_input = self.linear1(pois_input)
        pois_input = self.dropout(pois_input)
        # poi_mean_img = F.relu(poi_mean_img)
        pois_input = self.attention_layer(img_vec.unsqueeze(1), pois_input, poi_len)  # b * 1 * 512
        pois_input = pois_input.squeeze(1)  # b * 512, 若 无POI则该vec全0

        multi_vec_1 = torch.cat([img_vec, pois_feature, pois_input], 1)
        # multi_vec_2 = torch.cat([pooled_poi, vit_vec_2], 1)

        # multi_vec_1 = self.dropout(multi_vec_1)
        # multi_vec_1 = self.linear1(multi_vec_1)
        # multi_vec_2 = self.linear1(multi_vec_2)

        # merged = torch.cat([multi_vec_1, multi_vec_2], 1)

        # multi_vec_1 = self.dropout(multi_vec_1)
        # multi_vec_1 = F.relu(multi_vec_1)
        score_1 = self.linear3(multi_vec_1)
        score_1 = self.dropout(score_1)
        score_1 = F.relu(score_1)

        score_1 = self.linear4(score_1)
        score_1 = self.dropout(score_1)
        score_1 = F.relu(score_1)

        score_1 = self.linear2(score_1)
        # score_1 = self.sigmoid(score_1)

        # multi_vec_2 = self.dropout(multi_vec_2)
        # multi_vec_2 = self.linear1(multi_vec_2)
        # multi_vec_2 = F.relu(multi_vec_2)
        # score_2 = self.linear3(multi_vec_2)
        # score_2= F.relu(score_2)
        # score_2 = self.linear2(score_2)
        # score_2 = self.sigmoid(score_2)

        # outputs = (logits,)
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self._num_labels), labels.view(-1))
        #     # loss = categorical_crossentropy_with_prior(logits, labels)
        #     return (loss,) + outputs
        # print('score_1.szie', score_1.size())
        # if self.softmax:
        #     score_1, score_2 = self.softmax_fn(torch.stack([score_1, score_2], dim=0))

        return score_1


if __name__ == '__main__':
    # 测试用
    a = torch.randn((8, 5, 1024))
    print(a.size())
    print(torch.cat([a, a], dim=-1).size())
    # linear1 = nn.Linear(1024, 512)
    # b = linear1(a)
    # print(b.size())
