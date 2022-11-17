import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from efficientnet_pytorch import EfficientNet
from transformers import BertModel, BertTokenizer, BertConfig

logger = logging.getLogger(__name__)

class ALIGN(nn.Module):
    @classmethod
    def from_pretrained(cls, config):
        weight_path = config['model_path']
        if torch.cuda.is_available():
            pretrained_model = torch.load(weight_path)
        else:
            pretrained_model = torch.load(weight_path, map_location="cpu")

        # 不再使用初始化模型
        config['init_bert_weights'] = False
        config['efficientnet_weights_path'] = None

        model = ALIGN.from_config(config)
        model.load_state_dict(pretrained_model['state_dict'], strict=False)
        print('model.load_state_dict success~')

        return model

    @classmethod # 类方法（不需要实例化类就可以被类本身调用）
    def from_config(cls, conf): # cls : 表示没用被实例化的类本身
        import copy
        cv_type = getattr(conf, "efficientnet_type", "efficientnet-b3")
        bert_layers_num = getattr(conf, "bert_layers_num", 4)
        cv_weights = getattr(conf, "efficientnet_weights_path", None)
        bert_weights = getattr(conf, "bert_path", None)
        init_bert_weights = getattr(conf, "init_bert_weights", True)
        hidden_dim = getattr(conf, "hidden_dim", 512)

        print(cv_type, bert_layers_num, cv_weights, bert_weights, init_bert_weights, hidden_dim)
        # efficientnet-b3 4 None ./bert False clip 512 0.3 1.0
        model = ALIGN(cv_type, bert_layers_num, cv_weights, bert_weights, init_bert_weights, hidden_dim)

        return model

    def __init__(
        self, 
        efficientnet_type='efficientnet-b3',
        bert_layers_num=4,
        image_weights_path=None,
        bert_path=None,
        init_bert_weights=False,
        feature_dim=512
    ):
        super(ALIGN, self).__init__()
        # 初始化 efficient net
        
        if image_weights_path is None:
            self.image_model = EfficientNet.from_name(efficientnet_type, num_classes=feature_dim)
        else:
            self.image_model = EfficientNet.from_pretrained(efficientnet_type, weights_path=image_weights_path, 
                num_classes=feature_dim)
        image_feature_dim = feature_dim

        # 初始化bert模型
        # bert-base-chinese 默认配置
        config = BertConfig.from_pretrained(bert_path+"/config.json")
        config.num_hidden_layers = bert_layers_num
        self.tokenizer = BertTokenizer.from_pretrained(bert_path+"/vocab.txt")
        if not init_bert_weights:
            self.text_model = BertModel(config=config)
        else:
            self.text_model = BertModel.from_pretrained(bert_path+"/pytorch_model.bin", config=config)

        text_feature_dim = config.hidden_size

        # 非线性层
        self.logit_scale = nn.Parameter(torch.tensor(1.))

        self.image_hidden_layer = nn.Linear(in_features=image_feature_dim,
            out_features=feature_dim)
        self.text_hidden_layer = nn.Linear(in_features=text_feature_dim,
            out_features=feature_dim)

    def get_tokenizer(self):
        return self.tokenizer

    def encode_image(self, image, norm=True):
        # image_batch = image.shape[0]
        image_embeddings = self.image_model(image)
        image_embeddings = self.image_hidden_layer(image_embeddings)
        if norm:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        return image_embeddings

    def encode_text(self, text, norm=True):
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        # text_batch = input_ids.shape[0]
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = outputs.last_hidden_state[:, 0]
        text_embeddings = self.text_hidden_layer(text_embeddings)
        if norm:
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        return text_embeddings

    def forward(self, image, text):
        batch_size = image.shape[0]
        image_embeddings = self.encode_image(image)
        text_embeddings = self.encode_text(text)
        # temp = self.logit_scale.exp()
        sims = image_embeddings @ text_embeddings.t() * self.logit_scale
        labels = torch.arange(batch_size, dtype=torch.long, device=image_embeddings.device)
        loss = (F.cross_entropy(sims, labels) + F.cross_entropy(sims.t(), labels)) / 2

        return loss


def build_model(config):
    if hasattr(config, 'model_path'):
        model = ALIGN.from_pretrained(config)
    else:
        model = ALIGN.from_config(config)

    return model

