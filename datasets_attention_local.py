import io
from io import BytesIO
import cv2
# import oss2
import PIL
import numpy as np
from PIL import Image, ImageFile
import torch
import threading, traceback, requests

from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils



def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    # print(len(batch))
    # print(batch[0])
    batch = list(filter(lambda x: x[0] is not None, batch))
    # print(batch)
    if len(batch) == 0:
        print('ERROR: my_collate_fn batch.len=0')
        batch = [(torch.ones((3, 224, 224), dtype=torch.long), torch.ones((3, 224, 224), dtype=torch.long),
                  torch.zeros(200, dtype=torch.long),
                  torch.zeros(200, dtype=torch.long), torch.zeros(200, dtype=torch.long),
                  torch.zeros([1], dtype=torch.float))]
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


def normalize_image(img, size=(224, 224), channel_first=True, bgr2rgb=True, scale_normalize=True):
    img = np.array(img)
    if img is None:
        img = np.zeros((224, 224, 3))
    if size is not None:
        img = cv2.resize(img, size)
    # if bgr2rgb:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if scale_normalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img / 255.0 - mean) / std
    if channel_first:
        img = np.transpose(img, (2, 0, 1))

    return img.astype(np.float32)


def calcu_img_num_weight(ll):
    return ll[0] + ll[1] * 5 + ll[2] * 3


def create_dataloader(index_file, config, batch_size=8, num_workers=0, shuffle=False, mode='train'):
    if mode == 'train':
        preprocess = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.3),
            # transforms.ColorJitter(0.1, 0.1, 0.1, 0.03)
        ])
    else:
        preprocess = None
    dataset = OSSDataset( index_file, config, transform=preprocess, mode=mode)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=my_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )
    return data_loader


class OSSDataset(Dataset):
    def __init__(self, index_file, config, transform=None, mode='train'):
        index_file = index_file.replace('oss://poi-image/', '')
        self._init = False
        self._lock = threading.RLock()
        # self._bucket = oss2.Bucket(auth, endpoint, bucket)
        self._indices = []
        with open(index_file) as f:
            for line in f:
                line = line.strip()
                if len(line) < 2:
                    continue
                self._indices.append(line)

         # self._bucket.get_object(index_file).read().decode().strip().split('$line$')
        self._index_file = index_file
        self.transform = transform
        self._config = config
        self.mode = mode
        print(mode + 'init over... len(self._indices2)=', len(self._indices))

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index):
        line = self._indices[index].strip()
        if 'oss://' in line:
            line = line.replace('oss://poi-image/', '')
        if '$record$' not in line:
            return None, None, None, None, None, None, None, None, None
        try:
            # item_id, online_url, aes_url, base_pv, random_pv, base_ctr, random_ctr, title, cate_name, oss_url, oss_url2, poi_names, poi_img = line.split(
            #     '$record$')
            item_id, online_url, aes_url, base_ctr, random_ctr, title, cate_name, oss_url, oss_url2, poi_names = line.split(
                '$record$')
            # print(item_id)
            # print(label)
            # 返回 img_1,img_2,label, pois_feature, mean_poi_img_vec, poi_text_vec, poi_len
            img_str = requests.get(oss_url).content
            img = Image.open(BytesIO(img_str)).convert('RGB')

            if self.transform:
                img = self.transform(img)
            # img = np.asarray(img).copy()
            img_input1 = normalize_image(img)

            img_str = requests.get(oss_url2).content
            img = Image.open(BytesIO(img_str)).convert('RGB')

            if self.transform:
                img = self.transform(img)
            # img = np.asarray(img).copy()
            img_input2 = normalize_image(img)

            # 处理 poi 对应的 img vector,
            if len(poi_names.strip()) <= 1:
                # mean_poi_img_vec, poi_text_vec, poi_len
                mean_poi_img_vec = torch.zeros((5, 512), dtype=torch.float)
                poi_text_vec = torch.zeros((5, 512), dtype=torch.float)
                poi_len = 0

            else:
                # 图片数量多的 POI优先
                poi_names_list = poi_names.split(',')
                poi_names_list.sort(
                    key=lambda x: -calcu_img_num_weight(self._config.dd_poi_2_pic_num.get(x.strip(), [0, 0, 0])))
                if len(poi_names_list) > 0:
                    poi_len = min(len(poi_names_list), 5)  # 最多5个
                    mean_poi_img_vec = torch.zeros((5, 512), dtype=torch.float)
                    poi_text_vec = torch.zeros((5, 512), dtype=torch.float)
                    for i, poi_name in enumerate(poi_names_list[:5]):
                        img_vec1 = self._config.dd_poi_2_mean_10_img_vectors.get(poi_names_list[i],
                                                                                 torch.zeros(512, dtype=torch.float))
                        mean_poi_img_vec[i] = torch.tensor(img_vec1, dtype=torch.float)
                        txt_vec = self._config.dd_poi_name_2_vectors.get(poi_names_list[i],
                                                                         torch.zeros(512, dtype=torch.float))
                        poi_text_vec[i] = torch.tensor(txt_vec, dtype=torch.float)
                else:
                    mean_poi_img_vec = torch.zeros((5, 512), dtype=torch.float)
                    poi_text_vec = torch.zeros((5, 512), dtype=torch.float)
                    poi_len = 0

            # 处理title
            text_feature = self._config.align_model.tokenizer(title, padding=True,
                                                              truncation=True, return_tensors='pt', max_length=50)
            text_feature = text_feature.to(self._config.device)
            title_feature = self._config.align_model.encode_text(text_feature,
                                                                 norm=self._config.use_norm).detach().cpu().squeeze()

            # 处理poi names  对应的 all POI的name的tokenizer
            if len(poi_names.strip()) < 1:
                text_embeddings = title_feature
            else:
                text_feature = self._config.align_model.tokenizer(poi_names, padding=True,
                                                                  truncation=True, return_tensors='pt', max_length=50)
                text_feature = text_feature.to(self._config.device)
                text_embeddings = self._config.align_model.encode_text(text_feature,
                                                                       norm=self._config.use_norm).detach().cpu().squeeze()
        except Exception as e:
            print('ERROR: __getitem__', e)
            print(line)
            traceback.print_exc()
            return None, None, None, None, None, None, None
        if self._config.log:
            print(self.mode, 'title=', title)
            print(self.mode, 'in oss dataset: img_input1[0][0]', online_url, oss_url)
            print(self.mode, 'in oss dataset: img_input2[0][0]', aes_url, oss_url2)
        if float(base_ctr) > float(random_ctr):
            label = 1.
        else:
            label = 0.
        # img_vec1 = torch.tensor(img_vec1, dtype=torch.float)
        # img_vec2 = torch.tensor(img_vec2, dtype=torch.float)
        return img_input1, img_input2, torch.tensor([float(label)],
                                                    dtype=torch.float), text_embeddings, mean_poi_img_vec, poi_text_vec, torch.tensor(
            [float(poi_len)],
            dtype=torch.float)


if __name__ == '__main__':
    # 本地测试用
    pass
