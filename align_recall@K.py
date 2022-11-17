import json, os, sys, random, time, requests
from io import BytesIO
import numpy as np
import requests
from PIL import Image
import torch
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'}
import base64 as b64


def recall_eval():
    records = []
    dd_poi = {}
    with open(file) as f:
        for line in f:
            line = line.strip()
            if len(line) < 3:
                continue
            poi_id, name, image_names, score = line.split(',')
            img_id = os.path.split(image_names)[1].split('_')[0]

            records.append([poi_id, name, img_id, float(score)])
            if poi_id in dd_poi:
                dd_poi[poi_id].append([name, img_id, float(score)])
            else:
                dd_poi[poi_id] = [[name, img_id, float(score)]]
    correct_num = 0
    for poi in dd_poi.keys():

        rr = dd_poi[poi]
        rr.sort(key=lambda x: -x[-1])
        rr = rr[:recall_k]
        for name, img_id, score in rr:
            if img_id == poi:
                # print('有个对的：', name, poi, score)
                correct_num += 1
                break
    print('final recall k= {:.3f}'.format(correct_num / len(dd_poi.keys())))

if __name__ == '__main__':
    file = 'data/align_recall@k_with_MKG_20221021_poi_500.csv'
    # file = 'align_recall@k_taiyi_20221021_poi_500_epoch1.csv'
    recall_k = 5
    recall_eval()
    # recall_eval_m6()

    # ndcg_k()
    # ndcg_K2()
    # l1 = [4, 2, 1, 0, 0, 6, 0, 0, 1, 2]
    # l2 = list(range(10, 0, -1))
    # a = getDCG(np.array(l1) / 10)
    # b = getDCG(np.array(l2))
    # print(a, b, a / b)
