# AutoImageSelection
# 1. Introduction
Codes and weights of recall and ranking models for paper《What Image do You Need? A Two-stage Framework for Image Selection in E-commerce》. 
> This is the main project, and there is an extra visualization project of showing our recall ability using html where is at `https://github.com/youyuge34/AVAS` .

# 2. Requirments
please refer to the `requirments.txt`

# 3. Datasets


## Text-to-image_Recall_Dataset
- Test dataset with 500 popular and different items title and related 5k images (ten for each item). For the security of enterprise data, we desensitize the title into POI name, which we used the same dataset in the experimental results of the paper.

Its location is `data/temp_recall_biztype0_fc5_imagern_500.csv`  and the csv header is `poi_id,poi_name,image_url,rank_number,width,height,extend`. You should download the image_url into a dir by yourself using a script.

- Train dataset contains 1M text-image pair and is still under Enterprise Security Audit and will open source soon, which will not interfere with the replication of experimental results.


## Image_ranking_Dataset_on_Fliggy

- Training and testing datasets are `data/dwd_mml_train_dataset_1007_urls_train_with_poi_tuomin.csv` and `dwd_mml_train_dataset_1007_urls_val_with_poi_tuomin.csv`. We desensitize the CTR value and only preserve relative ordering which will not interfere with the replication of experimental results. 

The csv header is :
```
item_id, url_a, url_b, ctr_a, ctr_b, title, cate_name, oss_url, oss_url2, poi_names = line.split('$record$')
```

which means given one same item in online A/B testing, image with `url_a` and image with `url_b` of this item are shown under two traffic buckets separately within a same lifetime. Their CTRs are `ctr_a` and `ctr_b`. Refer to [paper](https://arxiv.org/pdf/2102.04033.pdf)  to learn more the delivery strategy. `oss_url` is the backup url of `url_a` due to that the `url_a` could be 404 after a long time but `oss_url` will not. So you should use `oss_url` in reproduction. 

# 4. Model weights 

## Model_weights_of_our_two-stage_model

- `Recall model weights`: download and rename `align_model.pth` into the root dir.
Download url : [recall model](https://github.com/youyuge34/AVAS/releases/download/v0.1/commodity_poi_mml_20211221_with_MKG_withMKG_iteration_200000_0.4215_0.407_0.018.pth)


- `Ranking model weights`: download `shoucai_siamese_train_concat_poi_020_0.6520.ckpt` and put under root dir without rename.
Download url: [ranking model](https://github.com/youyuge34/AVAS/releases/download/v0.2/shoucai_siamese_train_concat_poi_020_0.6520.ckpt)

## Model_weights_of_compared_baslines

- We finetuned [Taiyi-Clip](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese) under our datasets. and the finetuned Bert weight file is at [here](https://github.com/youyuge34/AVAS/releases/download/v0.3/pytorch_model.bin). You should replace the original `pytorch_model.bin` with this `.bin`.

- We finetuned [M6](https://arxiv.org/abs/2103.00823) on our datasets. However, since m6 only provides interface testing within Alibaba Group. If you need it, you can contact us through the email address of the paper.  

# 5. Paper experiment reproduction
### Recall model reproduction
1. download the `align_model.pth` and [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main) from hugging-face website, put them under root dir.
2. (optional) change the config dict into your weights dir in `align_recall_test_with_mmkg_model.py`
3. download recall test images from csv file into a dir `temp_recall_biztype0_fc5_imagern_500`
4. run the script `python align_recall_test_with_mmkg_model.py` to generate a result csv which outputs `data/align_recall@k_with_MKG_20221021_poi_500.csv` as default.
5. run the script `python align_recall@K.py` to evaluate the recall@5.
6. Get the recall@5 metric of 0.788 as shown in paper. 


### Ranking model reproduction
1. download the `align_model.pth` and [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main) from hugging-face website, put them under root dir.
2. download the `shoucai_siamese_train_concat_poi_020_0.6520.ckpt` and put it under the root dir.
3. run the script `python local_eval_test.py` to get the AUC score of 0.652 as shown in paper.



