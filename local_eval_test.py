# -*- coding:utf-8 -*-
# author: cangshui.lw
# CreateAt: '2021/1/4-11:10 上午'
import argparse, torch, random, time, os, shutil, sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from model_local import *
from datasets_attention_local import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loss import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--tables", default="", type=str, help="odps input table names")
parser.add_argument("--index_files",
                    default="data/dwd_mml_train_dataset_1007_urls_train_with_poi_tuomin.csv,data/dwd_mml_train_dataset_1007_urls_val_with_poi_tuomin.csv,dwd_mml_train_dataset_1007_urls_train_with_poi_tuomin.csv",
                    type=str, help="odps input table names")
parser.add_argument("--outputs", default="", type=str, help="odps outout table names")
parser.add_argument("--task_name", default="shoucai_siamese", type=str, help="output tables")
parser.add_argument("--note_txt", default="", type=str, help="output tables")
parser.add_argument("--save_test", default=0, type=int, help="label index")
parser.add_argument("--batch_size", default=16, type=int, help="label index")
parser.add_argument("--epoch", default=32, type=int, help="label index")
parser.add_argument("--eval_per_step", default=32, type=int, help="eval_per_step")
parser.add_argument("--margin", default=0.5, type=float, help="eval_per_step")
parser.add_argument("--aug", default=1, type=int, help="eval_per_step")
parser.add_argument("--lr", default=1e-3, type=float, help="eval_per_step")
parser.add_argument("--log", default=0, type=int, help="label index")
parser.add_argument("--softmax", default=0, type=int, help="label index")
parser.add_argument("--freeze", default=1, type=int, help="label index")
parser.add_argument("--use_norm_vector", default=1, type=int, help="label index")
parser.add_argument("--align_pth", default="youyi/models/align_相关性模型权重/align_with_mkg_0106/commodity_poi_mml_20211221_with_MKG_withMKG_iteration_200000_0.4215_0.407_0.018.pth", type=str, help="output tables")
args = parser.parse_args()


def test(config, test_dataloader):
    model = Multimodal_Model(config).to(config.device)
    load_model(model, config.save_path, config.ak,
               config.aks, config.endpoint, config.bucket_name)
    # args.learning_rate - default is 5e-5
    # args.adam_epsilon  - default is 1e-8

    model.eval()

    label_list = []
    pred_label_list = []
    preds_list = []
    sentence_list = []
    token_list = []
    id_list = []
    for batch in tqdm(test_dataloader):
        b_token_ids = batch[0].to(config.device)
        b_token_type_ids = batch[1].to(config.device)
        b_attention_mask = batch[2].to(config.device)
        b_labels = batch[3].to(config.device)
        b_id_list = batch[4]

        with torch.no_grad():
            output = model(b_token_ids, b_token_type_ids, b_attention_mask, b_labels)

        b_preds = output[1].detach().cpu()
        b_preds = F.softmax(b_preds).numpy()

        token_list.extend(
            [config.tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in b_token_ids.cpu().numpy()])
        label_list.extend(b_labels.cpu().numpy())
        preds_list.extend([i[1] for i in b_preds])
        pred_label_list.extend(np.argmax(b_preds, axis=1))
        sentence_list.extend([config.tokenizer.decode(i) for i in b_token_ids])
        id_list.extend(b_id_list)

    records = []
    for token, label, pred_label, preds, sentence, id in zip(token_list, label_list, pred_label_list, preds_list,
                                                             sentence_list, id_list):
        records.append([token, label, pred_label, preds, sentence] + id.tolist())
    return records


def init_config(path):
    config = Config(path, args)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    return config


def init_datasets(config):
    print("Loading data...")
    train_file, val_file, test_file = args.index_files.split(',')
    if args.aug == 1:
        train_dataloader = create_dataloader(train_file, config, args.batch_size, mode='train')
    else:
        train_dataloader = create_dataloader(train_file, config, args.batch_size, mode='train_no_aug')
    val_dataloader = create_dataloader(val_file, config, args.batch_size, mode='val')
    # test_dataloader = create_dataloader(test_file, config, args.batch_size, mode='test')
    # train_table, dev_table, test_table = args.tables.split(',')
    # train_dataset, new_line_list = load_table(args, train_table, config)
    # dev_dataset, new_line_list = load_table(args, dev_table, config)
    # test_dataset, new_line_list = load_table(args, test_table, config)
    # train_size = int(0.8 * len(dataset))
    # dev_size = int(0.1 * len(dataset))
    # test_size = len(dataset) - train_size - dev_size
    # train_dataset, dev_dataset, test_dataset = random_split(dataset, [train_size, dev_size, test_size])
    print('train_iters=', len(train_dataloader))
    print('val_iters=', len(val_dataloader))
    # print('test_iters=', len(test_dataloader))
    return train_dataloader, val_dataloader


def eval_test_dataloader(config, model, start_time, epoch, test_dataloader):
    print('>>> Epoch end. test_evaluate_and_save_ckpt...')
    model.eval()
    dev_preds = []
    dev_labels = []
    total_test_loss = 0
    softmax_fn = torch.nn.Softmax(dim=0)
    for batch in test_dataloader:
        # 准备cv数据
        img_1 = batch[0].to(config.device)
        img_2 = batch[1].to(config.device)
        if args.log:
            print('test img_1=', img_1[0][0][0][:10])
            print('test img_2=', img_2[0][0][0][:10])

        # 准备nlp数据
        token_ids = batch[2].to(config.device)
        if args.log:
            print('test  token_ids[0]=', token_ids[0][:15])
        token_type_ids = batch[3].to(config.device)
        attention_mask = batch[4].to(config.device)
        gt_labels = batch[5].to(config.device)
        poi_img = batch[6].to(config.device)
        token_ids2 = batch[7].to(config.device)
        token_type_ids2 = batch[8].to(config.device)
        attention_mask2 = batch[9].to(config.device)

        with torch.no_grad():
            b_preds_1 = model(token_ids, token_type_ids, attention_mask, img_1, poi_img, token_ids2, token_type_ids2, attention_mask2)
            b_preds_2 = model(token_ids, token_type_ids, attention_mask, img_2, poi_img, token_ids2, token_type_ids2, attention_mask2)
            # loss1, gt_labels_fuyi = loss_fn_rank(config, b_preds_1, b_preds_2, gt_labels)
            if args.softmax:
                b_preds_1, b_preds_2 = softmax_fn(torch.stack([b_preds_1, b_preds_2], dim=0))
            loss2 = rank_sigmoid_loss(b_preds_1, b_preds_2, gt_labels)
            loss = loss2
            # print('current loss=', loss.item())
        total_test_loss = total_test_loss + loss.item()

        b_preds_1 = b_preds_1.detach().cpu().numpy().squeeze()
        b_preds_2 = b_preds_2.detach().cpu().numpy().squeeze()
        print('test b_preds_1=', b_preds_1)
        print('test b_preds_2=', b_preds_2)
        gt_labels = gt_labels.detach().cpu().numpy().squeeze().tolist()
        predicts = np.array(b_preds_1 > b_preds_2, dtype=np.int32).tolist()
        dev_preds.extend(predicts)
        dev_labels.extend(gt_labels)

    total_dev_loss = total_test_loss / len(test_dataloader)
    print('test all dev_labels', dev_labels)
    print('test all dev_pre', dev_preds)
    dev_accuracy = accuracy_score(np.array(dev_labels), np.array(dev_preds))

    time_dif = get_time_dif(start_time)
    msg = 'EPOCH: {0:>6},  Test Loss: {1:.4f},  Test accuracy: {2:>6.2%}, Time: {3}'
    print(msg.format(epoch + 1, total_dev_loss, dev_accuracy, time_dif
                     ))
    model.train()

def eval_():
    # 初始化配置
    path = 'youyi/' + args.task_name + '/shoucai_multimodal/' + time.strftime("%Y-%m-%d-%H-%M")
    config = init_config(path)
    print('config 初始化完毕')

    start_time = time.time()

    # 初始化数据集
    train_dataloader, val_dataloader = init_datasets(config)
    print('数据集 初始化完毕')
    # train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=config.batch_size)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size)
    # test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

    # 初始化模型
    model = Multimodal_Model(config).to(config.device)
    print('Multimodal_Model 初始化完毕')
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load('shoucai_siamese_train_concat_poi_020_0.6520.ckpt', map_location='cpu'))
    else:
        model.load_state_dict(torch.load('shoucai_siamese_train_concat_poi_020_0.6520.ckpt'))
    print('Load Multimodal_Model success~')
    # model.eval()

    print('>>> Epoch end. val_evaluate_and_save_ckpt...')
    model.eval()
    dev_preds = []
    dev_labels = []
    total_dev_loss = 0
    softmax_fn = torch.nn.Softmax(dim=0)
    for batch in val_dataloader:
        # 准备cv数据
        img_1 = batch[0].to(config.device)
        img_2 = batch[1].to(config.device)

        # 准备nlp数据
        gt_labels = batch[2].to(config.device)
        # title_feature = batch[3].to(config.device)  # 512
        pois_feature = batch[3].to(config.device)  # 512 若关联不了则=title_feature
        # mean_poi_img_vec, poi_text_vec, poi_len
        mean_poi_img_vec = batch[4].to(config.device)  # 512 已经提前align model算好了
        poi_text_vec = batch[5].to(config.device)
        poi_len = batch[6].to(config.device)
        model.zero_grad()

        # b_preds_1 = model(img_1, pois_feature, mean_poi_img_vec, poi_text_vec, poi_len)
        # b_preds_2 = model(img_2, pois_feature, mean_poi_img_vec, poi_text_vec, poi_len)

        with torch.no_grad():
            start_ = time.time()
            b_preds_1 = model(img_1, pois_feature, mean_poi_img_vec, poi_text_vec, poi_len)
            b_preds_2 = model(img_2, pois_feature, mean_poi_img_vec, poi_text_vec, poi_len)
            print(args.batch_size *2, 'costs', time.time()-start_)
            # loss1, gt_labels_fuyi = loss_fn_rank(config, b_preds_1, b_preds_2, gt_labels)
            if args.softmax:
                b_preds_1, b_preds_2 = softmax_fn(torch.stack([b_preds_1, b_preds_2], dim=0))
            loss2 = rank_sigmoid_loss(b_preds_1, b_preds_2, gt_labels)
            loss = loss2
            # print('current loss=', loss.item())
        total_dev_loss = total_dev_loss + loss.item()

        b_preds_1 = b_preds_1.detach().cpu().numpy().squeeze()
        b_preds_2 = b_preds_2.detach().cpu().numpy().squeeze()
        print('val b_preds_1=', b_preds_1)
        print('val b_preds_2=', b_preds_2)
        gt_labels = gt_labels.detach().cpu().numpy().squeeze().tolist()
        predicts = np.array(b_preds_1 > b_preds_2, dtype=np.int32).tolist()
        dev_preds.extend(predicts)
        dev_labels.extend(gt_labels)

    total_dev_loss = total_dev_loss / len(val_dataloader)
    print('val all dev_labels', dev_labels)
    print('val all dev_pre', dev_preds)
    dev_accuracy = accuracy_score(np.array(dev_labels), np.array(dev_preds))
    # time_dif = get_time_dif(start_time)
    msg = ' Dev Loss: {:.4f},  Dev accuracy: {:.6f}, Time:'
    print(msg.format(total_dev_loss, dev_accuracy))


if __name__ == '__main__':
    # main()
    eval_()
