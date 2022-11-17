import time
import datetime
import io, torch, hashlib
from io import BytesIO
import numpy as np
from PIL import Image
from torch.nn import CrossEntropyLoss
from datetime import timedelta

try:
    import common_io
except Exception as e:
    print(e)
    pass




def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# def download_align_model_weights(key, local_file):
#     bucket.get_object_to_file(key, local_file)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


## 写入odps
# def write_odps_table(table, records):
#     writer = common_io.table.TableWriter(table)
#     print('recodes_len:', len(records))
#     print("records:", records[0])
#     writer.write(records, list(range(len(records[0]))))
#     writer.close()


# def save_model(model, path, ak, aks, endpoint, bucket):
#     print('******************* saving model *******************')
#     bucket_name = bucket
#     auth = oss2.Auth(ak, aks)
#     bucket = oss2.Bucket(auth, endpoint, bucket_name)
#     buffer = BytesIO()
#     torch.save(model.state_dict(), buffer)
#     print('save model in:', path)
#     bucket.put_object(path, buffer.getvalue())


# def load_model(model, path, ak, aks, endpoint, bucket):
#     print('******************* loading model *******************')
#     bucket_name = bucket
#     auth = oss2.Auth(ak, aks)
#     bucket = oss2.Bucket(auth, endpoint, bucket_name)
#     buffer = BytesIO(bucket.get_object(path).read())
#     model.load_state_dict(torch.load(buffer))


def genearteMD5(ss):
    # 创建md5对象
    hl = hashlib.md5()

    # Tips
    # 此处必须声明encode
    # 否则报错为：hl.update(str)    Unicode-objects must be encoded before hashing
    hl.update(ss.encode(encoding='utf-8'))

    # print('MD5加密前为 ：' + ss)
    # print('MD5加密后为 ：' + hl.hexdigest())
    return hl.hexdigest()


def GetRunTime(func):
    def call_func(*args, **kwargs):
        # Dome的起始时间
        begin_time = time.time()
        # 调用Dome
        ret = func(*args, **kwargs)
        # Dome的运行时间
        run_time = time.time() - begin_time
        # file_run_name = inspect.stack()[1].filename
        # 将Dome的名字以及Dome的运行时间,打印在终端
        print(str(func.__name__) + " running costs: {:2f}s".format(run_time))
        # 返回Dome的引用
        return ret

    # 当函数被装饰的时候, 返回装饰器内闭包函数的引用
    return call_func


def bytes2human(n):
    """
    显示内存占用情况的工具函数
    :param n:
    :return:
    """
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return '%sB' % n

def get_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def percentage(v):
    return "%.2f%%" % (v * 100)

def base64_to_image(base64):
    '''
    读取base64 （bytes或str）  ---》 RGB numpy
    :param base64:
    :return:
    '''
    try:
        img_str = base64.b64decode(base64)
        pil_img = Image.open(BytesIO(img_str)).convert('RGB')
        # pil_img = unify_image_as_rgb(np.asarray(pil_img))
        # image = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("base64 is broken", e)
        return None
    return np.asarray(pil_img)


class Dict2Obj(object):
    def __init__(self, dic):
        self.dic = dic
        for key, val in dic.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Dict2Obj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Dict2Obj(val) if isinstance(val, dict) else val)

    def __getitem__(self, key):
        return self.dic[key]

    def __setitem__(self, key, val):
        self.dic[key] = val
        if isinstance(val, (list, tuple)):
            setattr(self, key, [Dict2Obj(x) if isinstance(x, dict) else x for x in val])
        else:
            setattr(self, key, Dict2Obj(val) if isinstance(val, dict) else val)

    def __len__(self):
        return len(self.dic)