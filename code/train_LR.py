#%%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import time
from utils_ import *
# 屏蔽掉 sci-kit learn 的 warning
import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=3, help='3(mimic-iii) or 4(mimic-iv)')
parser.add_argument('--seed', type=int, default=1029, help='seed')
# args = parser.parse_args(args=[])
args = parser.parse_args()

def create_dataset(data, size_diag_voc, size_pro_voc, size_med_voc):
    i1_len = size_diag_voc
    i2_len = size_pro_voc
    output_len = size_med_voc
    input_len = i1_len + i2_len
    X = []
    y = []
    for patient in data:
        for visit in patient:
            i1 = visit[0]
            i2 = visit[1]
            o = visit[2]

            multi_hot_input = np.zeros(input_len)
            multi_hot_input[i1] = 1
            multi_hot_input[np.array(i2) + i1_len] = 1

            multi_hot_output = np.zeros(output_len)
            multi_hot_output[o] = 1

            X.append(multi_hot_input)
            y.append(multi_hot_output)

    return np.array(X), np.array(y)

#%% 训练模型
start_time=time.time()

dataset=''
if args.dataset == 3:
    dataset = 'mimic-iii'
elif args.dataset == 4:
    dataset = 'mimic-iv'

data,voc,ddi_A=get_data('../data/{}/'.format(dataset))
size_diag_voc, size_pro_voc, size_med_voc = len(voc['diag_voc']['idx2word']), len(voc['pro_voc']['idx2word']), len(voc['med_voc']['idx2word'])

np.random.seed(args.seed)
np.random.shuffle(data)
split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_eval = data[split_point + eval_len :]
data_test = data[split_point : split_point + eval_len]

train_X, train_y = create_dataset(data_train, size_diag_voc, size_pro_voc, size_med_voc)
test_X, test_y = create_dataset(data_test, size_diag_voc, size_pro_voc, size_med_voc)
eval_X, eval_y = create_dataset(data_eval, size_diag_voc, size_pro_voc, size_med_voc)
model = LogisticRegression()
classifier = OneVsRestClassifier(model)

tic = time.time()
print('fitting...')
classifier.fit(train_X, train_y)

fittime = time.time() - tic
print("fitting time: {}".format(fittime))
#%% 评估模型
result = []
for _ in range(5):
    index = np.random.choice(
        np.arange(len(test_X)), round(len(test_X) * 0.8), replace=True
    )
    test_sample = test_X[index]
    y_sample = test_y[index]
    tic = time.time()
    y_pred = classifier.predict(test_sample)
    y_prob = classifier.predict_proba(test_sample)
    print("inference time: {}".format(time.time() - tic))
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_sample, y_pred, y_prob)

    pred_med_codes = []
    med_cnt=0
    for adm in y_pred:
        m = np.where(adm==1)[0]
        pred_med_codes.append(m)
        med_cnt += len(m)
    ddi_rate = ddi_rate_score(pred_med_codes,ddi_A)
    result.append([ja, prauc, avg_p, avg_r, avg_f1,ddi_rate,med_cnt / y_pred.shape[0]])

result = np.array(result)
mean = result.mean(axis=0)
std = result.std(axis=0)

# %%
log_file = create_log_file('LR')
log(log_file, 'config: ' + str(args) + '\n')
log(log_file, time.strftime("Date: %Y%m%d-%H%M%S"))
log(log_file, "File: {}".format(__file__) + '\n')
log(log_file, 'ja\tprauc\tavg_p\tavg_r\tavg_f1\tddi\tavg_med')
log(log_file, 'mean:')
log(log_file, '%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f' % tuple(mean))
log(log_file, 'std:')
log(log_file, '%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % tuple(std))
log(log_file, "Time used: %.2f" % (time.time()-start_time))
log_file.close()

# %%
