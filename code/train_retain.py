#%%
import torch
import argparse
import numpy as np
from torch.optim import Adam
import time
import torch.nn.functional as F
from net_retain import Retain
from utils_ import multi_label_metric, ddi_rate_score, llprint,get_data,create_log_file,log
from collections import defaultdict
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=3, help='3(mimic-iii) or 4(mimic-iv)')
parser.add_argument('--seed', type=int, default=1029, help='seed')
parser.add_argument('--epoch', type=int, default=50, help='# of epoches')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
# args = parser.parse_args(args=[])
args = parser.parse_args()
log_file = create_log_file('RETAIN')
log(log_file, 'config: ' + str(args) + '\n')
log(log_file, time.strftime("Date: %Y%m%d-%H%M%S"))
log(log_file, "File: {}".format(__file__) + '\n')
log(log_file, 'ja\tprauc\tavg_p\tavg_r\tavg_f1\tddi\tavg_med')
smm_record=[]
med_records=[]
def eval(model, data_eval, voc_size, ddi_A):
    model.eval()

    # smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        if len(input) < 2:
            continue
        for i in range(1, len(input)):
            target_output = model(input[:i])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prob
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.4] = 1
            y_pred_tmp[y_pred_tmp < 0.4] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(y_pred_label_tmp)
            med_cnt += len(y_pred_label_tmp)
            visit_cnt += 1

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    for i in smm_record:
        for j in i:
            med_records.append(j)
    ddi_rate = ddi_rate_score(med_records, ddi_A)
    metric = (ddi_rate,np.mean(ja),np.mean(prauc),np.mean(avg_p),np.mean(avg_r),np.mean(avg_f1),med_cnt / visit_cnt,)
    llprint(
        "\nDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f, AVG_MED: %.4f\n"%metric
    )
    log(log_file, '%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f' % (
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        ddi_rate,
        med_cnt / visit_cnt,
    ))
    return metric

torch.manual_seed(args.seed)
np.random.seed(args.seed)
#%%
start_time=time.time()

dataset=''
if args.dataset == 3:
    dataset = 'mimic-iii'
elif args.dataset == 4:
    dataset = 'mimic-iv'

data,voc,ddi_A=get_data('../data/{}/'.format(dataset))
size_diag_voc, size_pro_voc, size_med_voc = len(voc['diag_voc']['idx2word']), len(voc['pro_voc']['idx2word']), len(voc['med_voc']['idx2word'])
#%%

np.random.seed(args.seed)
np.random.shuffle(data)
split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_eval = data[split_point + eval_len :]
data_test = data[split_point : split_point + eval_len]
voc_size = (size_diag_voc, size_pro_voc, size_med_voc)

model=Retain(voc_size=voc_size,device=torch.device('cpu'))
optimizer = Adam(model.parameters(), args.lr)

history = defaultdict(list)
best_epoch, best_ja = 0, 0
#%%
for epoch in range(args.epoch):
    tic = time.time()
    print("\nepoch {} --------------------------".format(epoch + 1))
    model.train()
    for step, input in enumerate(data_train):
        if len(input) < 2: continue

        loss = torch.zeros(1)
        for i in range(1, len(input)):
            target = np.zeros((1, voc_size[2]))
            target[:, input[i][2]] = 1

            output_logits = model(input[:i])
            loss += F.binary_cross_entropy_with_logits(output_logits, torch.FloatTensor(target))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        llprint('\rtraining step: {} / {}'.format(step, len(data_train)))
    
    print()
    tic2 = time.time()
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
        model, data_eval, voc_size, ddi_A
    )
    print(
        "training time: {}, test time: {}".format(
            time.time() - tic, time.time() - tic2
        )
    )
    history["ja"].append(ja)
    history["ddi_rate"].append(ddi_rate)
    history["avg_p"].append(avg_p)
    history["avg_r"].append(avg_r)
    history["avg_f1"].append(avg_f1)
    history["prauc"].append(prauc)
    history["med"].append(avg_med)

    if epoch >= 5:
        print(
            "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                np.mean(history["ddi_rate"][-5:]),
                np.mean(history["med"][-5:]),
                np.mean(history["ja"][-5:]),
                np.mean(history["avg_f1"][-5:]),
                np.mean(history["prauc"][-5:]),
            )
        )

    if epoch != 0 and best_ja < ja:
        best_epoch = epoch
        best_ja = ja

    print("best_epoch: {}".format(best_epoch))
# %%
# %%
log(log_file, 'best: ')
log(log_file, '%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f' % (
    history['ja'][best_epoch],
    history['prauc'][best_epoch],
    history['avg_p'][best_epoch],
    history['avg_r'][best_epoch],
    history['avg_f1'][best_epoch],
    history['ddi_rate'][best_epoch],
    history['med'][best_epoch],
))
log(log_file, "Time used: %.2f" % (time.time()-start_time))
log_file.close()
