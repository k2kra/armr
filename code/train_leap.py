#%%
import torch
import argparse
import numpy as np
import time
from torch.optim import Adam
import torch.nn.functional as F
from collections import defaultdict
from net_leap import Leap
from utils_leap import (
    sequence_metric,
    sequence_output_process,
)
from utils_ import get_data, ddi_rate_score,llprint, count_parameters,log,create_log_file

torch.manual_seed(1203)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=3, help='3(mimic-iii) or 4(mimic-iv)')
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
# args = parser.parse_args(args=[])
args = parser.parse_args()

log_file = create_log_file('LEAP')
log(log_file, 'config: ' + str(args) + '\n')
log(log_file, time.strftime("Date: %Y%m%d-%H%M%S"))
log(log_file, "File: {}".format(__file__) + '\n')
log(log_file, 'ja\tprauc\tavg_p\tavg_r\tavg_f1\tddi\tavg_med')

# evaluate
def eval(model, data_eval, voc_size,ddi_A):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []

        for adm_index, adm in enumerate(input):
            output_logits = model(adm)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            output_logits = output_logits.detach().cpu().numpy()

            # prediction med set
            out_list, sorted_predict = sequence_output_process(
                output_logits, [voc_size[2], voc_size[2] + 1]
            )
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(
            np.array(y_gt),
            np.array(y_pred),
            np.array(y_pred_prob),
            [np.array(p) for p in y_pred_label],
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
        med_records=[]
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


start_time=time.time()
dataset=''
if args.dataset == 3:
    dataset = 'mimic-iii'
elif args.dataset == 4:
    dataset = 'mimic-iv'
data,voc,ddi_A=get_data('../data/{}/'.format(dataset))
size_diag_voc, size_pro_voc, size_med_voc = len(voc['diag_voc']['idx2word']), len(voc['pro_voc']['idx2word']), len(voc['med_voc']['idx2word'])

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point : split_point + eval_len]
data_eval = data[split_point + eval_len :]
voc_size = (size_diag_voc, size_pro_voc, size_med_voc)

END_TOKEN = voc_size[2] + 1

model = Leap(voc_size)

print("parameters", count_parameters(model))
optimizer = Adam(model.parameters(), lr=args.lr)

history = defaultdict(list)
best_epoch, best_ja = 0, 0

EPOCH = 5
#%%
for epoch in range(EPOCH):
    tic = time.time()
    print("\nepoch {} --------------------------".format(epoch + 1))

    model.train()
    for step, input in enumerate(data_train):
        for adm in input:

            loss_target = adm[2] + [END_TOKEN]
            output_logits = model(adm)
            loss = F.cross_entropy(
                output_logits, torch.LongTensor(loss_target)
            )
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        llprint("\rtraining step: {} / {}".format(step, len(data_train)))

    print()
    tic2 = time.time()
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(
        model, data_eval, voc_size,ddi_A)
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
# %%