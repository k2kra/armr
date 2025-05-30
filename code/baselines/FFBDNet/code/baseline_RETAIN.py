import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict
from models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params


torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'Retain'
resume_name = ''
if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))
        
data_path = '../data/records_final.pkl'
voc_path = '../data/voc_final.pkl'

ehr_adj_path = '../data/ehr_adj_final.pkl'
ddi_adj_path = '../data/ddi_A_final.pkl'
device = torch.device('cuda:0')

ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
data = dill.load(open(data_path, 'rb'))
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]


def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)


        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f, AVG_MED: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))
    dill.dump(obj=smm_record, file=open('../data/gamenet_records.pkl', 'wb'))
    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


EPOCH = 40
LR = 0.0002
Neg_Loss = True
DDI_IN_MEM = True
TARGET_DDI = 0.05
T = 0.5
decay_weight = 0.85
voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))


model = Retain(voc_size, device=device)
best_model = Retain(voc_size, device=device)
model.to(device=device)
best_model.to(device=device)
print('parameters', get_n_params(model))
optimizer = Adam(list(model.parameters()), lr=LR)


history = defaultdict(list)
best_epoch = 0
best_ja = 0
for epoch in range(EPOCH):
    loss_record = []
    start_time = time.time()
    model.train()
    for step, input in enumerate(data_train):
        if len(input) < 2:
            continue
            
        loss = 0
        for i in range(1, len(input)):
            target = np.zeros((1, voc_size[2]))
            target[:, input[i][2]] = 1

            output_logits = model(input[:i])
            loss += F.binary_cross_entropy_with_logits(output_logits, torch.FloatTensor(target).to(device))
            loss_record.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))
        
    # annealing
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

    history['ja'].append(ja)
    history['ddi_rate'].append(ddi_rate)
    history['avg_p'].append(avg_p)
    history['avg_r'].append(avg_r)
    history['avg_f1'].append(avg_f1)
    history['prauc'].append(prauc)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                        np.mean(loss_record),
                                                                                        elapsed_time,
                                                                                        elapsed_time * (
                                                                                                    EPOCH - epoch - 1)/60))

    torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
    print('')
    if epoch != 0 and best_ja < ja:
        best_model.load_state_dict(model.state_dict())
        best_epoch = epoch
        best_ja = ja


dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

# test
torch.save(model.state_dict(), open(
    os.path.join('saved', model_name, 'final.model'), 'wb'))

print('best_epoch:', best_epoch)


eval(best_model, data_test, voc_size, 0)