#%%

from itertools import count
import numpy as np
import argparse
from collections import defaultdict
from torch.optim import RMSprop
import torch
import time
from net_micron import MICRON
from utils_ import get_data,ddi_rate_score,llprint,multi_label_metric,count_parameters,log,create_log_file
import torch.nn.functional as F

# torch.set_num_threads(30)
torch.manual_seed(1203)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate') # original 0.0002
parser.add_argument('--weight_decay', type=float, default=1e-5, help='learning rate')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--dataset', type=int, default=3, help='3(mimic-iii) or 4(mimic-iv)')
args = parser.parse_args()

log_file = create_log_file('MICRON')
log(log_file, 'config: ' + str(args) + '\n')
log(log_file, time.strftime("Date: %Y%m%d-%H%M%S"))
log(log_file, "File: {}".format(__file__) + '\n')
log(log_file, 'ja\tprauc\tavg_p\tavg_r\tavg_f1\tddi\tavg_med')

def ddi(smm_record, ddi_A):
    med_records=[]
    for i in smm_record:
        for j in i:
            med_records.append(j)
    return ddi_rate_score(med_records, ddi_A)

# evaluate
def eval(model, data_eval, voc_size, ddi_A, threshold1=0.8, threshold2=0.2):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0
    label_list, prob_list = [], []
    add_list, delete_list = [], []
    # 不同visit的指标统计
    ja_by_visit = [[] for _ in range(5)]
    prauc_by_visit = [[] for _ in range(5)]
    pre_by_visit = [[] for _ in range(5)]
    recall_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]
    smm_record_by_visit = [[] for _ in range(5)]

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        add_temp_list, delete_temp_list = [], []
        if len(input) < 2: continue
        for adm_idx, adm in enumerate(input):
            # 第0个visit也要添加到结果中去
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)
            label_list.append(y_gt_tmp)

            if adm_idx == 0:
                representation_base, _, _, _, _ = model(input[:adm_idx+1])
                # 第0个visit也添加
                y_pred_tmp = F.sigmoid(representation_base).detach().cpu().numpy()[0]
                y_pred_prob.append(y_pred_tmp)
                prob_list.append(y_pred_tmp)

                y_old = np.zeros(voc_size[2])
                y_old[y_pred_tmp>=threshold1] = 1
                y_old[y_pred_tmp<threshold2] = 0
                y_pred.append(y_old)
                # prediction label
                y_pred_label_tmp = np.where(y_old == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))
                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)
                
                single_ja, single_prauc, single_p, single_r, single_f1 = multi_label_metric(np.array([y_gt_tmp]), np.array([y_old]), np.array([y_pred_tmp]))
                ja_by_visit[0].append(single_ja)
                prauc_by_visit[0].append(single_prauc)
                pre_by_visit[0].append(single_p)
                recall_by_visit[0].append(single_r)
                f1_by_visit[0].append(single_f1)
                smm_record_by_visit[0].append([sorted(y_pred_label_tmp)])

                y_old = np.zeros(voc_size[2])
                y_old[adm[2]] = 1
            else:
                _, _, residual, _, _ = model(input[:adm_idx+1])
                # prediction prod
                representation_base += residual
                y_pred_tmp = F.sigmoid(representation_base).detach().cpu().numpy()[0]
                y_pred_prob.append(y_pred_tmp)
                prob_list.append(y_pred_tmp)

                previous_set = np.where(y_old == 1)[0]
                
                # prediction med set
                # result = F.sigmoid(result).detach().cpu().numpy()[0]
                # y_pred_tmp = result.copy()
                y_old[y_pred_tmp>=threshold1] = 1
                y_old[y_pred_tmp<threshold2] = 0
                y_pred.append(y_old)

                # prediction label
                y_pred_label_tmp = np.where(y_old == 1)[0]
                y_pred_label.append(sorted(y_pred_label_tmp))
                visit_cnt += 1
                med_cnt += len(y_pred_label_tmp)

                #### add or delete
                add_gt = set(np.where(y_gt_tmp == 1)[0]) - set(previous_set)
                delete_gt = set(previous_set) - set(np.where(y_gt_tmp == 1)[0])

                add_pre = set(np.where(y_old == 1)[0]) - set(previous_set)
                delete_pre = set(previous_set) - set(np.where(y_old == 1)[0])

                add_distance = len(set(add_pre) - set(add_gt)) + len(set(add_gt) - set(add_pre))
                delete_distance = len(set(delete_pre) - set(delete_gt)) + len(set(delete_gt) - set(delete_pre))
                ####

                add_temp_list.append(add_distance)
                delete_temp_list.append(delete_distance)

                if adm_idx<5:
                    single_ja, single_prauc, single_p, single_r, single_f1 = multi_label_metric(np.array([y_gt_tmp]), np.array([y_old]), np.array([y_pred_tmp]))
                    ja_by_visit[adm_idx].append(single_ja)
                    prauc_by_visit[adm_idx].append(single_prauc)
                    pre_by_visit[adm_idx].append(single_p)
                    recall_by_visit[adm_idx].append(single_r)
                    f1_by_visit[adm_idx].append(single_f1)
                    smm_record_by_visit[adm_idx].append([sorted(y_pred_label_tmp)])

        if len(add_temp_list) > 1:
            add_list.append(np.mean(add_temp_list))
            delete_list.append(np.mean(delete_temp_list))
        elif len(add_temp_list) == 1:
            add_list.append(add_temp_list[0])
            delete_list.append(delete_temp_list[0])

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # 分析各个visit的结果
    print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    print('count:', [len(buf) for buf in ja_by_visit])
    print('prauc:', [np.mean(buf) for buf in prauc_by_visit])
    print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
    print('precision:', [np.mean(buf) for buf in pre_by_visit])
    print('recall:', [np.mean(buf) for buf in recall_by_visit])
    print('f1:', [np.mean(buf) for buf in f1_by_visit])
    print('DDI:', [ddi(buf,ddi_A) for buf in smm_record_by_visit])

    # ddi rate

    ddi_rate = ddi(smm_record, ddi_A)

    llprint('\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4},  AVG_F1: {:.4}, Add: {:.4}, Delete: {:.4}, AVG_MED: {:.4}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_f1), np.mean(add_list), np.mean(delete_list), med_cnt / visit_cnt
    ))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), np.mean(add_list), np.mean(delete_list), med_cnt / visit_cnt


start_time=time.time()

dataset=''
if args.dataset == 3:
    dataset = 'mimic-iii'
elif args.dataset == 4:
    dataset = 'mimic-iv'

data,voc,ddi_A=get_data('../data/{}/'.format(dataset))
size_diag_voc, size_pro_voc, size_med_voc = len(voc['diag_voc']['idx2word']), len(voc['pro_voc']['idx2word']), len(voc['med_voc']['idx2word'])

device = torch.device('cuda:0')

# np.random.seed(1203)
# np.random.shuffle(data)

# "添加第一个visit"
# new_data = []
# for patient in data:
#     patient.insert(0, [[],[],[]])
#     # patient.insert(0, patient[0])
#     new_data.append(patient)
# data = new_data

split_point = int(len(data) * 2 / 3)
data_train = data[:split_point]
eval_len = int(len(data[split_point:]) / 2)
data_test = data[split_point:split_point + eval_len]
data_eval = data[split_point+eval_len:]

voc_size = (size_diag_voc, size_pro_voc, size_med_voc)

model = MICRON(voc_size, ddi_A, emb_dim=args.dim, device=device)
# model.load_state_dict(torch.load(open(args.resume_path, 'rb')))

print('parameters', count_parameters(model))
# exit()
optimizer = RMSprop(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

# start iterations
history = defaultdict(list)
best_epoch, best_ja = 0, 0

weight_list = [[0.25, 0.25, 0.25, 0.25]]
#%%
EPOCH = 40
for epoch in range(EPOCH):
    t = 0
    tic = time.time()
    print ('\nepoch {} --------------------------'.format(epoch + 1))
    
    sample_counter = 0
    mean_loss = np.array([0, 0, 0, 0])

    model.train()
    for step, input in enumerate(data_train):
        loss = torch.zeros(1).to(device)
        if len(input) < 2: continue
        for adm_idx, adm in enumerate(input):
            """第一个visit也参与训练"""
            # if adm_idx == 0: continue     
            # sample_counter += 1
            seq_input = input[:adm_idx+1]

            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1

            loss_bce_target_last = np.zeros((1, voc_size[2]))
            if adm_idx > 0:
                loss_bce_target_last[:, input[adm_idx-1][2]] = 1

            loss_multi_target = np.full((1, voc_size[2]), -1)
            for idx, item in enumerate(adm[2]):
                loss_multi_target[0][idx] = item

            loss_multi_target_last = np.full((1, voc_size[2]), -1)
            if adm_idx > 0:
                for idx, item in enumerate(input[adm_idx-1][2]):
                    loss_multi_target_last[0][idx] = item

            result, result_last, _, loss_ddi, loss_rec = model(seq_input)

            loss_bce = 0.75 * F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device)) + \
                (1 - 0.75) * F.binary_cross_entropy_with_logits(result_last, torch.FloatTensor(loss_bce_target_last).to(device))
            loss_multi = 5e-2 * (0.75 * F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device)) + \
                (1 - 0.75) * F.multilabel_margin_loss(F.sigmoid(result_last), torch.LongTensor(loss_multi_target_last).to(device)))

            y_pred_tmp = F.sigmoid(result).detach().cpu().numpy()[0]
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_label = np.where(y_pred_tmp == 1)[0]
            current_ddi_rate = ddi([[y_label]], ddi_A)
            
            # l2 = 0
            # for p in model.parameters():
            #     l2 = l2 + (p ** 2).sum()
            
            if sample_counter == 0:
                lambda1, lambda2, lambda3, lambda4 = weight_list[-1]
            else:
                current_loss = np.array([loss_bce.detach().cpu().numpy(), loss_multi.detach().cpu().numpy(), loss_ddi.detach().cpu().numpy(), loss_rec.detach().cpu().numpy()])
                current_ratio = (current_loss - np.array(mean_loss)) / np.array(mean_loss)
                instant_weight = np.exp(current_ratio) / sum(np.exp(current_ratio))
                lambda1, lambda2, lambda3, lambda4 = instant_weight * 0.75 + np.array(weight_list[-1]) * 0.25
                # update weight_list
                weight_list.append([lambda1, lambda2, lambda3, lambda4])
            # update mean_loss
            mean_loss = (mean_loss * (sample_counter - 1) + np.array([loss_bce.detach().cpu().numpy(), \
                loss_multi.detach().cpu().numpy(), loss_ddi.detach().cpu().numpy(), loss_rec.detach().cpu().numpy()])) / sample_counter
            # lambda1, lambda2, lambda3, lambda4 = weight_list[-1]
            if current_ddi_rate > 0.08:
                loss += lambda1 * loss_bce + lambda2 * loss_multi + \
                                lambda3 * loss_ddi +  lambda4 * loss_rec
            else:
                loss += lambda1 * loss_bce + lambda2 * loss_multi + \
                            lambda4 * loss_rec

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        llprint('\rtraining step: {} / {}'.format(step, len(data_train)))
    
    print()
    tic2 = time.time() 
    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, add, delete, avg_med = eval(model, data_eval, voc_size, ddi_A)
    print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

    log(log_file, '%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f' % (
        ja,
        prauc,
        avg_p,
        avg_r,
        avg_f1,
        ddi_rate,
        avg_med,
    ))

    history['ja'].append(ja)
    history['ddi_rate'].append(ddi_rate)
    history['avg_p'].append(avg_p)
    history['avg_r'].append(avg_r)
    history['avg_f1'].append(avg_f1)
    history['prauc'].append(prauc)
    history['add'].append(add)
    history['delete'].append(delete)
    history['med'].append(avg_med)

    if epoch >= 5:
        print ('ddi: {}, Med: {}, Ja: {}, F1: {}, Add: {}, Delete: {}'.format(
            np.mean(history['ddi_rate'][-5:]),
            np.mean(history['med'][-5:]),
            np.mean(history['ja'][-5:]),
            np.mean(history['avg_f1'][-5:]),
            np.mean(history['add'][-5:]),
            np.mean(history['delete'][-5:])
            ))

    if epoch != 0 and best_ja < ja:
        best_epoch = epoch
        best_ja = ja

    print ('best_epoch: {}'.format(best_epoch))


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
