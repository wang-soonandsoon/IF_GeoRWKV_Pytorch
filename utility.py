import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def createBerlinReport(net, data, device, forward_logits_fn=None):
    berlin_class_names = ['Forest', 'Residential Area', 'Industrial Area', 'Low Plants', 'Soil', 'Allotment', 'Commercial Area', 'Water']

    print("Berlin Start!")
    return createReport(net, data, berlin_class_names, device, forward_logits_fn=forward_logits_fn)
    

def createReport(net, data, class_names, device, forward_logits_fn=None):
    global cate
    device_t = device if isinstance(device, torch.device) else torch.device(device)
    net.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for batch in data:
            if forward_logits_fn is None:
                hsi, x, hsi_pca, test_labels, h, w = batch
                hsi_pca = hsi_pca.to(device_t)
                x = x.to(device_t)
                _ , outputs = net(hsi_pca.unsqueeze(1), x)
            else:
                outputs = forward_logits_fn(net, batch, device_t)
                test_labels = batch[3]
            y_pred_list.append(torch.argmax(outputs, dim=1).detach().cpu().numpy())
            y_true_list.append(test_labels.detach().cpu().numpy())

    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    classification = classification_report(
        y_true, y_pred, target_names=class_names, digits=4)
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    oa = accuracy_score(y_true, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_true, y_pred)

    classification = str(classification)
    confusion = str(confusion)
    oa = oa * 100
    each_acc = each_acc * 100
    aa = aa * 100
    kappa = kappa * 100

    logging.info(f'\n{classification}')
    logging.info(f'Overall accuracy (%) {oa}')
    logging.info(f'Average accuracy (%) {aa}')
    logging.info(f'Kappa accuracy (%){kappa}')
    logging.info(f'\n{confusion}')
    
    return oa,aa,kappa,each_acc

def createAutoReport(net, data, num_classes, device, forward_logits_fn=None):
    class_names = [f"Class{i}" for i in range(int(num_classes))]
    return createReport(net, data, class_names, device, forward_logits_fn=forward_logits_fn)
