import numpy as np

FMT = '%.4f'

def all_metrics(outputs, pre_labels, true_labels):
    metrics_name = ['hamming_loss', 'avg_precision', 'one_error', 'ranking_loss', 'coverage', 'macrof1', 'microf1','SubsetA','AUC']
    hamming_loss = HammingLoss(pre_labels, true_labels)
    avg_precision = AveragePrecision(outputs, true_labels)
    one_error = OneError(outputs, true_labels)
    ranking_loss = RankingLoss(outputs, true_labels)
    coverage = Coverage(outputs, true_labels)
    macrof1 = MacroF1(pre_labels, true_labels)
    microf1 = MicroF1(pre_labels, true_labels)
    SubsetA = SubsetAccuracy(pre_labels, true_labels)
    metrics_res = [hamming_loss, avg_precision, one_error, ranking_loss, coverage, macrof1, microf1, SubsetA]
    return  metrics_name,metrics_res


def HammingLoss(pre_labels, true_labels):
    m, q = true_labels.shape
    miss_label = np.sum((pre_labels == true_labels) == False)
    hl = miss_label / (m * q)
    return float(FMT % hl)


def AveragePrecision(outputs, true_labels):
    m, q = true_labels.shape
    ap = 0
    all_zero_m = 0
    for i in range(m):
        rel_lbl = np.count_nonzero(true_labels[i])
        if rel_lbl != 0:
            rel_lbl_idx = np.where(true_labels[i] == 1)[0]
            tmp_out = outputs[i]
            sort_idx = np.argsort(-tmp_out)
            cnt = 0
            for j in rel_lbl_idx:
                t = np.argwhere((tmp_out >= outputs[i, j]) == True)
                cntt = len(np.intersect1d(t, rel_lbl_idx))
                pre_lbl_rank = np.where(sort_idx[:] == j)[0][0] + 1
                cnt += cntt / pre_lbl_rank
            cnt /= rel_lbl
        else:
            all_zero_m += 1
            cnt = 0
        ap += cnt
    ap /= (m - all_zero_m)
    return float(FMT % ap)


def OneError(outputs, true_labels):
    m, q = true_labels.shape
    index = np.argmax(outputs, axis=1)
    true_labels = true_labels.reshape(1, m * q)
    index = [i * q for i in range(m)] + index
    oe = np.sum(true_labels[:, index] != 1) / m
    return float(FMT % oe)


def RankingLoss(outputs, true_labels):
    m, q = true_labels.shape
    rl = 0
    all_zero_m = 0
    for i in range(m):
        rel_lbl = np.count_nonzero(true_labels[i])
        if rel_lbl != 0:
            tmp_out = outputs[i, :]
            sort_idx = np.argsort(-tmp_out)
            tmp_true = true_labels[i, :][sort_idx]
            n_zero = 0
            rl_ins = 0
            for j in range(q):
                if tmp_true[j] == 0:
                    n_zero += 1
                elif tmp_true[j] == 1:
                    rl_ins += n_zero
            rl += rl_ins / (rel_lbl * (q - rel_lbl))
        else:
            all_zero_m += 1
    rl = rl / (m - all_zero_m)
    return float(FMT % rl)

def Coverage(outputs, true_labels):
    m, q = true_labels.shape
    cov = 0
    for i in range(m):
        tmp_out = outputs[i, :]
        sort_idx = np.argsort(-tmp_out)
        tmp_true = true_labels[i, :][sort_idx]
        if 0 != np.sum(tmp_true):
            cov += np.max(np.where(tmp_true == 1))
    return float(FMT % (cov / m / q))


def MacroF1(pre_labels, true_labels):
    m, q = true_labels.shape
    maf = 0
    for i in range(q):
        tp = np.sum(((pre_labels[:, i]) & (true_labels[:, i])) == True)
        fp = np.sum((pre_labels[:, i] & (1 - true_labels[:, i])) == True)
        fn = np.sum(((1 - pre_labels[:, i]) & true_labels[:, i]) == True)
        if tp + fp + fn == 0:
            tmp_maf = 0
        else:
            tmp_maf = (2 * tp) / (2 * tp + fp + fn)
        maf += tmp_maf
    return float(FMT % (maf / q))


def MicroF1(pre_labels, true_labels):
    tp = np.sum(((pre_labels) & (true_labels)) == True)
    fp = np.sum((pre_labels & (1 - true_labels)) == True)
    fn = np.sum(((1 - pre_labels) & true_labels) == True)
    if tp + fp + fn == 0:
        mif = 0
    else:
        mif = (2 * tp) / (2 * tp + fp + fn)
    return float(FMT % mif)

def SubsetAccuracy(predict_labels, test_target):
    test_data_num = predict_labels.shape[0]
    class_num = predict_labels.shape[1]
    correct_num = 0
    for i in range(test_data_num):
        for j in range(class_num):
            if predict_labels[i][j] != test_target[i][j]:
                break
        if j == class_num - 1:
            correct_num = correct_num + 1
            
    return correct_num*1.0/test_data_num
#    return float(FMT % SubsetA)


def MacroAveragingAUC(outputs, true_labels):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    P = []
    N = []
    labels_size = []
    not_labels_size = []
    AUC = 0
    for i in range(class_num):
        P.append([])
        N.append([])
    
    for i in range(test_data_num):
            for j in range(class_num):
                if true_labels[i][j] == 1:
                    P[j].append(i)
                else:
                    N[j].append(i)
    
    for i in range(class_num):
        labels_size.append(len(P[i]))
        not_labels_size.append(len(N[i]))
    
    for i in range(class_num):
        auc = 0
        for j in range(labels_size[i]):
            for k in range(not_labels_size[i]):
                pos = outputs[P[i][j]][i]
                neg = outputs[N[i][k]][i]
                if pos > neg:
                    auc = auc + 1
        AUC = AUC + auc*1.0/(labels_size[i]*not_labels_size[i] + 0.001)
    return AUC*1.0/class_num