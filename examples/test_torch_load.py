import time

import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path, cache_path='.criteo_test')
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def load_model():
    save_path = '/home/eduapp/pytorch-fm/examples/model/dcn_20201103__12_37_51.pt'
    save_path = '/home/eduapp/pytorch-fm/examples/model/afn_20201103__14_00_39.pt'
    save_path = '/home/eduapp/pytorch-fm/examples/model/xdfm_20201103__17_19_23.pt'
    model = torch.load(save_path)#.to(device)
    # print(model.eval())
    print(model)
    return model


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    result_list = []
    result_pred_true = []   # 预测为正的样本概率分布

    for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device).long(), target.to(device).long()
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())


    print('========pred result list save to file================')
    for i in range(len(targets)):
        result_list.append(str(targets[i]) + ',' + str(predicts[i]) + '\n')
        # 预测为正的样本中，有多少实际为正的
        if predicts[i] >= 0.5:
            result_pred_true.append(str(targets[i]) + ', ' + str(predicts[i]) + '\n')

    file = open('result_list.txt', "w")
    file.writelines(result_list)
    file.close()

    file = open('result_list_true.txt', "w")
    file.writelines(result_pred_true)
    file.close()


    from sklearn.metrics import classification_report
    arr = []
    for x in predicts:
        # print(x)
        arr.append(1) if x >= 0.5 else arr.append(0)

    print(classification_report(targets, arr))

    auc = roc_auc_score(targets, predicts)
    print('auc={}'.format(auc))


if __name__ == '__main__':
    model = load_model()
    device = torch.device('cpu')
    dataset_path = '/home/eduapp/best_flow/release-1.1.0/train_data/202011/dnn_part_test.csv'

    t1 = time.time()
    dataset = get_dataset('criteo', dataset_path)
    train_length = 0
    valid_length = 0
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))

    test_data_loader = DataLoader(test_dataset, batch_size=64, num_workers=0)
    print('dataset time={}'.format(time.time() - t1))
    test(model, test_data_loader, device)