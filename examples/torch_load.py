import torch
from sklearn.metrics import roc_auc_score
import tqdm
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
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def load_model():
    save_path = '/home/eduapp/pytorch-fm/data/criteo/save_dir/ipnn.pt'
    # save_path = '/home/eduapp/pytorch-fm/data/criteo/save_dir/lr.pt'
    model = torch.load(save_path)
    print(model.eval())
    return model


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    result_list = []
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device).long(), target.to(device).long()
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            # result_list.append(str(target.tolist()) + ',' + str(y.tolist()) + '\n')

    print('========================')
    # print(predicts)
    # print(targets)

    # result_list = []
    for i in range(len(targets)):
        # print('{}, {}, {}'.format(i, targets[i], predicts[i]))
        result_list.append(str(targets[i]) + ',' + str(predicts[i]) + '\n')
    file = open('result_list.txt', "w")
    file.writelines(result_list)
    file.close()


    from sklearn.metrics import classification_report
    arr = []
    for x in predicts:
        # print(x)
        arr.append(1) if x >= 0.5 else arr.append(0)

    print(classification_report(targets, arr))

    auc = roc_auc_score(targets, arr)
    print('auc={}'.format(auc))


if __name__ == '__main__':
    model = load_model()
    device = torch.device('cpu')

    dataset_path = '/home/eduapp/best_flow/20200907_more2more/all_features.train.1000.fe_output.csv'
    # dataset_path = '/home/eduapp/pytorch-fm/examples/all_features.train.1000.fe_output.csv'
    dataset = get_dataset('criteo', dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    test_data_loader = DataLoader(test_dataset, batch_size=2048, num_workers=0)

    test(model, test_data_loader, device)

