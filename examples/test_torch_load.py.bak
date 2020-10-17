import torch
from sklearn.metrics import roc_auc_score
import tqdm
from torch.utils.data import DataLoader

# from examples.main_params import get_model
from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
import time

from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
# from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork

def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    import numpy as np
    field_dims = np.array([2, 4, 2, 34, 18, 7, 998, 208, 47, 3, 2, 5, 3, 3, 982, 972, 375, 982, 951, 443, 979, 951, 455, 606, 850, 808, 1012, 573, 573, 87, 670, 588, 664, 663, 670, 99, 98, 52, 1072, 1057, 1059, 390, 1060, 479, 429, 175, 860, 860, 860, 860, 196, 893, 216, 903, 178, 665, 910, 570, 407, 4, 22, 139])
    # field_dims = dataset.field_dims
    # field_dims = [2    4    2   34   18    7  998  208   47    3    2    5    3    3
    #               982  972  375  982  951  443  979  951  455  606  850  808 1012  573
    #               573   87  670  588  664  663  670   99   98   52 1072 1057 1059  390
    #               1060  479  429  175  860  860  860  860  196  893  216  903  178  665
    #               910  570  407    4   22  139]

    print('field_dims={}'.format(field_dims))
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'hofm':
        # return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
        return LogisticRegressionModel(field_dims)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':    # 比较有效果
        return ProductNeuralNetworkModel(field_dims, embed_dim=64, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':    # precision：86%，recall：16%
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':    # 实验表明，该模型比现有的DeepFM、DCN和NFM等深度学习特征组合模型具有更强的表达能力
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':     # 效果比ipnn差一点点，但是已经很好了
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path, cache_path='.criteo_test2')
        # return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def load_model():
    save_path = '/home/eduapp/pytorch-fm/data/criteo/save_dir/ipnn.pt'
    # save_path = '/home/eduapp/pytorch-fm/data/criteo/save_dir/lr.pt'
    save_path = '/home/eduapp/pytorch-fm/examples/model/fm.pt'
    save_path = '/home/eduapp/pytorch-fm/examples/model/ipnn.pt'
    # model = torch.load(save_path)#.to(device)
    # print(model.eval())

    model_name = 'ipnn'
    dataset = None
    device = torch.device('cpu')
    model = get_model(model_name, dataset).to(device)
    model.load_state_dict(torch.load(save_path))
    # model.eval()
    print(model)
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

    print('========pred result list save to file================')
    for i in range(len(targets)):
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

    auc = roc_auc_score(targets, predicts)
    print('auc={}'.format(auc))


if __name__ == '__main__':
    model = load_model()
    device = torch.device('cpu')

    dataset_path = '/home/eduapp/best_flow/20200907_more2more/all_features.train.1000.fe_output.csv'
    # dataset_path = '/home/eduapp/pytorch-fm/examples/all_features.train.1000.fe_output.csv'
    dataset_path = '/home/eduapp/best_flow/20200907_more2more/all_features_use_model_estimate_path.fe_output.csv'
    # dataset_path = '/home/eduapp/best_flow/20200907_more2more/all_features.train.fe_output.csv'
    # dataset_path = '/home/eduapp/best_flow/20200907_more2more/all_features.train.fe_output_ipnn_test.csv'
    dataset_path = '/home/eduapp/pytorch-fm/examples/all_features.train.fe_output.10w.csv'
    dataset_path = '/home/eduapp/pytorch-fm/examples/all_features_use_model_estimate.fe_output.10w.csv'

    t1 = time.time()
    dataset = get_dataset('criteo', dataset_path)
    train_length = int(len(dataset) * 0.1)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    test_data_loader = DataLoader(test_dataset, batch_size=2048, num_workers=0)
    print('dataset time={}'.format(time.time() - t1))

    test(model, test_data_loader, device)

    # test_data_loader = DataLoader(dataset, batch_size=2048, num_workers=0)
    # test(model, test_data_loader, device)

