import numpy as np

from utils import plot_confusion_matrix, get_dfs, create_idxs, scale_df, load_csvs, find_best_threshold
from utils import features_extraction_new, load_single_cell_models_and_data, get_latent_space, features_extraction_cells
from datasets import DatasetLSTM, DatasetDecoder
from torch.utils.data import DataLoader
from architecture import LSTMAutoencoder, LSTMDecoder
from farmeworks.autoencoder import AE
from farmeworks.decode import Decoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

input_size = 11
hidden_size = 128  # Increased hidden size
num_layers = 3  # Increased number of layers
dropout = 0.1


def fill_missing_values(df):
    # fill null values with knn imputer not idx and step
    imputer = KNNImputer(n_neighbors=3)
    impute_df = df.drop(columns=['step', 'nrCellIdentity'])
    impute_df = pd.DataFrame(imputer.fit_transform(impute_df), columns=impute_df.columns)
    impute_df['step'] = df['step']
    impute_df['nrCellIdentity'] = df['nrCellIdentity']
    return impute_df


def plot_data(df, cell_id):
    cell_id = int(cell_id)
    df = df[df['nrCellIdentity'] == cell_id]
    df = df.drop(columns=['step', 'nrCellIdentity'])  # , 'idx'])
    df.plot(subplots=True, figsize=(10, 10), title=f'Cell {cell_id} data')
    plt.show()


def single_cell_train(cell_id=0, train_size=8, name=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_only = False
    print(f"Using device: {device}")
    ues, cells = load_csvs()
    dfs = get_dfs(ues, cells)
    # dfs = features_extraction_new(dfs, only_cell=cell_only, single_cell=cell_id)
    dfs = features_extraction_cells(dfs)

    train = create_idxs(dfs, win_size=3, only_cell=cell_id)
    # # test = create_idxs(dfs[train_size:], win_size=3, only_cell=cell_id)
    #
    train, scaler = scale_df(train)
    # # test, _ = scale_df(test, test=True, scaler=scaler)
    train.fillna(0, inplace=True)
    print("number of NaN values:\n", train.isna().sum().sum())
    # # todo fix scales
    with open(f'scalers/scaler_single_cell_{cell_id}_try.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Finished preprocessing data")
    print(train.shape)
    batch_size = 1
    h_params = {'lr': 0.0001, 'w_d': 1e-5, 'epochs': 2}
    input_size = 11  # Number of features
    hidden_size = 128  # Increased hidden size
    num_layers = 3  # Increased number of layers
    dropout = 0.1
    #
    model = LSTMAutoencoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=h_params['lr'], weight_decay=h_params['w_d'])
    criterion = nn.MSELoss(reduction='mean')

    train_dataset = DatasetLSTM(train)

    # test_dataset = DatasetLSTM(test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    #
    train = AE(model=model, optimizer=optimizer, criterion=criterion, device=device)
    train.train_single_ae_cell(train_loader, epochs=h_params['epochs'],
                               name=f'cell_{cell_id}_{name}_eps_{h_params["epochs"]}')
    print(f"Training completed cell_{cell_id}\n")


def single_cell_evaluation(cell_id=0, data=None):
    ue_test_mali = data[0]
    cell_test_mali = data[1]
    # normal data
    ue_test_norm = pd.read_csv('data/ue_norm100_seed_44.csv')
    cell_test_norm = pd.read_csv('data/cell_norm100_seed_44.csv')

    # ue_test_norm = ue_test_norm[ue_test_norm['step'] > 25]
    # cell_test_norm = cell_test_norm[cell_test_norm['step'] > 25]
    # process
    mali_dfs = get_dfs([ue_test_mali], [cell_test_mali])
    norm_dfs = get_dfs([ue_test_norm], [cell_test_norm])
    # extract features
    # mali_dfs = features_extraction_new(mali_dfs, only_cell=False, single_cell=cell_id)
    # norm_dfs = features_extraction_new(norm_dfs, only_cell=False, single_cell=cell_id)
    mali_dfs = features_extraction_cells(mali_dfs)
    norm_dfs = features_extraction_cells(norm_dfs)

    # create indexes for evaluation
    mali_test = create_idxs(mali_dfs, win_size=3, only_cell=cell_id)
    norm_test = create_idxs(norm_dfs, win_size=3, only_cell=cell_id)
    # scale using the train scaler with open
    # mali_tests = [create_idxs(mali_dfs, win_size=3, only_cell=i) for i in range(1, 7)]

    with open(f'scalers/scalers140724/scaler_single_cell_{cell_id}_140724.pkl', 'rb') as f:
        scaler = pickle.load(f)

    mali_test, _ = scale_df(mali_test, test=True, scaler=scaler)
    norm_test, _ = scale_df(norm_test, test=True, scaler=scaler)
    # norm_test.plot(subplots=True, figsize=(10, 10), title=f'Cell {cell_id} data')
    # plt.show()
    # fill_missing_values(mali_test)
    # fill_missing_values(norm_test)

    print("Finished preprocessing data")

    batch_size = 1

    test_norm_dataset = DatasetLSTM(norm_test)
    test_anom_dataset = DatasetLSTM(mali_test)

    test_norm_loader = DataLoader(test_norm_dataset, batch_size=batch_size, shuffle=False)
    test_anom_loader = DataLoader(test_anom_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(input_size=11, hidden_size=128, num_layers=3, dropout=0.1)
    model.to(device)
    model_pth = f"models/models_colab/models_140724/cell_{cell_id}__eps_150.pth"
    model.load_state_dict(torch.load(model_pth, map_location=device))

    criterion = nn.MSELoss(reduction='mean')

    evaluator = AE(model=model, optimizer=None, criterion=criterion, device=device)
    loss_normal, loss_anomaly = evaluator.evaluate_single_ae_cell(test_norm_loader, test_anom_loader,
                                                                  name=f'cell_{cell_id}_eps_2')

    print(f"loss normal:\n{loss_normal}\nloss anomaly:\n{loss_anomaly}\n")
    tp_f1 = find_best_threshold(loss_normal, loss_anomaly, save=True, name=f'cell_{cell_id}_eps_2')
    res = plot_confusion_matrix(loss_normal, loss_anomaly, best_threshold=tp_f1, name=f'cell_{cell_id}_eps_2',
                                save=True)
    return res


def train_decoder(cell_id=0, train_size=7, h_params=None):
    if h_params is None:
        h_params = {'lr': 0.0001, 'w_d': 1e-5, 'epochs': 50}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # print(f"Using device: {device}")
    # ues, cells = load_csvs()
    # dfs = get_dfs(ues, cells)
    #
    # dfs = features_extraction_cells(dfs)  # , only_cell=False, single_cell=False)
    #
    # trains = [create_idxs(dfs, win_size=3, only_cell=i) for i in range(1, 7)]
    #
    # train_df = []
    # for cid in range(1, 7):
    #     with open(f'scalers/scalers140724/scaler_single_cell_{cid}_140724.pkl', 'rb') as f:
    #         scaler = pickle.load(f)
    #     train_c, _ = scale_df(trains[cid - 1], test=True, scaler=scaler)
    #     train_df.append(train_c)

    trains = []
    for cid in range(1, 7):
        with open(f'pickles/datasets/dataset_single_cell_{cid}.pkl', 'rb') as f:
            train = pickle.load(f)
            trains.append(train)

    train_df = trains
    print("Finished preprocessing data")

    batch_size = 1
    lr = 0.0001  # learning rate
    w_d = 1e-5  # weight decay
    epochs = 3

    input_size = 11
    hidden_size = 128  # Increased hidden size
    num_layers = 3  # Increased number of layers
    dropout = 0.1
    #
    ae = LSTMAutoencoder(input_size, hidden_size, num_layers, dropout)
    args = (input_size, hidden_size, num_layers, dropout)
    path = f'models/models_colab/models_140724/'
    single_cell_models, cell_loaders = load_single_cell_models_and_data(train_df, ae, args, path, device)
    latent_space = get_latent_space(single_cell_models, cell_loaders)

    print([len(v) for k, v in latent_space.items()])
    #
    train_set = DatasetDecoder(latent_space, cell_id)
    train_loader = DataLoader(train_set, batch_size=batch_size)

    print("latent space ready")
    model = LSTMDecoder(input_size, hidden_size * 2, num_layers, cell_id)
    model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=w_d)

    train = Decoder(model=model, optimizer=optimizer, criterion=criterion, device=device)
    train.train_single_decoder_cell(train_loader, epochs=10, name=f'decode_{cell_id}_eps_{epochs}')
    print(f"Training completed cell_{cell_id}\n")


def decoder_evaluation(cell_id=0, data=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ue_test_mali = pd.read_csv(f'data/ue_mali{cell_id}_seed_8.csv')
    cell_test_mali = pd.read_csv(f'data/cell_mali{cell_id}_seed_8.csv')

    if data is not None:
        ue_test_mali = data[0]
        cell_test_mali = data[1]

    # process
    mali_dfs = get_dfs([ue_test_mali], [cell_test_mali])
    # mali_dfs[0].dropna(inplace=True)

    # a = mali_dfs[0][mali_dfs[0]['step'] >= 15]
    # print(a[a['nrCellIdentity'] == 5])
    # extract features
    mali_dfs = features_extraction_cells(mali_dfs)
    # mali_dfs = [fill_missing_values(df) for df in mali_dfs]
    # print(mali_dfs[0])
    # plot_data(mali_dfs[0], 5)
    # # create indexes for evaluation
    mali_tests = [create_idxs(mali_dfs, win_size=3, only_cell=i) for i in range(1, 7)]
    # print(len(mali_dfs))

    tests_df = []
    # scale using the train scaler with open
    for cid in range(1, 7):
        with open(f'pickles/scalers/scaler_single_cell_{cid}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        mali_test, _ = scale_df(mali_tests[cid - 1], test=True, scaler=scaler)
        tests_df.append(mali_test)

    print("Finished preprocessing data")

    input_size = 11
    hidden_size = 128  # Increased hidden size
    num_layers = 3  # Increased number of layers
    dropout = 0.1

    ae = LSTMAutoencoder(input_size, hidden_size, num_layers, dropout)
    args = (input_size, hidden_size, num_layers, dropout)

    path = f'models/models_colab/models_040824/'
    # TODO: test load single cell models
    single_cell_models, cell_loaders = load_single_cell_models_and_data(tests_df, ae, args, path, device, decoder=True)

    # TODO: test get latent space
    latent_space = get_latent_space(single_cell_models, cell_loaders)
    # print(latent_space["4"][21][0].shape)
    model = LSTMDecoder(input_size, hidden_size * 2, num_layers, cell_id)
    model.to(device)
    model.load_state_dict(
        torch.load(f'models/decoders/decoders_040824/decode_{cell_id}_eps_200.pth', map_location=device))
    criterion = nn.MSELoss(reduction='mean')

    decoder = Decoder(model=model, optimizer=None, criterion=criterion, device=device)
    loss_normal, loss_anomaly = decoder.evaluate_decoder_cell(latent_space, cell_id, name=f'decode_{cell_id}_eps_{5}')
    print(
        f"cell{cell_id}\naverage loss normal: {sum(loss_normal) / len(loss_normal)}, average loss anomaly: {sum(loss_anomaly) / len(loss_anomaly)}\n")
    print(f"len loss_normal: {len(loss_normal)}, len loss_anomaly: {len(loss_anomaly)}")

    print(f"avg loss normal {sum(loss_normal) / len(loss_normal)}")
    print(f"avg loss anomaly {sum(loss_anomaly) / len(loss_anomaly)}")
    # print first 10 average loss
    print(f"first 10 average loss normal {sum(loss_normal[:10]) / len(loss_normal[:10])}")
    print(f"first 10 average loss anomaly {sum(loss_anomaly[:10]) / len(loss_anomaly[:10])}")

    tp_f1 = find_best_threshold(loss_normal, loss_anomaly, save=True, name=f'cell_{cell_id}_eps_2', verbose=True)
    res = plot_confusion_matrix(loss_normal, loss_anomaly, best_threshold=tp_f1, name=f'cell_{cell_id}_eps_2',
                                save=True)

    # print index of all normal loss bigger then 10
    print('print index of all normal loss bigger then 10')
    print([i for i, x in enumerate(loss_normal) if x > 10])

    # plt.plot(loss_normal, label='normal')
    # plt.plot(loss_anomaly, label='anomaly')
    # # plot threshold
    # plt.axhline(y=tp_f1, color='r', linestyle='--', label='threshold')
    # plt.legend()
    # plt.title(f'Loss distribution for cell {cell_id}')
    # plt.show()

    return res

def create_single_cell_dataset(cid, pth):
    ues, cells = load_csvs()
    # print(f"len ues: {len(ues)}, len cells: {len(cells)}")
    dfs = get_dfs(ues, cells)
    dfs = features_extraction_cells(dfs)

    train = create_idxs(dfs, win_size=3, only_cell=cid)

    train, scaler = scale_df(train)
    train.fillna(0, inplace=True)

    with open(f'pickles/scalers/scaler_single_cell_{cid}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # train_dataset = DatasetLSTM(train)
    with open(f'pickles/datasets/dataset_single_cell_{cid}.pkl', 'wb') as f:
        pickle.dump(train, f)
    print(f"Dataset for cell {cid} created size: {train.shape}")



if __name__ == '__main__':

    # for i in range(1, 7):
    #     create_single_cell_dataset(i, 'pickles')
    # decoder_evaluation(cell_id=2)
    # train_decoder(cell_id=1, train_size=7)
    # # single_cell_train(cell_id=1, train_size=8)
    # for cell_id in range(1, 7):
    #     single_cell_train(cell_id=cell_id, train_size=7)
    # for cell_id in range(1, 7):
    #     single_cell_evaluation(cell_id=cell_id)
    # for cell_id in range(1, 7):
    #     decoder_evaluation(cell_id=cell_id)
    # single_cell_evaluation(cell_id=3)

    # train_decoder(cell_id=1, train_size=7)
    # decoder_evaluation(cell_id=1)
    #
    # ue100 = pd.read_csv('data/ue_mali3_100_seed_44.csv')
    # cell100 = pd.read_csv('data/cell_mali3_100_seed_44.csv')
    # # only steps 25 or more
    # ue100_25 = ue100[ue100['step'] > 25]
    # cell100_25 = cell100[cell100['step'] > 25]
    # # reduce 25 from steps
    fix_steps = []
    for i in range(72):
        fix_steps += [i] * 50

    # ue100_25['step'] = fix_steps
    # cell100_25['step'] = cell100_25['step'] - 26
    #
    # print(ue100.shape, cell100.shape)
    # decoder_evaluation(cell_id=3, data=[ue100_25, cell100_25])
    # single_cell_evaluation(cell_id=1)
    res_dicts = []
    for cid in [2, 3, 5]:
        print(f"\n-------------------\nCell {cid}\n-------------------")
        ue100 = pd.read_csv(f'data/eval_data/ue_mali{cid}_100_seed_44.csv')
        cell100 = pd.read_csv(f'data/eval_data/cell_mali{cid}_100_seed_44.csv')
        # only steps 25 or more
        ue100_25 = ue100[ue100['step'] > 25]
        cell100_25 = cell100[cell100['step'] > 25]
        # reduce 25 from steps
        ue100_25['step'] = fix_steps
        cell100_25['step'] = cell100_25['step'] - 26
        temp = decoder_evaluation(cell_id=cid, data=[ue100_25, cell100_25])
        res_dicts.append(temp)

    acc, f1, prec, rec, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0, 0, 0
    for res in res_dicts:
        acc += res['Accuracy']
        f1 += res['F1 Score']
        prec += res['Precision']
        rec += res['Recall']
        tp += res['tp']
        fp += res['fp']
        tn += res['tn']
        fn += res['fn']
    acc /= len(res_dicts)
    f1 /= len(res_dicts)
    prec /= len(res_dicts)
    rec /= len(res_dicts)
    # balance accuracy
    bacc = (rec + tn / (tn + fp)) / 2
    print(f"Average accuracy: {acc}, Average f1: {f1}, Average precision: {prec}, Average recall: {rec}, Balance accuracy: {bacc}")
    # plot confusion matrix from tp, fp, tn, fn
    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Reds)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Normal', 'Anomaly'])
    plt.yticks([0, 1], ['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix')
    plt.grid(0)
    plt.show()
