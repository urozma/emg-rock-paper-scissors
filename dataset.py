import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import kagglehub

class EMGDataset(Dataset):
    def __init__(self):
        pass

def get_data():
    path = kagglehub.dataset_download("kyr7plus/emg-4")
    # Define file names
    gesture_files = ['0.csv', '1.csv', '2.csv', '3.csv']
    dataframes = []

    # Extract data from each file and combine into one dataframe
    for i, file in enumerate(gesture_files):
        df = pd.read_csv(path+'/'+file, header=None)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)

    # Split dataframe into features (recordings) and targets (gestures)
    recordings = data.iloc[:, :-1]
    gestures = data[64]

    # Scale feature data
    scaler = StandardScaler()
    recordings_scaled = scaler.fit_transform(recordings)

    # Split data into training and test data
    recordings_trn, recordings_tst, gestures_trn, gestures_tst = train_test_split(recordings_scaled,
                                                                                  gestures,
                                                                                  test_size=0.2,
                                                                                  random_state=0)

    # Split training data into training and validation data
    recordings_trn, recordings_val, gestures_trn, gestures_val = train_test_split(recordings_trn,
                                                                                  gestures_trn,
                                                                                  test_size=0.1,
                                                                                  random_state=0)
    split_data = {'trn': {'recordings': recordings_trn, 'gestures': gestures_trn},
                  'val': {'recordings': recordings_val, 'gestures': gestures_val},
                  'tst': {'recordings': recordings_tst, 'gestures': gestures_tst}}

    return split_data

def get_loaders(batch_size):
    data = get_data()

    # Get data loaders
    recordings_trn_tensor = torch.tensor(data['trn']['recordings'], dtype=torch.float32)
    gestures_trn_tensor = torch.tensor(data['trn']['gestures'].values, dtype=torch.long)

    recordings_val_tensor = torch.tensor(data['val']['recordings'], dtype=torch.float32)
    gestures_val_tensor = torch.tensor(data['val']['gestures'].values, dtype=torch.long)

    recordings_tst_tensor = torch.tensor(data['tst']['recordings'], dtype=torch.float32)
    gestures_tst_tensor = torch.tensor(data['tst']['gestures'].values, dtype=torch.long)

    train_dataset = TensorDataset(recordings_trn_tensor, gestures_trn_tensor)
    val_dataset = TensorDataset(recordings_val_tensor, gestures_val_tensor)
    test_dataset = TensorDataset(recordings_tst_tensor, gestures_tst_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

