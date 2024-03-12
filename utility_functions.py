import os
from io import open
import torch
import numpy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class TimeseriesTorch(object):
    def __init__(self, path, num_features):
        self.train = self.load_series(os.path.join(path, 'train.dat'), num_features)
        self.valid = self.load_series(os.path.join(path, 'dev.dat'), num_features)
        self.test = self.load_series(os.path.join(path, 'test.dat'), num_features)

    def load_series(self, path, num_features):
        """Load health data to Tensor objects."""
        assert os.path.exists(path)
        
        # Get sequence length
        with open(path, 'r', encoding="utf8") as f:
            timesteps = 0
            for line in f:
                timesteps += 1

        # Load multidimensional data with targets
        with open(path, 'r', encoding="utf8") as f:
            steps = torch.Tensor(timesteps, num_features)
            targets = torch.Tensor(timesteps)
            pos = 0
            for line in f:
                healthdata = line.split()
                values = []
                # remove encid,sepsis,severity,timestep
                # from multidimensional data
                for value in healthdata[4:4+num_features]:
                    values.append(float(value))
                # keep severity as target
                severity = healthdata[2]
                    
                steps[pos] = torch.from_numpy(numpy.array(values))
                targets[pos] = float(severity)
                pos += 1
        return [steps,targets]

class TimeseriesNumPy(object):
    def __init__(self, path, num_features):
        self.train = self.load_series(os.path.join(path, 'train.dat'), num_features)
        self.valid = self.load_series(os.path.join(path, 'dev.dat'), num_features)
        self.test = self.load_series(os.path.join(path, 'test.dat'), num_features)

    def load_series(self, path, num_features):
        """Load health data to NumPy arrays."""
        assert os.path.exists(path)
        
        # Get sequence length
        with open(path, 'r', encoding="utf8") as f:
            timesteps = 0
            for line in f:
                timesteps += 1

        # Load multidimensional data with targets
        with open(path, 'r', encoding="utf8") as f:
            stepsl = []
            targetsl = []
            for line in f:
                healthdata = line.split()
                values = []
                # remove encid,sepsis,severity,timestep
                # from multidimensional data
                for value in healthdata[4:4+num_features]:
                    values.append(float(value))
                # keep severity as target
                severity = float(healthdata[2])
                    
                stepsl.append(values)
                targetsl.append(severity)

        steps = numpy.array(stepsl,dtype=float)
        targets = numpy.array(targetsl,dtype=float)
        
        return [steps,targets]
    



class VariableLengthTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, exclude_columns, target_column):
        self.dataframe = dataframe
        # Automatically determine feature columns by excluding specified columns
        self.feature_columns = [col for col in dataframe.columns if col not in exclude_columns]
        self.target_column = target_column
        self.patients = dataframe['id'].unique()
        
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_data = self.dataframe[self.dataframe['id'] == patient_id]
        
        # Extract features and target
        features = patient_data[self.feature_columns].values
        target = patient_data[self.target_column].iloc[0]  # Assuming the target is the same for all rows of a patient
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float)
        target_tensor = torch.tensor(target, dtype=torch.float)
        
        return features_tensor, target_tensor


def collate_fn(batch):
    # Separate features and targets, and pad features
    features, targets = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    targets_tensor = torch.tensor(targets, dtype=torch.float)
    
    return features_padded, targets_tensor


'''Converts features to tensors. Last column is target'''
class TimeseriesTorchDF(object):
    def __init__(self, dataframe, feature_columns):
        self.dataframe = dataframe
        self.feature_columns = feature_columns[:-1]
        self.target_column = feature_columns[-1]
        
    def load_series(self):
        features = torch.tensor(self.dataframe[self.feature_columns].values, dtype=torch.float)
        targets = torch.tensor(self.dataframe[self.target_column].values, dtype=torch.float)
        
        return [features, targets]
