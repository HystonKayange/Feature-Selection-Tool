import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

class GenericDataset(Dataset):
    def __init__(self, data_dir, delimiter=',', header=True):
        if data_dir.lower().endswith('.csv') or data_dir.lower().endswith('.txt'):
            # Load CSV or TXT file with the specified delimiter
            data = pd.read_csv(data_dir, header=None if not header else 0, delimiter=delimiter)

            # Separate features and labels
            self.features = data.iloc[:, :-1]
            self.labels = data.iloc[:, -1]

            # Identify categorical columns
            categorical_columns = self.features.select_dtypes(include=['object']).columns

            if not categorical_columns.empty:
                # One-hot encode categorical columns
                encoder = OneHotEncoder(sparse=False, drop='first')
                encoded_categorical = pd.DataFrame(encoder.fit_transform(self.features[categorical_columns]))

                # Concatenate numerical and encoded categorical features
                self.features = pd.concat([self.features.drop(categorical_columns, axis=1), encoded_categorical], axis=1)

        else:
            raise ValueError("Unsupported file format. Please provide a CSV or TXT file.")

        self.n_samples = data.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return np.array(self.features.iloc[index]), self.labels.iloc[index]

   