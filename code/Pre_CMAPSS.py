import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

np.set_printoptions(threshold=np.inf)

seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

TIME_WINDOW = 60          # 25,30,35,40,45,50....

class MyDataG(Dataset):
    def __init__(self, filepath, step_size=1):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.step_size = step_size

        self.x_data = torch.from_numpy(xy[:, 2:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.ids = torch.from_numpy(xy[:, [0]])

        self.samples_x = []
        self.samples_y = []

        unique_ids = torch.unique(self.ids)

        for engine_id in unique_ids:
            engine_indices = (self.ids == engine_id).nonzero(as_tuple=True)[0]
            engine_x = self.x_data[engine_indices]
            engine_y = self.y_data[engine_indices]

            for i in range(0, engine_x.shape[0] - TIME_WINDOW + 1, step_size):
                window_x = engine_x[i:i+TIME_WINDOW]
                label_y = engine_y[i + TIME_WINDOW - 1]

                self.samples_x.append(window_x.unsqueeze(-1))
                self.samples_y.append(label_y)

        self.len = len(self.samples_x)

    def __getitem__(self, index):
        return self.samples_x[index], self.samples_y[index]

    def __len__(self):
        return self.len



class MyDataG2(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, 2:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.ids = torch.from_numpy(xy[:, [0]])

        self.samples_x = []
        self.samples_y = []

        unique_ids = torch.unique(self.ids)

        for engine_id in unique_ids:
            engine_indices = (self.ids == engine_id).nonzero(as_tuple=True)[0]
            engine_x = self.x_data[engine_indices]
            engine_y = self.y_data[engine_indices]

            if engine_x.shape[0] >= TIME_WINDOW:
                start_idx = engine_x.shape[0] - TIME_WINDOW
                window_x = engine_x[start_idx:start_idx + TIME_WINDOW]
                label_y = engine_y[start_idx + TIME_WINDOW - 1]

                self.samples_x.append(window_x.unsqueeze(-1))
                self.samples_y.append(label_y)

        self.len = len(self.samples_x)

    def __getitem__(self, index):
        return self.samples_x[index], self.samples_y[index]

    def __len__(self):
        return self.len