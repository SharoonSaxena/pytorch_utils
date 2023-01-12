import torch


from torch.utils.data import Dataset, DataLoader


def tabular_data_loader():
    """
class TabularDataLoader(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt("WineQT.csv", delimiter=",", dtype=np.float16, skiprows=1)
        self.x = xy[:,:-2]
        self.y = xy[:, [-2]]
        self.n_sample = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_sample

class CustomTransform: #ToTensor
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)
    """
    pass
