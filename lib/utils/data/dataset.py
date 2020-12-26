import pandas as pd
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """
    Standard torch.utils.data.Dataset class for frame-based data sets.
    """
    def __init__(self, folder_path, info_path, features, transforms, filter_by: dict = None):
        """
        :param folder_path: absolute path to data set root
        :param info_path: pass
        :param features: List with names of the features that would be sampled. For example: ['rgb', 'optical_flow', 'lbp']
        :param transforms: Function that takes the dictionary of sampled features and transforms features into a valid
        neural network input
        For example:
        def rgb_lbr_pair_transform(features_dict):
            rgb = np.moveaxis(features_dict['rgb_pair'], -1, 1) / 255.
            lbp = features_dict['lbp'].mean(axis=0)
            return {'rgb': rgb, 'lbp': lbp}
        :param filter_by: dictionary in format {column name: value(s) to filter by}. If dictionary value is a list,
        it's treated as list of choices; else column value and dictionary value equality is checked.
        If several columns are mentioned in filter_by, filters are applied one-by-one.
        """
        self.folder_path = folder_path
        self.info = pd.read_csv(info_path)
        self.features = [feature(self.folder_path, self.info) for feature in features]
        self.transforms = transforms

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        sample_dict = dict()
        for feature in self.features:
            sample_dict[feature.name] = feature.read(index)
        if self.transforms is not None:
            sample_dict = self.transforms(sample_dict)
        return sample_dict
