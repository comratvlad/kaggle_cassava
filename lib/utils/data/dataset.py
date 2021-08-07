from typing import List

import pandas as pd
from torch.utils.data import Dataset

from lib.utils.data.features import DatasetFeature


class FolderDataset(Dataset):
    """
    Standard torch.utils.data.Dataset class for frame-based data sets.
    """

    def __init__(self, folder_path: str, info_path: str, features: List[DatasetFeature],
                 transforms=None, filter_by: dict = None):
        """
        :param folder_path: absolute path to data set root
        :param info_path: absolute path to data set info file
        :param features: list with features that would be sampled
        :param transforms: function that takes the dictionary of sampled features and transforms features into a valid
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
        if filter_by is not None:
            _not_filtered_info_len = len(self.info)
            # Sic! If several columns are mentioned in filter_by, filters will be applied one-by-one.
            for column_name, value in filter_by.items():
                if column_name not in self.info.columns:
                    raise ValueError(f'{info_path}: cannot filter by column {column_name}; column does not exist')
                if isinstance(value, list):
                    do_not_exist = [v for v in value if v not in self.info[column_name].values]
                    if do_not_exist:
                        raise ValueError(f'{info_path}: values {do_not_exist} mentioned in filter_by '
                                         f'do not appear in column "{column_name}"')
                    self.info = self.info[self.info[column_name].isin(value)]
                else:
                    if value not in self.info[column_name].values:
                        raise ValueError(f'{info_path}: value "{value}" mentioned in filter_by '
                                         f'do not appear in column "{column_name}"')
                    self.info = self.info[self.info[column_name] == value]
            assert(len(self.info) < _not_filtered_info_len), 'Info length did not decrease after filtration'
        self.features = [feature(self.folder_path, self.info) for feature in features]
        self.transforms = transforms

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        sample_dict = dict()
        for feature in self.features:
            sample_dict[feature.name] = feature.read(index)
        if self.transforms is not None:
            sample_dict = self.transforms(**sample_dict)
        return sample_dict
