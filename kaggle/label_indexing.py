import pandas as pd
import os

dataset_file = os.path.join(os.path.dirname(
    __file__), '../preprocess/Detection_NIH_1024.csv')
destination_data_file = os.path.join(os.path.dirname(
    __file__), '../preprocess/Detection_NIH_1024_version_2.csv')

if __name__ == '__main__':
    data = pd.read_csv(dataset_file)
    label_map = {v: i for i, v in enumerate(data['Finding Label'].unique())}
    data['Label'] = data['Finding Label'].map(label_map)

    data['bbox'] = data[['Bbox [x', 'y', 'w', 'h]']].values.tolist()
    data = data.groupby('Image Index').agg(
        lambda x: [i for i in x]).reset_index()

    data = data[['Image Index', 'Label', 'bbox']]

    data.to_csv(destination_data_file)
