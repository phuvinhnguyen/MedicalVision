import pandas as pd
import os

dataset_file = os.path.join(os.path.dirname(
    __file__), '../MedicalVision/preprocess/Detection_NIH_1024.csv')
destination_data_file = os.path.join(os.path.dirname(
    __file__), '../MedicalVision/preprocess/Detection_NIH_1024_version_2.csv')

if __name__ == '__main__':
    data = pd.read_csv(dataset_file)
    label_map = {v: i for i, v in enumerate(data['Finding Label'].unique())}
    data['Label'] = data['Finding Label'].map(label_map)

    data['bbox'] = data[['Bbox [x', 'y', 'w', 'h]']].values.tolist()
    
    # image_path of this dataframe should be agg as first
    data = data.groupby('Image Index').agg({
        'Label': list,
        'bbox': list,
        'image_path': 'first'  # Assuming that image_path is consistent for each Image Index
    }).reset_index()

    data = data[['Image Index', 'Label', 'bbox', 'image_path']]

    data.to_csv(destination_data_file)
