import pandas as pd
import json
import os

dataset_file = os.path.join(os.path.dirname(
    __file__), '../MedicalVision/preprocess/Detection_NIH_1024.csv')
destination_data_file = os.path.join(os.path.dirname(
    __file__), '../MedicalVision/preprocess/Detection_NIH_1024_version_2.csv')
destination_label_map = os.path.join(os.path.dirname(
    __file__), '../MedicalVision/preprocess/Detection_NIH_1024_version_2.json')

if __name__ == '__main__':
    data = pd.read_csv(dataset_file)
    label_map = {int(i): v for i, v in enumerate(data['Finding Label'].unique())}
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

    with open(destination_label_map, 'w') as f:
        json.dump(label_map, f)