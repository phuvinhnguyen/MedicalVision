import json
import argparse
import funcy
from sklearn.model_selection import train_test_split

def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'images': images, 
            'annotations': annotations, 'categories': categories}, coco)
    print(f"Saved {len(images)} entries in {file}.")

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def split_train_test(annotations, train, test, split=0.8):
    with open(annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']
        
        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)
        
        print(f"There are {len(images)} in this annotation")

        x, y = train_test_split(images, train_size=split, random_state=79)

        save_coco(train, x, filter_annotations(annotations, x), categories)
        save_coco(test,  y, filter_annotations(annotations, y), categories)
