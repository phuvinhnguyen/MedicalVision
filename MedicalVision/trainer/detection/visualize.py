import matplotlib.pyplot as plt
import torch
import os
from PIL import Image

def plot_results(pil_img, prediction, ground_truth, id2label=None):
    print(ground_truth)

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    # Plot ground truth
    for gt in ground_truth:
        gt_bbox = gt['bbox']
        gt_category = gt['category']
        ax.add_patch(plt.Rectangle(
            (gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], fill=False, color='green', linewidth=3)
            )
        text = f'{id2label[gt_category]}'
        ax.text(gt_bbox[0], gt_bbox[1], text, fontsize=15,
                bbox=dict(facecolor='green', alpha=0.5))
    
    # Plot prediction
    scores, labels, boxes = prediction['scores'].tolist(), prediction['labels'].tolist(), prediction['boxes'].tolist()
    for score, label, (xmin, ymin, xmax, ymax) in zip(scores, labels, boxes):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=1))
        print(label)
        text = f'{id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymax, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.savefig('example.png')
    plt.show()


def plot_from_dataset(model,
                      dataset,
                      processor,
                      idx=1,
                      threshold=0.5,
                      device='cuda',
                      ):
    pixel_values, target = dataset[idx]
    image, ground_truth = dataset.__class__.__base__.__getitem__(dataset, idx)
    pixel_values = pixel_values.unsqueeze(0).to(device)
    print(ground_truth)
    ground_truth = [{
        'bbox': item['bbox'],
        'category': item['category_id']
    } for item in ground_truth]
    id2label = {k:v for k,v in model.id2label.items()}
    print(id2label)

    # Prediction
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # Post process
    results = processor.post_process_object_detection(outputs,
                                                    target_sizes=[image.size],
                                                    threshold=threshold)[0]
    
    # Visualize
    plot_results(image, results, ground_truth, id2label)
