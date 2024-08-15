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
    gt_bbox = ground_truth['bbox']
    gt_category = ground_truth['category']
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
        text = f'{id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymax, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.savefig('example.png')
    plt.show()


def plot_from_dataset(model,
                      dataset,
                      processor,
                      idx=1,
                      image_dir=None,
                      threshold=0.5,
                      device='cuda',
                      ):
    ground_truth = list(dataset.coco.anns.values())[idx]
    
    image_id = ground_truth['image_id']
    id2label = model.id2label
    pixel_values = dataset[0][0].unsqueeze(0).to(device)

    ground_truth = {
        'bbox': ground_truth['bbox'],
        'category': ground_truth['category_id'],
    }

    # Prediction
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=None)

    # Get image
    image = dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(image_dir, image['file_name']))

    # Post process
    results = processor.post_process_object_detection(outputs,
                                                    target_sizes=[image.size],
                                                    threshold=threshold)[0]
    
    # Visualize
    plot_results(image, results, ground_truth, id2label)

