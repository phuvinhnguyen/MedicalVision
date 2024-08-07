import torch
from PIL import Image
import os
import matplotlib.pyplot as plt

def test_and_visualize_model(
    val_dataset, model, processor, image_idx=1, image_dir=None, 
    threshold=0.5, device='cuda', colors=None, figsize=(16, 10)
):
    """
    Test the model on a specific image from the validation dataset and visualize the results.

    Args:
    - val_dataset: Dataset object containing the validation images and annotations.
    - model: The object detection model to be evaluated.
    - processor: Processor object for post-processing model outputs.
    - image_idx: Index of the image to visualize in the validation dataset.
    - image_dir: Directory where the images are stored (optional).
    - threshold: Confidence threshold for displaying bounding boxes.
    - device: Device to run the model on (e.g., 'cuda' or 'cpu').
    - colors: List of colors for bounding boxes (optional).
    - figsize: Tuple specifying the figure size for visualization.

    Returns:
    - None
    """
    if colors is None:
        # Default colors for visualization
        colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                  [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    # Load the image and corresponding target
    pixel_values, target = val_dataset[image_idx]
    print(target)
    pixel_values = pixel_values.unsqueeze(0).to(device)
    model = model.to(device)
    print(f"Pixel values shape: {pixel_values.shape}")

    # Forward pass to get class logits and bounding boxes
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=None)
    print("Outputs:", outputs.keys())

    def plot_results(pil_img, scores, labels, boxes):
        plt.figure(figsize=figsize)
        plt.imshow(pil_img)
        ax = plt.gca()
        color_cycle = colors * 100
        for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), color_cycle):
            if score >= threshold:
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, color=c, linewidth=3))
                text = f'{model.config.id2label[label]}: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

    # Load the image based on ID
    image_id = target['image_id'].item()
    image_info = val_dataset.coco.loadImgs(image_id)[0]
    image_path = os.path.join(image_dir, image_info['file_name']) if image_dir else image_info['file_name']
    image = Image.open(image_path)

    # Post-process model outputs
    width, height = image.size
    postprocessed_outputs = processor.post_process_object_detection(
        outputs, target_sizes=[(height, width)], threshold=threshold
    )
    results = postprocessed_outputs[0]

    # Plot the results
    print(results)
    plot_results(image, results['scores'], results['labels'], results['boxes'])

# Example usage:
# test_and_visualize_model(val_dataset, model, processor, image_idx=1, image_dir='/path/to/images', threshold=0.9, device='cuda')
