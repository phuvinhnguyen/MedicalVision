from transformers import YolosForObjectDetection, YolosFeatureExtractor
from .lightning import lightning_detection
from ...utils.model import change_dropout_rate
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
import numpy as np
from torch import nn

def convert_to_heatmap(grayscale_array, colormap='viridis'):
    normalized_array = (grayscale_array - grayscale_array.min()) / (grayscale_array.max() - grayscale_array.min())
    
    cmap = plt.get_cmap(colormap)
    heatmap_array = cmap(normalized_array)
    
    rgb_heatmap = (heatmap_array[:, :, :3] * 255).astype(np.uint8)
    
    return rgb_heatmap

def get_one_query_meanattn(vis_attn,h_featmap,w_featmap):
    mean_attentions = vis_attn.mean(0).reshape(h_featmap, w_featmap)
    mean_attentions = nn.functional.interpolate(mean_attentions.unsqueeze(0).unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    return mean_attentions

def get_one_query_attn(vis_attn, h_featmap, w_featmap, nh):
    attentions = vis_attn.reshape(nh, h_featmap, w_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[0].cpu().numpy()
    return attentions

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def draw_bbox_in_img(model, im, bbox_scaled, score, color=[0,255,0]):
    tl = 3
    tf = max(tl-1,1)
    for p, (xmin, ymin, xmax, ymax) in zip(score, bbox_scaled.tolist()):
        c1, c2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        cv2.rectangle(im, c1, c2, color, tl, cv2.LINE_AA)
        cl = p.argmax()
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, text, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return im

class Yolos(lightning_detection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _init_model(self, model_name, revision):
        if len(self.id2label) == 0:
            self.model = YolosForObjectDetection.from_pretrained(
                model_name,
                revision=revision,
            )
        else:
            self.model = YolosForObjectDetection.from_pretrained(
                model_name,
                revision=revision,
                num_labels=len(self.id2label),
                ignore_mismatched_sizes=True,
            ).to(self.device)
            self.model.config.id2label = self.id2label
        self.extractor = YolosFeatureExtractor.from_pretrained(
            model_name,
            revision=revision,
        )

        change_dropout_rate(self.model, self.dropout_rate)

    def forward(self, pixel_values, pixel_mask=None):
        """
        Forward pass of the model.
        """
        return self.model(pixel_values=pixel_values)
    
    def common_step(self, batch, batch_idx):
        """
        Common step for training and validation.
        """
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        return outputs.loss, outputs.loss_dict

    def run(
            self,
            image,
            threshold=0.9,
            layers=[0,1,2,3,4,5,6,7,8,9,10,11],
            with_image=False,
            with_bbox=True,
            with_smooth=False,
            ):
        pixel_values = self.extractor(image, return_tensors="pt").pixel_values
        h, w = pixel_values.shape[2:]
        model_output = self.model(pixel_values, output_attentions=True)
        attentions = model_output.attentions

        probas = model_output.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        vis_indexs = torch.nonzero(keep).squeeze(1)

        grad_results = {}

        for vis_index in vis_indexs:
            bbox = rescale_bboxes(model_output.pred_boxes[0, vis_index].unsqueeze(0).cpu(), (w,h))
            score = probas[vis_index].unsqueeze(0)
            w_featmap = pixel_values.shape[3] // self.model.config.patch_size
            h_featmap = pixel_values.shape[2] // self.model.config.patch_size

            for layer in layers:
                attention = attentions[layer].detach().cpu()
                nh = attention.shape[1]
                attention = attention[0, :, -self.model.config.num_detection_tokens:, 1:-self.model.config.num_detection_tokens]

                vis_attn = attention[:, vis_index, :]
                attn = get_one_query_attn(vis_attn, h_featmap, w_featmap, nh)

                for j in range(nh):
                    attn_map = (attn[j] - attn[j].min()) / (attn[j].max() - attn[j].min())
                    attention_normalized = (attn_map * 255).astype(np.uint8)
                    attn_image = convert_to_heatmap(attention_normalized)

                    if with_smooth:
                        attn_image = Image.fromarray(attn_image)
                        attn_image = np.array(attn_image.filter(ImageFilter.GaussianBlur(radius=3)))

                    if with_bbox:
                        attn_image = draw_bbox_in_img(self.model, attn_image, bbox, score, color=[0,0,255])

                    if with_image:
                        attn_image = Image.fromarray(attn_image)
                        attn_image = np.array(Image.blend(image, attn_image.resize(image.size), alpha=0.7))

                    l, la, head = f'l{layer}', f'la{vis_index}', f'h{j}'
                    if l not in grad_results:
                        grad_results[l] = {la: {head: [Image.fromarray(attn_image)]}}
                    elif la not in grad_results[l]:
                        grad_results[l][la] = {head: [Image.fromarray(attn_image)]}
                    elif head not in grad_results[l][la]:
                        grad_results[l][la][head] = [Image.fromarray(attn_image)]
                    else:
                        grad_results[l][la][head].append(Image.fromarray(attn_image))

        return grad_results
