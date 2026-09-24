from networks.MFDL import mfdl
from transformers import CLIPVisionConfig
from accelerate import Accelerator

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import argparse
from torchmetrics.classification import BinaryAUROC,BinaryAccuracy,BinaryAveragePrecision,BinaryF1Score
# There is your test_image_path
from testdataloader import image_dataloader
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置只使用 GPU 0 和 GPU 1





def test():
    image_list = []
    results = []
    root_path = '/root/xzl/NewData/ForenSynths_4classtrain_val_test'
    model_path = '/root/xzl/modelpthsave/MFDL63.pth'
    csv_dir = './results'
    os.makedirs(csv_dir, exist_ok=True)

    # 设置要测试的增强类型
    settings = {
        'plain': {'apply_jpeg': False, 'apply_blur': False},
        # 'jpeg40': {'apply_jpeg': True, 'jpeg_quality': 40, 'apply_blur': False},
        # 'jpeg60': {'apply_jpeg': True, 'jpeg_quality': 60, 'apply_blur': False},
        # 'blur': {'apply_jpeg': False, 'apply_blur': True, 'blur_radius': 1.5},
    }

    for mode, kwargs in settings.items():
        print(f'\n=== Running Mode: {mode} ===')
        results = []

        # 加载模型
        with torch.no_grad():
            accelerator = Accelerator()
            device = accelerator.device
            acc_metric = BinaryAccuracy().to(device)
            auc_metric = BinaryAUROC().to(device)
            ap_metric = BinaryAveragePrecision().to(device)
            f1_metric = BinaryF1Score().to(device)
            model = mfdl(num_classes=1)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            for path in os.listdir(root_path):
                acc_metric.reset()
                auc_metric.reset()
                ap_metric.reset()
                f1_metric.reset()

                dataloader = image_dataloader(path=os.path.join(root_path, path), **kwargs)
                model, dataloader = accelerator.prepare(model, dataloader)

                for batch in dataloader:
                    if batch is None:
                        continue
                    image, labels = batch
                    outputs = model(image)
                    outputs=outputs.squeeze(1)
                    # print(outputs.shape)
                    # print(labels.shape)
                    probs = torch.nn.functional.sigmoid(outputs)
                    preds = probs
                    acc_metric(preds, labels)
                    auc_metric(preds, labels)
                    ap_metric(preds, labels)
                    f1_metric(preds, labels)

                accuracy = acc_metric.compute().item()
                auc = auc_metric.compute().item()
                ap = ap_metric.compute().item()
                f1 = f1_metric.compute().item()
                results.append({
                    'path': path,
                    'accuracy': accuracy,
                    'auc': auc,
                    'ap': ap,
                    'F1': f1,
                })

            csv_file = os.path.join(csv_dir, f'{mode}_results.csv')
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['path', 'accuracy', 'auc', 'ap', 'F1'])
                writer.writeheader()
                writer.writerows(results)
            print(f'Saved: {csv_file}')

test()

