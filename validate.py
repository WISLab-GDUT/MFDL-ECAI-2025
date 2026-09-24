import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score

from data import create_dataloader
from networks.MFDL import mfdl
from options.test_options import TestOptions


def validate(model, opt):
    data_loader = create_dataloader(opt)
    print(f"Validation samples: {len(data_loader.dataset)}")

    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            outputs = model(in_tens)
            outputs = outputs.sigmoid().flatten()
            y_pred.extend(outputs.cpu().numpy().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = mfdl(num_classes=1)
    model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)
    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
