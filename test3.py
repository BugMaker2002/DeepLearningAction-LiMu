import torch
from d2l import torch as d2l

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
cls_probs, offset_preds, anchors, nms_threshold = cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(
    dim=0), anchors.unsqueeze(dim=0), 0.5
pos_threshold = 0.009999999

device, batch_size = cls_probs.device, cls_probs.shape[0]
anchors = anchors.squeeze(0)
num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
out = []
for i in range(batch_size):
    cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
    conf, class_id = torch.max(cls_prob[1:], 0)
    predicted_bb = d2l.offset_inverse(anchors, offset_pred)
    '''
    predicted_bb: tensor([[0.1000, 0.0800, 0.5200, 0.9200],
        [0.0800, 0.2000, 0.5600, 0.9500],
        [0.1500, 0.3000, 0.6200, 0.9100],
        [0.5500, 0.2000, 0.9000, 0.8800]])
    conf: tensor([0.9000, 0.8000, 0.7000, 0.9000])
    keep: tensor([0, 3])
    '''
    # print(predicted_bb)
    # print(conf)
    keep = d2l.nms(predicted_bb, conf, nms_threshold)
    # print(keep)
    # 找到所有的non_keep索引，并将类设置为背景
    all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
    combined = torch.cat((keep, all_idx))  # combined: tensor([0, 3, 0, 1, 2, 3])
    # print(combined)
    uniques, counts = combined.unique(return_counts=True)  # tensor([0, 1, 2, 3]), tensor([2, 1, 1, 2])
    # print(uniques, counts)
    non_keep = uniques[counts == 1]
    all_id_sorted = torch.cat((keep, non_keep))  # class_id: tensor([0, 0, 0, 1])
    # all_id_sorted: tensor([0, 3, 1, 2])
    class_id[non_keep] = -1
    class_id = class_id[all_id_sorted]  # class_id: tensor([ 0,  1, -1, -1])
    conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
    # pos_threshold是一个用于非背景预测的阈值
    below_min_idx = (conf < pos_threshold)
    class_id[below_min_idx] = -1
    conf[below_min_idx] = 1 - conf[below_min_idx]
    print(class_id)
    print(conf)
    print(predicted_bb)
    pred_info = torch.cat((class_id.unsqueeze(1),
                           conf.unsqueeze(1),
                           predicted_bb), dim=1)
    out.append(pred_info)
# print(torch.stack(out))
