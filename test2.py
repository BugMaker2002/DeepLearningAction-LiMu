import torch
from d2l import torch as d2l

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = d2l.box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full((num_anchors, ), -1, dtype=torch.long, device=device)
    # print(jaccard)
    max_ious, indices = torch.max(jaccard, dim=1)

    print(max_ious)
    # print(indices)

    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]

    print(anc_i)
    # print(box_j)

    anchors_bbox_map[anc_i] = box_j

    # print(anchors_bbox_map)

    col_discard = torch.full((num_anchors, ), -1)
    row_discard = torch.full((num_gt_boxes, ), -1)

    for _ in range(num_gt_boxes):
        max_id = torch.argmax(jaccard)
        box_id = (max_id % num_gt_boxes).long()
        anc_id = (max_id / num_gt_boxes).long()
        anchors_bbox_map[anc_id] = box_id
        jaccard[:, box_id] = col_discard
        jaccard[anc_id, :] = row_discard
    return anchors_bbox_map

ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])
anchors, labels = anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0)

batch_size, anchors = labels.shape[0], anchors.squeeze(0)


batch_offset, batch_mask, batch_class_labels = [], [], []
device, num_anchors = anchors.device, anchors.shape[0]
for i in range(batch_size):
    label = labels[i, :, :]
    anchors_bbox_map = assign_anchor_to_bbox(
        label[:, 1:], anchors, device)
    print(anchors_bbox_map)
    bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
        1, 4)
    print((anchors_bbox_map >= 0).float())
    print(bbox_mask)
    # 将类标签和分配的边界框坐标初始化为零
    class_labels = torch.zeros(num_anchors, dtype=torch.long,
                               device=device)
    assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                              device=device)
    # 使用真实边界框来标记锚框的类别。
    # 如果一个锚框没有被分配，标记其为背景（值为零）
    indices_true = torch.nonzero(anchors_bbox_map >= 0)
    bb_idx = anchors_bbox_map[indices_true]
    print(indices_true)
    print(bb_idx)
    class_labels[indices_true] = label[bb_idx, 0].long() + 1
    print(class_labels)
    assigned_bb[indices_true] = label[bb_idx, 1:]
    print(assigned_bb)
    # 偏移量转换
    offset = d2l.offset_boxes(anchors, assigned_bb) * bbox_mask
    print(offset.shape)
    batch_offset.append(offset.reshape(-1))
    batch_mask.append(bbox_mask.reshape(-1))
    batch_class_labels.append(class_labels)
print(batch_offset)
bbox_offset = torch.stack(batch_offset)
print(bbox_offset)
bbox_mask = torch.stack(batch_mask)
class_labels = torch.stack(batch_class_labels)
print(class_labels)

