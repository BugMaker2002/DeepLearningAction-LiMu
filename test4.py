import torch
from d2l import torch as d2l

boxes = torch.tensor([[0.1000, 0.0800, 0.5200, 0.9200],
                             [0.0800, 0.2000, 0.5600, 0.9500],
                             [0.1500, 0.3000, 0.6200, 0.9100],
                             [0.5500, 0.2000, 0.9000, 0.8800]])
scores = torch.tensor([0.9000, 0.8000, 0.7000, 0.9000])
iou_threshold = 0.6

B = torch.argsort(scores, dim=-1, descending=True)
print(B)
keep = []  # 保留预测边界框的指标
step=1
while B.numel() > 0:
    print(f"第{step}轮")
    i = B[0]
    keep.append(i)
    if B.numel() == 1: break
    # print(i)
    print(boxes[i, :].reshape(-1, 4))
    print(boxes[B[1:], :].reshape(-1, 4))
    # print(B[1:])
    iou = d2l.box_iou(boxes[i, :].reshape(-1, 4),
                  boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
    print(iou)
    inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
    print(inds)
    B = B[inds + 1]
    print(B)
    step+=1
print("最终结果", torch.tensor(keep, device=boxes.device))
