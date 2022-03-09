from backbone import *
from neck import *
from head import *
from yololoss import *


class Yolox(nn.Module):
    def __init__(self, num_cls, training=False):
        super(Yolox, self).__init__()
        self.training = training
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(num_classes=num_cls)
        self.criteria = YOLOLoss(num_classes=num_cls)

    def forward(self, img_tensor, target=None):
        backbone_output = self.backbone(img_tensor)
        neck_output = self.neck(backbone_output)
        head_output = self.head(neck_output)
        if self.training:
            return self.criteria(head_output, labels=target)
        else:
            return head_output