import torch
import torch.nn as nn
from utils import IoU

# parameterize number of classes, nc=20 default

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        
        self.mse = nn.MSELoss(reduction="sum")
        
        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, target):
        # reshape flattened prediction
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # compute IoU given bbox coordinates
        iou_b1 = IoU(predictions[..., 21:25], target[..., 21:25] )
        iou_b2 = IoU(predictions[..., 26:30], target[..., 26:30] )
        ious = torch.cat( [ iou_b1.unsqueeze(0), iou_b2.unsqueeze(0) ], dim=0 )
        iou_maxes, bestbox = torch.max(ious, dim=0)

        obj_exists = target[..., 20].unsqueeze(3) # 11obj_i, confidence

        ## Compute BBox Loss
        
        box_predictions = obj_exists * (
            (  # best box 0 or 1, thus only considers boxprops of best box of the 2
               # todo: rsm: generalize to B boxes
                bestbox * predictions[..., 26:30] + (1 - bestbox)* predictions[..., 21:25]
            )
        )

        box_targets = obj_exists * target[..., 21:25]
        
        box_predictions[..., 2:4] =  torch.sign(box_predictions[...,2:4])*torch.sqrt(
            torch.abs( box_predictions[...,2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[...,2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        ## Compute Object Loss
        pred_box = (
            bestbox* predictions[..., 25:26] + (1 - bestbox)*predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten( obj_exists * pred_box ),
            torch.flatten( obj_exists * target[...,20:21] )
        )

        ## Compute NoObject Loss        

        noobject_loss = self.mse(
            torch.flatten( (1-obj_exists) * predictions[...,20:21] ),
            torch.flatten( (1-obj_exists) * target[...,20:21] )
        )
        noobject_loss += self.mse(
            torch.flatten( (1-obj_exists) * predictions[...,25:26] ),
            torch.flatten( (1-obj_exists) * target[...,20:21] )
        )

        ## Compute Multi-Classification Loss
        mclass_loss = self.mse(
            torch.flatten( obj_exists*predictions[...,:20], end_dim=-2 ), # todo: rsm: generalize to C classes
            torch.flatten( obj_exists*target[...,:20] , end_dim=-2), # todo: rsm: generalize to C classes
        )

        loss = self.lambda_coord*box_loss + object_loss + self.lambda_noobj*noobject_loss + mclass_loss

        return loss



