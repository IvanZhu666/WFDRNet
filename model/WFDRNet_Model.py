import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.FDB_Model import FDB
from model.FRB_Model import FRB
from model.AMG_Model import AMG

def l2_normalize(input_tensor, dim=1, eps=1e-12):
    denom = torch.sqrt(torch.sum(input_tensor ** 2, dim=dim, keepdim=True))
    return input_tensor / (denom + eps)


class TeacherNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: [batch, 3, 256, 256]
          x1: [batch, 64, 64, 64]
          x2: [batch, 128, 32, 32]
          x3: [batch, 256, 16, 16]
          x4: [batch, 512, 8, 8]
        """
        self.eval()
        x1, x2, x3, x4 = self.encoder(x)
        return (x1, x2, x3, x4)


class WFDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = TeacherNet()
        self.frb = FRB()
        self.fdb = FDB()
        self.amg = AMG()

    def forward(self, anomalous_img, origin_img=None):
        self.teacher.eval()

        if origin_img is None:
            origin_img = anomalous_img.clone()

        outputs_r, inputs_d = self.frb(anomalous_img)
        outputs_frb = [l2_normalize(o) for o in outputs_r]

        fdb_outputs = self.fdb(inputs_d)
        outputs_fdb = [l2_normalize(o) for o in fdb_outputs]

        teacher_anom_feats = self.teacher(anomalous_img)
        teacher_anom_feats = [l2_normalize(t.detach()) for t in teacher_anom_feats]

        output_tr, output_td = [], []
        for teacher_feat, fdb_feat, frb_feat in zip(teacher_anom_feats, outputs_fdb, outputs_frb):
            output_td.append(-teacher_feat * fdb_feat)
            output_tr.append(-teacher_feat * frb_feat)

        predicted_mask = self.amg(output_tr, output_td)

        teacher_orig_feats = self.teacher(origin_img)
        teacher_orig_feats = [l2_normalize(t.detach()) for t in teacher_orig_feats]

        output_td_list = []
        output_tr_list = []
        output_rd_list = []
        for teacher_feat, fdb_feat, frb_feat in zip(teacher_orig_feats, outputs_fdb, outputs_frb):
            a_map_td = 1 - torch.sum(fdb_feat * teacher_feat, dim=1, keepdim=True)
            output_td_list.append(a_map_td)
            a_map_tr = 1 - torch.sum(frb_feat * teacher_feat, dim=1, keepdim=True)
            output_tr_list.append(a_map_tr)
            a_map_rd = 1 - torch.sum(frb_feat * fdb_feat, dim=1, keepdim=True)
            output_rd_list.append(a_map_rd)

        output_first = []
        output_mask = []
        for td, rd, tr in zip(output_td_list, output_rd_list, output_tr_list):
            first_total = (td + rd + tr) / 3
            first_ = (td + tr) / 2
            output_first.append(first_total)
            output_mask.append(first_)

        first_mask = torch.cat(
            [
                F.interpolate(
                    output_m,
                    size=output_mask[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_m in output_mask
            ],
            dim=1,
        )

        first_mask = torch.prod(first_mask, dim=1, keepdim=True)

        return predicted_mask, output_first, first_mask

