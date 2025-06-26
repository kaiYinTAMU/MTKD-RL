import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Policy', 'PolicyTrans']

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(input_size, 128, bias=False),
            nn.ReLU(True),
            nn.Linear(128, output_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.head(input)
        return output 


class PolicyTrans(nn.Module):
    def __init__(self, input_size, teacher_num, dynamic=False):
        super(PolicyTrans, self).__init__()
        self.teacher_num = teacher_num
        
        self.sim_trans = nn.ModuleList([])
        all_input_size = 0
        for idx in range(teacher_num):
            all_input_size = all_input_size + input_size[idx]
        
        self.steam = nn.Sequential(
                nn.Linear(all_input_size, 128, bias=False),
                nn.ReLU())
        self.logit_head = nn.Linear(128, teacher_num, bias=True)
        self.feature_head = nn.Linear(128, teacher_num, bias=True)
                
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dynamic = dynamic
        if dynamic:
            self.logit_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)
            self.feature_weight_factor = torch.nn.Parameter(torch.tensor([1., 1., 1.]), requires_grad=True)

    def forward(self, agent_state):
        teacher_infos, t_ces, t_s_logit_div, t_s_feat_div = agent_state

        weight_loss_t = (1. - F.softmax(t_ces, dim=1)) / (self.teacher_num - 1)
        weight_loss_t_s_logit_div = F.softmax(t_s_logit_div, dim=1)
        weight_loss_t_s_feat_div = F.softmax(t_s_feat_div, dim=1)

        ## teacher_infos的形状是（假设3个teacher），[[B, D_info],[B, D_info], [B, D_info]]
        ## cat后，all_teacher_infos的形状就是[B, D_info*3]，
        ## 也就是3个teacher的stateconcatenate起来输入进policy network中，两个head，第一个得到每个teacher的logit weight, 第二个得到每个teacher的feature weight 
        ##【如果没用feature weight这个行不行？？】
        all_teacher_infos = torch.cat(teacher_infos, dim=1)
        out1 = self.steam(all_teacher_infos)
        logit_weights = self.softmax(self.logit_head(out1))
        feature_weights = self.softmax(self.feature_head(out1))

        ## 这里就是A.1 Methodology中提到的加上了confidence-and divergence-aware weight generation strategies。
        ## 对于这3个weight，可以直接取平均，也可以训练一个权重系数
        if self.dynamic:
            l_f = F.softmax(self.logit_weight_factor, dim=0)
            f_f = F.softmax(self.feature_weight_factor, dim=0)
            all_logit_weights = (l_f[0]*logit_weights + l_f[1]*weight_loss_t + l_f[2]*weight_loss_t_s_logit_div)
            all_feature_weights = (f_f[0] * feature_weights + f_f[1] * weight_loss_t + f_f[2] * weight_loss_t_s_feat_div)
        else:
            all_logit_weights = (logit_weights + weight_loss_t + weight_loss_t_s_logit_div)/ 3.
            all_feature_weights = (feature_weights + weight_loss_t + weight_loss_t_s_feat_div) / 3.


        return all_logit_weights,  all_feature_weights
