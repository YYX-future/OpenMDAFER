from torch import nn
from models.Prompt_Learner import *
from loss.mmd_0 import mmd_loss
from pseudo.pseudo import get_ps_label_acc
import torch.nn.functional as F
from models.Resnet import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets_i):

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, (targets_i.to(torch.int64)).unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.to(device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class ConLoss(nn.Module):
    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
        输出:
            loss值
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''
        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator) + 1e-4  # epsilon = 1e-4, used to get rid of inifity gradient
        if torch.any(torch.isnan(log_probs)):
            log_probs = 0
            # raise ValueError("Log_prob has nan!")
            return 0
        else:
            log_probs = torch.sum(
                log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                            num_positives_per_row > 0]
            '''
            计算正样本平均的log-likelihood
            考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
            所以这里只计算正样本个数>0的    
            '''
            # loss
            loss = -log_probs
            if self.scale_by_temperature:
                loss *= self.temperature
            loss = loss.mean()
            if torch.isnan(loss):
                loss = 0
            return loss


class GenerateModel(nn.Module):
    def __init__(self, input_text, class_name, clip_model, args):
        super().__init__()
        self.args = args
        self.input_text = input_text
        self.class_name = class_name
        self.prompt_learner = PromptLearner(input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.un_text_encoder = clip_model.encode_text
        self.dtype = clip_model.dtype
        if args.backbone == "18":
            self.image_encoder = resnet18(True)
        elif args.backbone == "50":
            self.image_encoder = resnet50(True)
        elif args.backbone == "101":
            self.image_encoder = resnet101(True)
        else:
            self.image_encoder = clip_model.visual
        self.clip_model_ = clip_model

        self.l_smooth = CrossEntropyLabelSmooth(args.num_classes).to(device)
        self.l_con = ConLoss().to(device)
        self.conv = nn.Linear(2048, 512)

    def text_semantic_opposite_loss(self, text_features, text_features_no, mode="L2"):
        if mode == "L2":
            l2_distance = 2 - 2 * (text_features * text_features_no).sum(-1) + 1e-4
            # epsilon = 1e-4, used to get rid of inifity gradient
            loss = 2 - torch.sqrt(l2_distance)  # \in [0,2]
        if mode == "cosine":
            loss = (text_features * text_features_no).sum(-1) + 1.0  # \in [0,2]

        return loss.mean()

    def image_text_binary_opposite_loss(self, logits_per_image_yes_no, eyes):

        N = logits_per_image_yes_no.shape[0]
        binary_yes_no = eyes * logits_per_image_yes_no[:, :, 0] + (1 - eyes) * logits_per_image_yes_no[:, :, 1]
        loss_bin = - torch.log(binary_yes_no)
        loss_bin_no = ((1 - eyes) * loss_bin).view(-1).sum() / (N ** 2 - N)

        return loss_bin_no


    def forward(self, src_img, tgt_img, src_label, tgt_label):

        un_cls_l, cls_l, un_tgt_l, un_ps_l_acc, opposite_l, con_l = 0, 0, 0, 0, 0, 0
        tgt_l, ps_l_acc = 0, 0
        align_l = 0
        if self.training:

            ################# Visual Part #################
            src_features = self.image_encoder(src_img.type(self.dtype), self.args.drop_rate, self.args.drop)
            tgt_features = self.image_encoder(tgt_img.type(self.dtype), self.args.drop_rate, self.args.drop)

            src_features = src_features / src_features.norm(dim=-1, keepdim=True)
            tgt_features = tgt_features / tgt_features.norm(dim=-1, keepdim=True)

            ################# Text Part #################
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            ################# Pred #################
            if self.args.margin:
                src_features -= self.args.lam * text_features[src_label]

            src_pred = src_features @ text_features.t() / 0.01
            tgt_pred = tgt_features @ text_features.t() / 0.01

            ################# loss #################
            if self.args.align:
                align_l = mmd_loss(src_features, tgt_features)

            k_cls_l = self.l_smooth(src_pred, src_label)

            if self.args.tgt:
                ps_l_acc, __, __, tgt_l = get_ps_label_acc(tgt_pred, self.args.threshold, tgt_label)

            if self.args.con:
                con_l = self.l_con(src_features, src_label) + self.l_con(tgt_features)

            if self.args.negative:

                unclass_text_prompt = "This expression is not "
                unclass_text = [unclass_text_prompt + word + "." for word in self.class_name]
                unclass_text_token = clip.tokenize(unclass_text).to(device)
                un_text_features = self.un_text_encoder(unclass_text_token)
                un_text_features = un_text_features / un_text_features.norm(dim=-1, keepdim=True)

                un_src_pred = src_features @ un_text_features.t() / 0.01
                un_tgt_pred = tgt_features @ un_text_features.t() / 0.01
                un_cls_l = self.l_smooth(un_src_pred, src_label)

                if self.args.avg:
                    votes = torch.stack([src_pred, un_src_pred])
                    cls_l = self.l_smooth(votes.mean(dim=0), src_label)

                if self.args.tgt:
                    un_ps_l_acc, __, __, un_tgt_l = get_ps_label_acc(un_tgt_pred, self.args.threshold, tgt_label)

                if self.args.opposite:
                    opposite_l = self.text_semantic_opposite_loss(text_features, un_text_features)

            return k_cls_l, un_cls_l, cls_l, align_l, tgt_l, ps_l_acc, un_tgt_l, un_ps_l_acc, opposite_l, con_l

        else:

            ################# Visual Part #################
            tgt_features = self.image_encoder(src_img.type(self.dtype))
            tgt_features = tgt_features / tgt_features.norm(dim=-1, keepdim=True)

            ################# Text Part #################
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            tgt_pre_vl = tgt_features @ text_features.t() / 0.01

            if self.args.negative:
                unclass_text_prompt = "This expression is not "
                unclass_text = [unclass_text_prompt + word + "." for word in self.class_name]
                unclass_text_token = clip.tokenize(unclass_text).to(device)
                un_text_features = self.un_text_encoder(unclass_text_token)
                un_text_features = un_text_features / un_text_features.norm(dim=-1, keepdim=True)

                un_tgt_pre_vl = tgt_features @ un_text_features.t() / 0.01

                return tgt_pre_vl, un_tgt_pre_vl

            else:
                return tgt_pre_vl, tgt_pre_vl
