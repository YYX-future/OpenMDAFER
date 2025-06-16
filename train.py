import torch
import torch.nn.functional
from torch.autograd import Variable
import os
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from data import data_prepare
from models.mymodel import GenerateModel
from models.clip import clip
from unknown import unknown
from utils import setup_seed
import numpy as np
from sklearn.metrics import roc_auc_score
from models.Text import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_parse_option():
    parser = argparse.ArgumentParser('training and evaluation script', add_help=False)

    parser.add_argument('--seed', type=int, default=3407)  # 10
    parser.add_argument('--devices', type=str, default='0', help='Set the CUDA_VISIBLE_DEVICES var from this string')

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr-image-encoder', type=float, default=1e-5)  # 1e-5
    parser.add_argument('--lr-prompt-learner', type=float, default=1e-3)  # 1e-3
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--milestones', nargs='+', type=int, default=[10, 20, 30, 40],
                        help='List of milestone values. Default is [30, 40]')

    parser.add_argument('--batch_size', type=int, default=256, help="batch size for single GPU")  # 100
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print-freq', type=int, default=10)

    parser.add_argument('--src_domain', type=str, action='append', default=['FER2013', 'RAF', 'Oulu', 'Aff', ])
    parser.add_argument('--tgt_domain', type=str, default="RAF")
    parser.add_argument('--num_classes', type=int, default=7)

    parser.add_argument('--checkpoint_dir', type=str, default=r"./checkpoint")
    parser.add_argument('--last_model_path', type=str, default=None)
    parser.add_argument('--record_folder', type=str, default=r"./records")

    parser.add_argument('--text_type', type=str, default="class_descriptor")
    parser.add_argument('--class-token-position', type=str, default="end")
    parser.add_argument('--class-specific-contexts', action='store_true')
    parser.add_argument('--load_and_tune_prompt_learner', action='store_false')
    parser.add_argument('--drop', action='store_false')

    parser.add_argument('--align', action='store_false')
    parser.add_argument('--margin', action='store_false')
    parser.add_argument('--negative', action='store_false')
    parser.add_argument('--opposite', action='store_false')
    parser.add_argument('--con', action='store_false')
    parser.add_argument('--tgt', action='store_false')
    parser.add_argument('--bin', action='store_false')
    parser.add_argument('--avg', action='store_false')
    parser.add_argument('--hyper', action='store_true')

    parser.add_argument('--contexts_number', type=int, default=8, help="hyper parameter of prompts length")
    parser.add_argument('--hyper_align', type=float, default=0.2, help="hyper parameter of align")
    parser.add_argument('--hyper_tgt', type=float, default=0.1, help="hyper parameter of tgt ps")
    parser.add_argument('--hyper_con', type=float, default=0.6, help="hyper parameter of contrastive")
    parser.add_argument('--hyper_opposite', type=float, default=0.3, help="hyper parameter of opposite")
    parser.add_argument('--threshold', type=float, default=0.8, help="pseudo label threshold")
    parser.add_argument('--lam', type=float, default=0.3, help="hyper parameter of margin")
    parser.add_argument('--drop_rate', type=float, default=0.3, help="hyper parameter of drop out")
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--text', type=str, default="ViT-B/32")

    parser.add_argument('--start_class', type=int, default=0, help="start class")
    parser.add_argument('--end_class', type=int, default=7, help="end class")

    args, unparsed = parser.parse_known_args()
    return args


def main(model, record_r, record_l, record_c, checkpoint, i, num_classes):

    setup_seed(args.seed)

    root = r'./data/'
    open_class = i
    # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

    if args.tgt_domain in ["JAFFE", "CK", "Oulu"]:
        src_loaders, tgt_train_dl, tgt_test_dl = data_prepare.get_loaders_epoch(root, args, open_class, False)

    else:
        src_loaders, tgt_train_dl, tgt_test_dl = data_prepare.get_loaders_epoch(root, args, open_class, False)

    src_domains = args.src_domain.copy()
    src_domains.remove(args.tgt_domain)
    src_acc, best_tgt_acc, best_epoch, start_epoch = [[0]] * 2, 0, 0, 0
    best_tgt_close_acc, best_tgt_auc, best_tgt_h = 0, 0, 0
    best_fpr = 1
    if args.last_model_path is not None:
        model_checkpoint = torch.load(args.last_model_path, map_location=device)
        start_epoch = model_checkpoint['epoch']

    # only open learnable part
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = True
        if "prompt_learner" in name:
            param.requires_grad = True
        # if "conv" in name:
        #     param.requires_grad = True
    # model = torch.nn.DataParallel(model).to(device)
    # print params
    # print('************************')
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # print('************************')

    # define optimizer
    optimizer = torch.optim.SGD([{"params": model.image_encoder.parameters(), "lr": args.lr_image_encoder},
                                 {"params": model.prompt_learner.parameters(), "lr": args.lr_prompt_learner}],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # define scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    for epoch in tqdm(range(0, args.epochs)):

        train(src_loaders, tgt_train_dl, model, optimizer, args, record_l)

        acc, close_acc, h, auc, N_correct, N_total = test(model, tgt_test_dl, num_classes, open_class)

        scheduler.step()

        if auc > best_tgt_auc:
            best_tgt_auc = auc
            best_epoch = epoch
            # best_acc_model_path = '%s/%s_%s_%s.pt' % (
            #     checkpoint_path, target, 'best', str(start_steps))
            # torch.save(model_checkpoint, best_acc_model_path)
        best_tgt_h = max(best_tgt_h, max(h))
        best_tgt_acc = max(best_tgt_acc, max(acc))
        best_tgt_close_acc = max(best_tgt_close_acc, max(close_acc))

        f1 = open(record_r, mode='a+')
        f1.writelines(
            ['epoch now: ', str(epoch), ' ', 'Open class: ', str(open_class), ' ',
             'best test auc: {:.2f} '.format(float(best_tgt_auc)), ' ',
             'best test h-score: {:.2f} '.format(float(best_tgt_h)), ' ',
             'best test acc: {:.2f} '.format(float(best_tgt_acc)), ' ',
             'best test close acc: {:.2f} '.format(float(best_tgt_close_acc)), ' ',
             # 'best fprBase: {:.2f} '.format(float(best_fpr)), ' ',
             'best epoch: ', str(best_epoch), '\n',
             'test auc now: {:.2f} '.format(float(auc)), ' ',
             'test h-score now: {:.2f} '.format(float(max(h))), ' ',
             'test acc now: {:.2f} '.format(float(max(acc))), ' ',
             'test close acc now: {:.2f} '.format(float(max(close_acc))), '\n',
             'test h_pos: {:.2f} '.format(float(h[0])), ' ',
             'test h_neg: {:.2f} '.format(float(h[1])), ' ',
             'test h_avg: {:.2f} '.format(float(h[2])), ' ',
             'test acc_pos: {:.2f} '.format(float(acc[0])), ' ',
             'test acc_neg: {:.2f} '.format(float(acc[1])), ' ',
             'test acc_avg: {:.2f} '.format(float(acc[2])), '\n',
             '{:.2f} {:.2f} {:.2f} {:.2f}'.format(best_tgt_auc, best_tgt_h, best_tgt_acc, best_tgt_close_acc), ' ',
             # 'test fprBase: {:.2f} '.format(float(fprBase)), ' ',
             '\n\n',
             ])

        f1.close()

        f2 = open(record_c, mode='a+')
        f2.writelines(
            ['correct for each class: ', str(float(N_correct[0])), ' ',
             str(float(N_correct[1])), ' ', str(float(N_correct[2])), ' ',
             str(float(N_correct[3])), ' ', str(float(N_correct[4])), ' ',
             str(float(N_correct[5])), ' ', str(float(N_correct[6])), ' ',
             'total for each class: ', str(float(N_total[0])), ' ',
             str(float(N_total[1])), ' ', str(float(N_total[2])), ' ',
             str(float(N_total[3])), ' ', str(float(N_total[4])), ' ',
             str(float(N_total[5])), ' ', str(float(N_total[6])), '\n\n',
             ])
        f2.close()


def train(src_loaders, tgt_train_dl, model, optimizer, args, record_l):

    for i, (((src_data, src_label), d_idx), (tgt_data, tgt_label)) in enumerate(zip(src_loaders, tgt_train_dl)):

        model.train()

        src_data, src_label = src_data.to(device), src_label.to(device)
        tgt_data, tgt_label = tgt_data.to(device), tgt_label.to(device)

        k_cls_l, un_cls_l, cls_l, align_l, tgt_l, ps_l_acc, un_tgt_l, un_ps_l_acc, opposite_l, con_l \
            = model(src_data, tgt_data, src_label, tgt_label)

        loss = k_cls_l + un_cls_l + cls_l + align_l * args.hyper_align + (tgt_l + un_tgt_l) * args.hyper_tgt \
               + opposite_l * args.hyper_opposite + con_l * args.hyper_con

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(
                'iter {}: Train target {} total: {:.2f} k_cls_Loss: {:.2f} un_cls_Loss: {:.2f} cls_Loss: {:.2f} '
                'align_Loss: {:.2f} tgt_Loss_l: {:.2f} un_tgt_Loss_l: {:.2f} tgt_lb_acc: {:.2f} un_tgt_lb_acc: {:.2f} '
                'opposite_loss: {:.2f} con_loss: {:.2f}'.format(
                    i, args.tgt_domain, loss.item(), k_cls_l.item(), un_cls_l, cls_l,
                    align_l * args.hyper_align, tgt_l * args.hyper_tgt, un_tgt_l * args.hyper_tgt,
                    ps_l_acc, un_ps_l_acc, opposite_l * args.hyper_opposite, con_l * args.hyper_con))

        if i % (args.print_freq * 5) == 0:
            f = open(record_l, mode='a+')
            f.writelines(
                [
                    'loss: ', f"{float(loss.item()):.2f}", ' ',
                    'k_cls: ', f"{float(k_cls_l.item()):.2f}", ' ',
                    'un_cls: ', f"{float(un_cls_l):.2f}", ' ',
                    'cls: ', f"{float(cls_l):.2f}", ' ',
                    'align_l: ', f"{float(align_l * args.hyper_align):.2f}", ' ',
                    'k_tgt: ', f"{float(tgt_l * args.hyper_tgt):.2f}", ' ',
                    'un_tgt: ', f"{float(un_tgt_l * args.hyper_tgt):.2f}", ' ',
                    'ps_acc: ', f"{float(ps_l_acc):.2f}", ' ',
                    'un_ps_acc: ', f"{float(un_ps_l_acc):.2f}", ' ',
                    'opposite_l: ', f"{float(opposite_l * args.hyper_opposite):.2f}", ' ',
                    'con_l: ', f"{float(con_l * args.hyper_con):.2f}", '\n\n',
                ]
            )


def test(model, tgt_test_dl, num_classes, open_class):
    model.eval()

    # 记录H-score
    all_confidence, all_labels = [], []
    p_label1, p_label2, p_label3 = [], [], []
    sft_scores = []
    pro_confidence = []

    with torch.no_grad():
        for (data, tgt_label), d_inx in tgt_test_dl:
        # for data, tgt_label in tgt_test_dl:
            if torch.cuda.is_available():
                data, tgt_label = data.to(device), tgt_label.to(device)
            data, tgt_label = Variable(data), Variable(tgt_label)

            pred1, pred_2 = model(data, None, None, None)
            pred_avg = (pred1 + pred_2) / 2
            confidence, p_label_1, p_label_2, p_label_3 = unknown.get_confidence_three_prediction(pred1,
                                                                                                  pred_2, pred_avg)
            all_confidence.extend(confidence)
            p_label1.extend(p_label_1)
            p_label2.extend(p_label_2)
            p_label3.extend(p_label_3)
            all_labels.extend(tgt_label)

            pred_avg = torch.softmax(pred_avg, dim=1)
            sft_scores.append(pred_avg.flatten())
            pro_confidence.extend(confidence)

        all_confidence = unknown.normalize_weight(torch.tensor(all_confidence))
        all_score = all_confidence
        # calculate threshold
        unknown_threshold = unknown.get_threshold(sft_scores)
        print(f'unknown_threshold: {unknown_threshold}')
        # get counters
        counters1, counters2, counters3, known_scores, unknown_scores \
            = unknown.get_three_counters(num_classes, unknown_threshold, p_label1, p_label2, p_label3,
                                         all_labels, all_score, pro_confidence, open_class)

        acc, close_acc, h, auc, N_correct, N_total = \
            validate(counters1, counters2, counters3, known_scores, unknown_scores, open_class)

    return acc, close_acc, h, auc, N_correct, N_total


def validate(counters1, counters2, counters3, known_scores, unknown_scores, open_class):

    # AUC
    counters_list = [counters1, counters2, counters3]
    auc_scores = [unknown.calculate_auc(counters, known_scores, unknown_scores, open_class)
                  for counters in counters_list]
    auc = max(auc_scores)

    # fprBase_list = [unknown.calculate_fprBasec(counters, known_scores, unknown_scores) for counters in counters_list]
    # fprBase = min(fprBase_list)

    # H1
    h1, close_acc1, N_correct1, __ = counters1.h_score(open_class)
    h2, close_acc2, N_correct2, __ = counters2.h_score(open_class)
    h3, close_acc3, N_correct3, N_total = counters3.h_score(open_class)
    acc1 = counters1.mean_accuracy() * 100
    acc2 = counters2.mean_accuracy() * 100
    acc3 = counters3.mean_accuracy() * 100

    max_list = list(map(lambda x: max(x), zip(N_correct1, N_correct2, N_correct3)))
    counters3.add_correct_list(max_list)
    h4, close_acc4, N_correct, __ = counters3.h_score(open_class)
    acc4 = counters3.mean_accuracy() * 100

    acc = [acc1, acc2, acc3, acc4]
    close_acc = [val * 100 for val in [close_acc1, close_acc2, close_acc3, close_acc4]]
    h = [val * 100 for val in [h1, h2, h3, h4]]

    return acc, close_acc, h, auc * 100, N_correct, N_total


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_clip_to_cpu():

    backbone_name = args.text  # "ViT-B/32"  # ViT-B/32
    backbone_path = "./checkpoint"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, backbone_path)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


if __name__ == '__main__':

    global args

    args = get_parse_option()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.record_folder):
        os.mkdir(args.record_folder)

    record_path = os.path.join(args.record_folder, args.tgt_domain)
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    now = datetime.now().strftime("%m-%d-%y_%H%M%S")
    record_result = record_path + "\\" + now + "_" + args.tgt_domain + 'result.txt'
    record_loss = record_path + "\\" + now + "_" + args.tgt_domain + 'loss.txt'
    record_correct = record_path + "\\" + now + "_" + args.tgt_domain + 'correct.txt'
    record_hyper = record_path + "\\" + now + "_" + args.tgt_domain + 'lr.txt'
    checkpoint_path = os.path.join(args.checkpoint_dir, args.tgt_domain, now)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # writer = SummaryWriter(os.path.join("../train_logs", args.tgt_domain, now))

    class_index = 7
    num_classes = args.num_classes - 1

    if args.hyper:
        select_txt = os.path.join(os.getcwd(), 'data', 'hp_search', args.tgt_domain + '_ab.txt')
        print("Parameter search space: ")
        with open(select_txt, 'r') as ff:
            lines = ff.readlines()
        for line in lines:
            hypers = line.strip().split(' ')
            print(hypers)
            # args.hyper_align = float(hypers[0])
            # args.hyper_con = float(hypers[1])
            # args.hyper_opposite = float(hypers[2])
            # args.contexts_number = int(hypers[3])
            # args.threshold = float(hypers[4])
            # 0 0 1 1 1 1 1
            # 1 0 1 1 1 1 1
            # 2 1 1 1 1 1 1
            # 2 1 0 1 1 1 1
            # 2 1 0 0 1 1 1
            # 2 1 0 0 0 0 1
            # 0 0 0 0 0 0 0
            text_type = int(hypers[0])
            if text_type == 0:
                args.text_type = "class_names"
            elif text_type == 1:
                args.text_type = "class_names_with_context"
            else:
                args.text_type = "class_descriptor"

            args.class_specific_contexts = str2bool(hypers[1])
            args.con = str2bool(hypers[2])
            args.align = str2bool(hypers[3])
            args.opposite = str2bool(hypers[4])
            args.tgt = str2bool(hypers[5])
            args.negative = str2bool(hypers[6])
            args.batch_size = int(hypers[7])
            args.hyper_tgt = float(hypers[8])
            # parser.add_argument('--text-type', type=str, default="class_descriptor")
            # parser.add_argument('--class-specific-contexts', action='store_true'))
            # parser.add_argument('--con', action='store_false')
            # parser.add_argument('--opposite', action='store_false')
            # parser.add_argument('--tgt', action='store_false')
            # parser.add_argument('--negative', action='store_false')

            # CLIP_model, _ = clip.load("ViT-B/32", device='cpu')
            CLIP_model = load_clip_to_cpu()
            # CLIP's default precision is fp16
            CLIP_model.float()
            print(args.text_type)
            if args.text_type == "class_names":
                input_text = class_names_7
            elif args.text_type == "class_names_with_context":
                input_text = class_names_with_context_7
            elif args.text_type == "class_descriptor":
                input_text = class_descriptor_7

            model = GenerateModel(input_text=input_text, class_name=class_names_7, clip_model=CLIP_model, args=args)
            model = model.to(device)

            for i in range(args.start_class, args.end_class):
                print(args)
                f1 = open(record_hyper, mode='a+')
                f1.writelines(
                    ['negative: ', str(args.negative), ' ', 'opposite: ', str(args.opposite), ' ',
                     'con: ', str(args.con), ' ', 'tgt: ', str(args.tgt), ' ', 'align: ', str(args.align), ' ',
                     'hyper: ', str(args.hyper), ' ',
                     'class_specific_contexts: ', str(args.class_specific_contexts), ' ',
                     'hyper_align: ', str(args.hyper_align), ' ', 'hyper_tgt: ', str(args.hyper_tgt), ' ',
                     'hyper_con: ', str(args.hyper_con), ' ', 'hyper_opposite: ', str(args.hyper_opposite), ' ',
                     'hyper_margin: ', str(args.lam), ' ', 'threshold: ', str(args.threshold), ' ',
                     'number: ', str(args.contexts_number), ' ', 'text_type: ', str(args.text_type), ' ',
                     '\n\n',
                     ])
                f1.close()
                print(f'num_class open class: {num_classes} {i}')
                main(model, record_result, record_loss, record_correct, checkpoint_path, i, args.num_classes)

    else:
        # CLIP_model, _ = clip.load("ViT-B/32", device='cpu')
        CLIP_model = load_clip_to_cpu()
        # CLIP's default precision is fp16
        CLIP_model.float()

        if args.text_type == "class_names":
            input_text = class_names_7
        elif args.text_type == "class_names_with_context":
            input_text = class_names_with_context_7
        elif args.text_type == "class_descriptor":
            input_text = class_descriptor_7

        model = GenerateModel(input_text=input_text, class_name=class_names_7, clip_model=CLIP_model, args=args)
        model = model.to(device)
        for i in range(args.start_class, args.end_class):
            print(args)
            f1 = open(record_hyper, mode='a+')
            f1.writelines(
                ['negative: ', str(args.negative), ' ', 'opposite: ', str(args.opposite), ' ',
                 'con: ', str(args.con), ' ', 'tgt: ', str(args.tgt), ' ', 'align: ', str(args.align), ' ',
                 'hyper: ', str(args.hyper), ' ',
                 'class_specific_contexts: ', str(args.class_specific_contexts), ' ',
                 'hyper_align: ', str(args.hyper_align), ' ', 'hyper_tgt: ', str(args.hyper_tgt), ' ',
                 'hyper_con: ', str(args.hyper_con), ' ', 'hyper_opposite: ', str(args.hyper_opposite), ' ',
                 'hyper_margin: ', str(args.lam), ' ', 'threshold: ', str(args.threshold), ' ',
                 'number: ', str(args.contexts_number), ' ', 'text_type: ', str(args.text_type), ' ',
                 '\n\n',
                 ])
            f1.close()
            print(f'num_class open class: {num_classes} {i}')
            main(model, record_result, record_loss, record_correct, checkpoint_path, i, args.num_classes)
