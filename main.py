import os
import torch
from dataset import DAD
import spatial_transforms
from model import generate_model
import argparse
from test import get_normal_vector, split_acc_diff_threshold, cal_score
from utils import adjust_learning_rate, AverageMeter, Logger, get_fusion_label, l2_normalize, post_process, evaluate, \
    get_score
from NCEAverage import NCEAverage
from NCECriterion import NCECriterion
import torch.backends.cudnn as cudnn
from temporal_transforms import TemporalSequentialCrop
from models import resnet, shufflenet, shufflenetv2, mobilenet, mobilenetv2
import ast
import numpy as np
from dataset_test import DAD_Test


def parse_args():
    parser = argparse.ArgumentParser(description='DAD training on Videos')
    parser.add_argument('--root_path', default='', type=str, help='root path of the dataset')
    parser.add_argument('--mode', default='train', type=str, help='train | test(validation)')
    parser.add_argument('--view', default='front_IR', type=str, help='front_depth | front_IR | top_depth | top_IR')
    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of each video clip')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--shortcut_type', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--pre_train_model', default=True, type=ast.literal_eval, help='Whether use pre-trained model')
    parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')
    parser.add_argument('--n_train_batch_size', default=3, type=int, help='Batch Size for normal training data')
    parser.add_argument('--a_train_batch_size', default=25, type=int, help='Batch Size for anormal training data')
    parser.add_argument('--val_batch_size', default=25, type=int, help='Batch Size for validation data')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--epochs', default=250, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_threads', default=8, type=int, help='num of workers loading dataset')
    parser.add_argument('--tracking', default=True, type=ast.literal_eval,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--norm_value', default=255, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--cal_vec_batch_size', default=20, type=int,
                        help='batch size for calculating normal driving average vector.')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='a temperature parameter that controls the concentration level of the distribution of embedded vectors')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--memory_bank_size', default=200, type=int, help='Memory bank size')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--lr_decay', default=100, type=int,
                        help='Number of epochs after which learning rate will be reduced to 1/10 of original value')
    parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--resume_head_path', default='', type=str, help='path of previously trained model head')
    parser.add_argument('--initial_scales', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--scale_step', default=0.9, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--n_scales', default=3, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--train_crop', default='corner', type=str,
                        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--checkpoint_folder', default='./checkpoints/', type=str, help='folder to store checkpoints')
    parser.add_argument('--log_folder', default='./logs/', type=str, help='folder to store log files')
    parser.add_argument('--log_resume', default=False, type=ast.literal_eval, help='True|False: a flag controlling whether to create a new log file')
    parser.add_argument('--normvec_folder', default='./normvec/', type=str, help='folder to store norm vectors')
    parser.add_argument('--score_folder', default='./score/', type=str, help='folder to store scores')
    parser.add_argument('--Z_momentum', default=0.9, help='momentum for normalization constant Z updates')
    parser.add_argument('--groups', default=3, type=int, help='hyper-parameters when using shufflenet')
    parser.add_argument('--width_mult', default=2.0, type=float,
                        help='hyper-parameters when using shufflenet|mobilenet')
    parser.add_argument('--val_step', default=10, type=int, help='validate per val_step epochs')
    parser.add_argument('--downsample', default=2, type=int, help='Downsampling. Select 1 frame out of N')
    parser.add_argument('--save_step', default=10, type=int, help='checkpoint will be saved every save_step epochs')
    parser.add_argument('--n_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--a_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--window_size', default=6, type=int, help='the window size for post-processing')

    args = parser.parse_args()
    return args


def train(train_normal_loader, train_anormal_loader, model, model_head, nce_average, criterion, optimizer, epoch, args,
          batch_logger, epoch_logger, memory_bank=None):
    losses = AverageMeter()
    prob_meter = AverageMeter()

    model.train()
    model_head.train()
    for batch, ((normal_data, idx_n), (anormal_data, idx_a)) in enumerate(
            zip(train_normal_loader, train_anormal_loader)):
        if normal_data.size(0) != args.n_train_batch_size:
            break
        data = torch.cat((normal_data, anormal_data), dim=0)  # n_vec as well as a_vec are all normalized value
        if args.use_cuda:
            data = data.cuda()
            idx_a = idx_a.cuda()
            idx_n = idx_n.cuda()
            normal_data = normal_data.cuda()

        # ================forward====================
        unnormed_vec, normed_vec = model(data)
        vec = model_head(unnormed_vec)
        n_vec = vec[0:args.n_train_batch_size]
        a_vec = vec[args.n_train_batch_size:]
        outs, probs = nce_average(n_vec, a_vec, idx_n, idx_a, normed_vec[0:args.n_train_batch_size])
        loss = criterion(outs)

        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===========update memory bank===============
        model.eval()
        _, n = model(normal_data)
        n = n.detach()
        average = torch.mean(n, dim=0, keepdim=True)
        if len(memory_bank) < args.memory_bank_size:
            memory_bank.append(average)
        else:
            memory_bank.pop(0)
            memory_bank.append(average)
        model.train()

        # ===============update meters ===============
        losses.update(loss.item(), outs.size(0))
        prob_meter.update(probs.item(), outs.size(0))

        # =================logging=====================
        batch_logger.log({
            'epoch': epoch,
            'batch': batch,
            'loss': losses.val,
            'probs': prob_meter.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        print(
            f'Training Process is running: {epoch}/{args.epochs}  | Batch: {batch} | Loss: {losses.val} ({losses.avg}) | Probs: {prob_meter.val} ({prob_meter.avg})')
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'probs': prob_meter.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    return memory_bank, losses.avg


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.exists(args.normvec_folder):
        os.makedirs(args.normvec_folder)
    if not os.path.exists(args.score_folder):
        os.makedirs(args.score_folder)
    torch.manual_seed(args.manual_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manual_seed)
    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening
    args.scales = [args.initial_scales]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
    assert args.train_crop in ['random', 'corner', 'center']
    if args.train_crop == 'random':
        crop_method = spatial_transforms.MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size, crop_positions=['c'])
    before_crop_duration = int(args.sample_duration * args.downsample)

    if args.mode == 'train':
        temporal_transform = TemporalSequentialCrop(before_crop_duration, args.downsample)

        if args.view == 'front_depth' or args.view == 'front_IR':

            spatial_transform = spatial_transforms.Compose([
                crop_method,

                spatial_transforms.RandomRotate(),
                spatial_transforms.SaltImage(),
                spatial_transforms.Dropout(),
                spatial_transforms.ToTensor(args.norm_value),
                spatial_transforms.Normalize([0], [1])
            ])
        elif args.view == 'top_depth' or args.view == 'top_IR':
            spatial_transform = spatial_transforms.Compose([
                spatial_transforms.RandomHorizontalFlip(),
                spatial_transforms.Scale(args.sample_size),
                spatial_transforms.CenterCrop(args.sample_size),

                spatial_transforms.RandomRotate(),
                spatial_transforms.SaltImage(),
                spatial_transforms.Dropout(),
                spatial_transforms.ToTensor(args.norm_value),
                spatial_transforms.Normalize([0], [1])
            ])

        print(
            "=================================Loading Anormal-Driving Training Data!=================================")
        training_anormal_data = DAD(root_path=args.root_path,
                                    subset='train',
                                    view=args.view,
                                    sample_duration=before_crop_duration,
                                    type='anormal',
                                    spatial_transform=spatial_transform,
                                    temporal_transform=temporal_transform
                                    )

        training_anormal_size = int(len(training_anormal_data) * args.a_split_ratio)
        training_anormal_data = torch.utils.data.Subset(training_anormal_data, np.arange(training_anormal_size))

        train_anormal_loader = torch.utils.data.DataLoader(
            training_anormal_data,
            batch_size=args.a_train_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("=================================Loading Normal-Driving Training Data!=================================")
        training_normal_data = DAD(root_path=args.root_path,
                                   subset='train',
                                   view=args.view,
                                   sample_duration=before_crop_duration,
                                   type='normal',
                                   spatial_transform=spatial_transform,
                                   temporal_transform=temporal_transform
                                   )

        training_normal_size = int(len(training_normal_data) * args.n_split_ratio)
        training_normal_data = torch.utils.data.Subset(training_normal_data, np.arange(training_normal_size))

        train_normal_loader = torch.utils.data.DataLoader(
            training_normal_data,
            batch_size=args.n_train_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("========================================Loading Validation Data========================================")
        val_spatial_transform = spatial_transforms.Compose([
            spatial_transforms.Scale(args.sample_size),
            spatial_transforms.CenterCrop(args.sample_size),
            spatial_transforms.ToTensor(args.norm_value),
            spatial_transforms.Normalize([0], [1])
        ])
        validation_data = DAD(root_path=args.root_path,
                              subset='validation',
                              view=args.view,
                              sample_duration=args.sample_duration,
                              type=None,
                              spatial_transform=val_spatial_transform,
                              )

        validation_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        len_neg = training_anormal_data.__len__()
        len_pos = training_normal_data.__len__()
        num_val_data = validation_data.__len__()
        print(f'len_neg: {len_neg}')
        print(f'len_pos: {len_pos}')
        print(
            "============================================Generating Model============================================")

        if args.model_type == 'resnet':
            model_head = resnet.ProjectionHead(args.feature_dim, args.model_depth)
        elif args.model_type == 'shufflenet':
            model_head = shufflenet.ProjectionHead(args.feature_dim)
        elif args.model_type == 'shufflenetv2':
            model_head = shufflenetv2.ProjectionHead(args.feature_dim)
        elif args.model_type == 'mobilenet':
            model_head = mobilenet.ProjectionHead(args.feature_dim)
        elif args.model_type == 'mobilenetv2':
            model_head = mobilenetv2.ProjectionHead(args.feature_dim)
        if args.use_cuda:
            model_head.cuda()

        if args.resume_path == '':
            # ===============generate new model or pre-trained model===============
            model = generate_model(args)
            optimizer = torch.optim.SGD(list(model.parameters()) + list(model_head.parameters()), lr=args.learning_rate, momentum=args.momentum,
                                        dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
            nce_average = NCEAverage(args.feature_dim, len_neg, len_pos, args.tau, args.Z_momentum)
            criterion = NCECriterion(len_neg)
            begin_epoch = 1
            best_acc = 0
            memory_bank = []
        else:
            # ===============load previously trained model ===============
            args.pre_train_model = False
            model = generate_model(args)
            resume_path = os.path.join(args.checkpoint_folder, args.resume_path)
            resume_checkpoint = torch.load(resume_path)
            model.load_state_dict(resume_checkpoint['state_dict'])
            resume_head_checkpoint = torch.load(os.path.join(args.checkpoint_folder, args.resume_head_path))
            model_head.load_state_dict(resume_head_checkpoint['state_dict'])
            if args.use_cuda:
                model_head.cuda()
            optimizer = torch.optim.SGD(list(model.parameters()) + list(model_head.parameters()), lr=args.learning_rate, momentum=args.momentum,
                                        dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            nce_average = resume_checkpoint['nce_average']
            criterion = NCECriterion(len_neg)
            begin_epoch = resume_checkpoint['epoch'] + 1
            best_acc = resume_checkpoint['acc']
            memory_bank = resume_checkpoint['memory_bank']
            del resume_checkpoint
            torch.cuda.empty_cache()
            adjust_learning_rate(optimizer, args.learning_rate)

        print(
            "==========================================!!!START TRAINING!!!==========================================")
        cudnn.benchmark = True
        batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'],
                              args.log_resume)
        epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'],
                              args.log_resume)
        val_logger = Logger(os.path.join(args.log_folder, 'val.log'),
                            ['epoch', 'accuracy', 'normal_acc', 'anormal_acc', 'threshold', 'acc_list',
                             'normal_acc_list', 'anormal_acc_list'], args.log_resume)

        for epoch in range(begin_epoch, begin_epoch + args.epochs + 1):
            memory_bank, loss = train(train_normal_loader, train_anormal_loader, model, model_head, nce_average,
                                      criterion, optimizer, epoch, args, batch_logger, epoch_logger, memory_bank)

            if epoch % args.val_step == 0:

                print(
                    "==========================================!!!Evaluating!!!==========================================")
                normal_vec = torch.mean(torch.cat(memory_bank, dim=0), dim=0, keepdim=True)
                normal_vec = l2_normalize(normal_vec)

                model.eval()
                accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
                    model, normal_vec, validation_loader, args.use_cuda)
                print(
                    f'Epoch: {epoch}/{args.epochs} | Accuracy: {accuracy} | Normal Acc: {acc_n} | Anormal Acc: {acc_a} | Threshold: {best_threshold}')
                print(
                    "==========================================!!!Logging!!!==========================================")
                val_logger.log({
                    'epoch': epoch,
                    'accuracy': accuracy * 100,
                    'normal_acc': acc_n * 100,
                    'anormal_acc': acc_a * 100,
                    'threshold': best_threshold,
                    'acc_list': acc_list,
                    'normal_acc_list': acc_n_list,
                    'anormal_acc_list': acc_a_list
                })
                if accuracy > best_acc:
                    best_acc = accuracy
                    print(
                        "==========================================!!!Saving!!!==========================================")
                    checkpoint_path = os.path.join(args.checkpoint_folder,
                                                   f'best_model_{args.model_type}_{args.view}.pth')
                    states = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'acc': accuracy,
                        'threshold': best_threshold,
                        'nce_average': nce_average,
                        'memory_bank': memory_bank
                    }
                    torch.save(states, checkpoint_path)

                    head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                        f'best_model_{args.model_type}_{args.view}_head.pth')
                    states_head = {
                        'state_dict': model_head.state_dict()
                    }
                    torch.save(states_head, head_checkpoint_path)

            if epoch % args.save_step == 0:
                print(
                    "==========================================!!!Saving!!!==========================================")
                checkpoint_path = os.path.join(args.checkpoint_folder,
                                               f'{args.model_type}_{args.view}_{epoch}.pth')
                states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': accuracy,
                    'nce_average': nce_average,
                    'memory_bank': memory_bank
                }
                torch.save(states, checkpoint_path)

                head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                    f'{args.model_type}_{args.view}_{epoch}_head.pth')
                states_head = {
                    'state_dict': model_head.state_dict()
                }
                torch.save(states_head, head_checkpoint_path)

            if epoch % args.lr_decay == 0:
                lr = args.learning_rate * (0.1 ** (epoch // args.lr_decay))
                adjust_learning_rate(optimizer, lr)
                print(f'New learning rate: {lr}')

    elif args.mode == 'test':
        if not os.path.exists(args.normvec_folder):
            os.makedirs(args.normvec_folder)
        score_folder = './score/'
        if not os.path.exists(score_folder):
            os.makedirs(score_folder)
        args.pre_train_model = False

        model_front_d = generate_model(args)
        model_front_ir = generate_model(args)
        model_top_d = generate_model(args)
        model_top_ir = generate_model(args)

        resume_path_front_d = './checkpoints/best_model_' + args.model_type + '_front_depth.pth'
        resume_path_front_ir = './checkpoints/best_model_' + args.model_type + '_front_IR.pth'
        resume_path_top_d = './checkpoints/best_model_' + args.model_type + '_top_depth.pth'
        resume_path_top_ir = './checkpoints/best_model_' + args.model_type + '_top_IR.pth'

        resume_checkpoint_front_d = torch.load(resume_path_front_d)
        resume_checkpoint_front_ir = torch.load(resume_path_front_ir)
        resume_checkpoint_top_d = torch.load(resume_path_top_d)
        resume_checkpoint_top_ir = torch.load(resume_path_top_ir)

        model_front_d.load_state_dict(resume_checkpoint_front_d['state_dict'])
        model_front_ir.load_state_dict(resume_checkpoint_front_ir['state_dict'])
        model_top_d.load_state_dict(resume_checkpoint_top_d['state_dict'])
        model_top_ir.load_state_dict(resume_checkpoint_top_ir['state_dict'])

        model_front_d.eval()
        model_front_ir.eval()
        model_top_d.eval()
        model_top_ir.eval()

        val_spatial_transform = spatial_transforms.Compose([
            spatial_transforms.Scale(args.sample_size),
            spatial_transforms.CenterCrop(args.sample_size),
            spatial_transforms.ToTensor(args.norm_value),
            spatial_transforms.Normalize([0], [1]),
        ])

        print("========================================Loading Test Data========================================")
        test_data_front_d = DAD_Test(root_path=args.root_path,
                                     subset='validation',
                                     view='front_depth',
                                     sample_duration=args.sample_duration,
                                     type=None,
                                     spatial_transform=val_spatial_transform,
                                     )
        test_loader_front_d = torch.utils.data.DataLoader(
            test_data_front_d,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        num_val_data_front_d = test_data_front_d.__len__()
        print('Front depth view is done')

        test_data_front_ir = DAD_Test(root_path=args.root_path,
                                      subset='validation',
                                      view='front_IR',
                                      sample_duration=args.sample_duration,
                                      type=None,
                                      spatial_transform=val_spatial_transform,
                                      )
        test_loader_front_ir = torch.utils.data.DataLoader(
            test_data_front_ir,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        num_val_data_front_ir = test_data_front_ir.__len__()
        print('Front IR view is done')

        test_data_top_d = DAD_Test(root_path=args.root_path,
                                   subset='validation',
                                   view='top_depth',
                                   sample_duration=args.sample_duration,
                                   type=None,
                                   spatial_transform=val_spatial_transform,
                                   )
        test_loader_top_d = torch.utils.data.DataLoader(
            test_data_top_d,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        num_val_data_top_d = test_data_top_d.__len__()
        print('Top depth view is done')

        test_data_top_ir = DAD_Test(root_path=args.root_path,
                                    subset='validation',
                                    view='top_IR',
                                    sample_duration=args.sample_duration,
                                    type=None,
                                    spatial_transform=val_spatial_transform,
                                    )
        test_loader_top_ir = torch.utils.data.DataLoader(
            test_data_top_ir,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        num_val_data_top_ir = test_data_top_ir.__len__()
        print('Top IR view is done')
        assert num_val_data_front_d == num_val_data_front_ir == num_val_data_top_d == num_val_data_top_ir

        print("==========================================Loading Normal Data==========================================")
        training_normal_data_front_d = DAD(root_path=args.root_path,
                                           subset='train',
                                           view='front_depth',
                                           sample_duration=args.sample_duration,
                                           type='normal',
                                           spatial_transform=val_spatial_transform,
                                           )

        training_normal_size = int(len(training_normal_data_front_d) * args.n_split_ratio)
        training_normal_data_front_d = torch.utils.data.Subset(training_normal_data_front_d,
                                                               np.arange(training_normal_size))

        train_normal_loader_for_test_front_d = torch.utils.data.DataLoader(
            training_normal_data_front_d,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'Front depth view is done (size: {len(training_normal_data_front_d)})')

        training_normal_data_front_ir = DAD(root_path=args.root_path,
                                            subset='train',
                                            view='front_IR',
                                            sample_duration=args.sample_duration,
                                            type='normal',
                                            spatial_transform=val_spatial_transform,
                                            )

        training_normal_size = int(len(training_normal_data_front_ir) * args.n_split_ratio)
        training_normal_data_front_ir = torch.utils.data.Subset(training_normal_data_front_ir,
                                                                np.arange(training_normal_size))

        train_normal_loader_for_test_front_ir = torch.utils.data.DataLoader(
            training_normal_data_front_ir,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'Front IR view is done (size: {len(training_normal_data_front_ir)})')

        training_normal_data_top_d = DAD(root_path=args.root_path,
                                         subset='train',
                                         view='top_depth',
                                         sample_duration=args.sample_duration,
                                         type='normal',
                                         spatial_transform=val_spatial_transform,
                                         )

        training_normal_size = int(len(training_normal_data_top_d) * args.n_split_ratio)
        training_normal_data_top_d = torch.utils.data.Subset(training_normal_data_top_d,
                                                             np.arange(training_normal_size))

        train_normal_loader_for_test_top_d = torch.utils.data.DataLoader(
            training_normal_data_top_d,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'Top depth view is done (size: {len(training_normal_data_top_d)})')

        training_normal_data_top_ir = DAD(root_path=args.root_path,
                                          subset='train',
                                          view='top_IR',
                                          sample_duration=args.sample_duration,
                                          type='normal',
                                          spatial_transform=val_spatial_transform,
                                          )

        training_normal_size = int(len(training_normal_data_top_ir) * args.n_split_ratio)
        training_normal_data_top_ir = torch.utils.data.Subset(training_normal_data_top_ir,
                                                              np.arange(training_normal_size))

        train_normal_loader_for_test_top_ir = torch.utils.data.DataLoader(
            training_normal_data_top_ir,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'Top IR view is done (size: {len(training_normal_data_top_ir)})')

        print(
            "============================================START EVALUATING============================================")
        normal_vec_front_d = get_normal_vector(model_front_d, train_normal_loader_for_test_front_d,
                                               args.cal_vec_batch_size,
                                               args.feature_dim,
                                               args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec_front_d.npy'), normal_vec_front_d.cpu().numpy())

        normal_vec_front_ir = get_normal_vector(model_front_ir, train_normal_loader_for_test_front_ir,
                                                args.cal_vec_batch_size,
                                                args.feature_dim,
                                                args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec_front_ir.npy'), normal_vec_front_ir.cpu().numpy())

        normal_vec_top_d = get_normal_vector(model_top_d, train_normal_loader_for_test_top_d, args.cal_vec_batch_size,
                                             args.feature_dim,
                                             args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec_top_d.npy'), normal_vec_top_d.cpu().numpy())

        normal_vec_top_ir = get_normal_vector(model_top_ir, train_normal_loader_for_test_top_ir,
                                              args.cal_vec_batch_size,
                                              args.feature_dim,
                                              args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec_top_ir.npy'), normal_vec_top_ir.cpu().numpy())

        cal_score(model_front_d, model_front_ir, model_top_d, model_top_ir, normal_vec_front_d,
                  normal_vec_front_ir,
                  normal_vec_top_d, normal_vec_top_ir, test_loader_front_d, test_loader_front_ir,
                  test_loader_top_d,
                  test_loader_top_ir, score_folder, args.use_cuda)

        gt = get_fusion_label(os.path.join(args.root_path, 'LABEL.csv'))

        hashmap = {'top_d': 'Top(D)',
                   'top_ir': 'Top(IR)',
                   'fusion_top': 'Top(DIR)',
                   'front_d': 'Front(D)',
                   'front_ir': 'Front(IR)',
                   'fusion_front': 'Front(DIR)',
                   'fusion_d': 'Fusion(D)',
                   'fusion_ir': 'Fusion(IR)',
                   'fusion_all': 'Fusion(DIR)'
                   }

        for mode, mode_name in hashmap.items():
            score = get_score(score_folder, mode)
            best_acc, best_threshold, AUC = evaluate(score, gt, False)
            print(
                f'Mode: {mode_name}:      Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)}')
            score = post_process(score, args.window_size)
            best_acc, best_threshold, AUC = evaluate(score, gt, False)
            print(
                f'View: {mode_name}(post-processed):       Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)} \n')

