import torch
from torch import nn
from models import shufflenet, shufflenetv2, resnet, mobilenet, mobilenetv2
from utils import _construct_depth_model

def generate_model(args):
    assert args.model_type in ['resnet', 'shufflenet', 'shufflenetv2', 'mobilenet', 'mobilenetv2']
    if args.pre_train_model == False or args.mode == 'test':
        print('Without Pre-trained model')
        if args.model_type == 'resnet':
            assert args.model_depth in [18, 50, 101]
            if args.model_depth == 18:

                model = resnet.resnet18(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type=args.shortcut_type,
                    tracking=args.tracking,
                    pre_train = args.pre_train_model
                )

            elif args.model_depth == 50:
                model = resnet.resnet50(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type=args.shortcut_type,
                    tracking=args.tracking,
                    pre_train=args.pre_train_model
                )

            elif args.model_depth == 101:
                model = resnet.resnet101(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type=args.shortcut_type,
                    tracking=args.tracking,
                    pre_train=args.pre_train_model
                )

        elif args.model_type == 'shufflenet':
            model = shufflenet.get_model(
                groups=args.groups,
                width_mult=args.width_mult,
                output_dim=args.feature_dim,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'shufflenetv2':
            model = shufflenetv2.get_model(
                output_dim=args.feature_dim,
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'mobilenet':
            model = mobilenet.get_model(
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'mobilenetv2':
            model = mobilenetv2.get_model(
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )

        model = nn.DataParallel(model, device_ids=None)
    else:
        if args.model_type == 'resnet':
            pre_model_path = './premodels/kinetics_resnet_' + str(args.model_depth) + '_RGB_16_best.pth'
            ###default pre-trained model is trained on kinetics dataset which has 600 classes
            if args.model_depth == 18:
                model = resnet.resnet18(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type='A',
                    tracking=args.tracking,
                    pre_train=args.pre_train_model
                )


            elif args.model_depth == 50:
                model = resnet.resnet50(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type='B',
                    tracking=args.tracking,
                    pre_train=args.pre_train_model
                )

            elif args.model_depth == 101:
                model = resnet.resnet101(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type='B',
                    tracking=args.tracking,
                    pre_train=args.pre_train_model
                )

        elif args.model_type == 'shufflenet':
            pre_model_path = './premodels/kinetics_shufflenet_'+str(args.width_mult)+'x_G3_RGB_16_best.pth'
            model = shufflenet.get_model(
                groups=args.groups,
                width_mult=args.width_mult,
                output_dim=args.feature_dim,
                pre_train=args.pre_train_model

            )

        elif args.model_type == 'shufflenetv2':
            pre_model_path = './premodels/kinetics_shufflenetv2_'+str(args.width_mult)+'x_RGB_16_best.pth'
            model = shufflenetv2.get_model(
                output_dim=args.feature_dim,
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train = args.pre_train_model
            )
        elif args.model_type == 'mobilenet':
            pre_model_path = './premodels/kinetics_mobilenet_'+ str(args.width_mult) +'x_RGB_16_best.pth'
            model = mobilenet.get_model(
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'mobilenetv2':
            pre_model_path = './premodels/kinetics_mobilenetv2_' + str(args.width_mult) + 'x_RGB_16_best.pth'
            model = mobilenetv2.get_model(
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )

        model = nn.DataParallel(model, device_ids=None)  # in order to load pre-trained model
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pre_model_path)['state_dict']
        #print(len(pretrained_dict.keys()))
        #print({k for k, v in pretrained_dict.items() if k not in model_dict})
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = _construct_depth_model(model)
    if args.use_cuda:
        model = model.cuda()
    return model




