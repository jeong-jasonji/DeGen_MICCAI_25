import argparse
import os

from uvcgan2               import ROOT_OUTDIR, train
from uvcgan2.presets       import GEN_PRESETS, BH_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train DecisionViT I2I model'
    )

    add_preset_name_parser(parser, 'gen',  GEN_PRESETS, 'uvcgan2-decision')
    add_preset_name_parser(parser, 'head', BH_PRESETS,  'bsd', 'batch head')

    parser.add_argument(
        '--no-pretrain', dest = 'no_pretrain', action = 'store_true',
        help = 'disable uasge of the pre-trained generator'
    )

    parser.add_argument(
        '--lambda-gp', dest = 'lambda_gp', type = float,
        default = 0.01, help = 'magnitude of the gradient penalty'
    )

    parser.add_argument(
        '--lambda-cycle', dest = 'lambda_cyc', type = float,
        default = 10.0, help = 'magnitude of the cycle-consisntecy loss'
    )
    
    parser.add_argument(
        '--lambda-fwd', dest = 'lambda_fwd', type = float,
        default = 1.0, help = 'magnitude of the latent discriminator class prediction loss (forward - real to fake). 0.0 if just latent vector extraction'
    )
    
    parser.add_argument(
        '--eval-cycle-mode', dest = 'eval_cycle_mode', type = str,
        default = 'vanilla', help = 'backward class loss mode [vanilla, decision_fwd, decision_full]'
    )
    
    parser.add_argument(
        '--softmax_out', dest = 'softmax_out', type = int,
        default = 0, help = 'use softmax for latent prediction [0=False, 1=True]'
    )

    parser.add_argument(
        '--lr-gen', dest = 'lr_gen', type = float,
        default = 1e-4, help = 'learning rate of the generator'
    )
    
    parser.add_argument(
        '--dataset', dest = 'dataset', type = str,
        default = 'celeba_makeup', help = 'dataset to train: [celeba_makeup, breast_density, headct_any]'
    )
    
    parser.add_argument(
        '--architecture', dest = 'architecture', type = str,
        default = 'cycle', help = 'type of architecture to train: [cycle, star]'
    )
    
    parser.add_argument(
        '--img_ch', dest = 'img_ch', type = int,
        default = 3, help = 'image channels'
    )

    return parser.parse_args()

def get_transfer_preset(cmdargs):
    if cmdargs.no_pretrain:
        return None

    base_model = (
        'celeba_preproc/'
        'model_m(simple-autoencoder)_d(None)'
        f"_g({GEN_PRESETS[cmdargs.gen]['model']})_pretrain-{cmdargs.gen}"
    )

    return {
        'base_model' : base_model,
        'transfer_map'  : {
            'gen_ab' : 'gen_ab',
            'gen_ba' : 'gen_ba',
            'disc_a' : 'disc_a',
            'disc_b' : 'disc_b',
        },
        'strict'        : True,
        'allow_partial' : False,
        'fuzzy'         : None,
    }

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : 1,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'image-csv',
                    'domain' : domain,
                    'path'   : './dataframes/{}'.format(cmdargs.dataset),
                },
                'shape'           : (cmdargs.img_ch, 256, 256),
                'transform_train' : [
                    'random-flip-horizontal',
                    { 'name' : 'resize',      'size' : 256, },
                    { 'name' : 'random-crop', 'size' : 256, },
                ],
                'transform_test' : [
                    { 'name' : 'resize',      'size' : 256, },
                    { 'name' : 'center-crop', 'size' : 256, },
                ],
            } for domain in [ 'a', 'b' ]
        ],
        'merge_type' : 'unpaired',
        'workers'    : 1,
    },
    'epochs'      : 100,
    'discriminator' : {
        'model'      : 'basic',
        'model_args' : { 'shrink_output' : False, },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'spectr_norm' : True,
    },
    # latent discriminator
    'l_discriminator' : {
        'model'      : 'latent',
        'model_args' : {'latent_dim' : 384, 'n_classes' : 2,},
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'softmax_out': cmdargs.softmax_out,
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr_gen,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'uvcgan2-decision',
    'model_args' : {
        'lambda_a'        : cmdargs.lambda_cyc,
        'lambda_b'        : cmdargs.lambda_cyc,
        'lambda_idt'      : 0.5,
        'lambda_l_fwd'    : cmdargs.lambda_fwd,
        'lambda_l_bwd'    : cmdargs.lambda_fwd,
        'avg_momentum'    : 0.9999,
        'head_queue_size' : 3,
        'head_config'     : {
            'name'            : BH_PRESETS[cmdargs.head],
            'input_features'  : 512,
            'output_features' : 1,
            'activ'           : 'leakyrelu',
        },
        'eval_cycle_mode' :cmdargs.eval_cycle_mode,
    },
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : cmdargs.lambda_gp,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
    'scheduler'       : None,
    'loss'            : 'vanilla',
    'steps_per_epoch' : 2000,
    'transfer'        : None,
    'label'  : (
        f'{cmdargs.gen}-{cmdargs.head}_({cmdargs.no_pretrain}'
        f':{cmdargs.lambda_cyc}:{cmdargs.lambda_gp}:{cmdargs.lr_gen})'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, cmdargs.dataset, '256_decision_{}'.format(cmdargs.eval_cycle_mode)),
    'log_level'  : 'DEBUG',
    'checkpoint' : '25_50',
}

train(args_dict)

