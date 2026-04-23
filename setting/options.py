import argparse

parser = argparse.ArgumentParser('The training and evaluation script', add_help=False)
# training set
parser.add_argument('--epoch', type=int, default=40, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (AdamW)')
parser.add_argument(
    '--wd_mode',
    type=str,
    default='all',
    choices=['all', 'matrix_only'],
    help='weight decay mode: all params or matrix_only (exclude bias/norm/time_/gate_)',
)
parser.add_argument('--adam_eps', type=float, default=1e-8, help='AdamW epsilon')
parser.add_argument('--adam_beta1', type=float, default=0.9, help='AdamW beta1')
parser.add_argument('--adam_beta2', type=float, default=0.999, help='AdamW beta2')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing for CrossEntropyLoss')

parser.add_argument('--gpu_id', type=str, default='0', help='the gpu id')
parser.add_argument('--num_work', type=int, default=4, help='dataloader workers')
parser.add_argument('--start_epoch', type=int, default=1)

# training dataset
parser.add_argument(
    '--model',
    type=str,
    default='GeoRwkvV2_MixSigmoid_WeakBeta',
    help='model name (see model/factory.py)',
)
parser.add_argument(
    '--dataset',
    type=str,
    default='Houston2013',
    choices=['Houston2013', 'Houston2018', 'Trento'],
    help='dataset name',
)
parser.add_argument('--useval', type=int, default=0)
parser.add_argument('--save_path', type=str, default='./checkpoints/', help='the path to save models and logs')
parser.add_argument('--data_root', type=str, default='./data', help='dataset root dir, expects <data_root>/<dataset>/*.mat')

parser.add_argument('--dry_run', action='store_true', help='run a single forward/backward step then exit')
parser.add_argument('--eval_interval', type=int, default=1, help='evaluate every N epochs (0=only last epoch)')
parser.add_argument('--seed', type=int, default=6, help='random seed')
parser.add_argument(
    '--deterministic',
    type=int,
    default=0,
    choices=[0, 1],
    help='set 1 to enable deterministic training mode (slower, more reproducible)',
)
parser.add_argument(
    '--init_weights',
    type=str,
    default='',
    help='optional path to init state_dict (.pth) loaded before training',
)
parser.add_argument(
    '--init_weights_strict',
    type=int,
    default=1,
    choices=[0, 1],
    help='strict load for --init_weights (1=strict, 0=allow missing/unexpected keys)',
)
parser.add_argument(
    '--grad_clip_norm',
    type=float,
    default=0.0,
    help='global grad-norm clipping value (0=disabled)',
)
parser.add_argument(
    '--ema_decay',
    type=float,
    default=0.0,
    help='EMA decay for model weights used during evaluation (0=disabled, typical: 0.999)',
)

# performance flags
parser.add_argument(
    '--amp',
    type=int,
    default=1,
    choices=[0, 1],
    help='use autocast mixed precision when CUDA is available',
)
parser.add_argument(
    '--amp_dtype',
    type=str,
    default='bf16',
    choices=['fp16', 'bf16'],
    help='autocast dtype for CUDA training',
)
parser.add_argument(
    '--compile_model',
    type=int,
    default=0,
    choices=[0, 1],
    help='use torch.compile on the model (PyTorch 2.x)',
)
parser.add_argument(
    '--allow_tf32',
    type=int,
    default=1,
    choices=[0, 1],
    help='allow TF32 matmul/cudnn on supported GPUs',
)

# dataloader / preprocessing performance
parser.add_argument(
    '--cache_pca',
    type=int,
    default=1,
    choices=[0, 1],
    help='cache standardized PCA / secondary modality arrays under <dataset>/.cache',
)
parser.add_argument(
    '--return_full_hsi',
    type=int,
    default=0,
    choices=[0, 1],
    help='return full HSI patches in the dataloader (0 keeps a lightweight placeholder)',
)
parser.add_argument(
    '--persistent_workers',
    type=int,
    default=1,
    choices=[0, 1],
    help='enable persistent dataloader workers when num_work>0',
)
parser.add_argument(
    '--prefetch_factor',
    type=int,
    default=2,
    help='dataloader prefetch factor when num_work>0',
)

parser.add_argument('--best_acc', type=float, default=0, help='save best accuracy')
parser.add_argument('--best_epoch', type=int, default=1, help='save best epoch')

opt = parser.parse_args()
