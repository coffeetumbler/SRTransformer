import argparse


def parse_args_train():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--', type=, default=,
#                         help='')
    
    # General options
    parser.add_argument('--model', type=str, default='MSIRE', help='MSIRE/DWIRE')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--master_port', default='12355', type=str)

    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Checkpoint state dictionary')
    parser.add_argument('--resume_training', default=None, type=str,
                        help='Resuming training with log files; path to log folder required')
    parser.add_argument('--start_epoch', '--start_step', default=None, type=int,
                        help='Staring epoch when resuming training')
    parser.add_argument('--load_options', default=False, action='store_true',
                        help='Loading options when resuming training')
    
    # Model options
    parser.add_argument('--lr_img_res', type=int, default=64,
                        help='Resolution of low-resolution image')
    parser.add_argument('--upscale', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--d_embed', type=str, default='152',
                        help='Embedding dimensions; a common dimension or list of dimensions (with seperater ",")')
    parser.add_argument('--n_layer', type=str, default='3,3,3,3,3,3',
                        help='Number of encoder layers; total number of layers or list of each number of layer (with seperater ",")')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--hidden_dim_rate', type=int, default=3,
                        help='hidden dimension rate of FF layers in encoders')
    parser.add_argument('--conv_hidden_rate', type=float, default=2,
                        help='hidden dimension rate of expansion layers in depth-wise separable convolutions')
    parser.add_argument('--residual_hidden_rate', type=int, default=4,
                        help='hidden dimension rate of hidden layers at the ends of residual connections')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--path_dropout', type=float, default=0)
    
    parser.add_argument('--train_dataset', type=str, default='mixed',
                        help='Training dataset; mixed/DIV2K/Flickr2K')
    parser.add_argument('--valid_dataset', type=str, default='Set5',
                        help='Validation dataset; BSD100/Urban100/manga109/Set5/Set14')
    parser.add_argument('--channelwise_noise_rate', type=float, default=0.5)
    
    parser.add_argument('--num_epochs', type=int, default=3000)
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum training steps; None for automatic steps from n_epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--loss', type=str, default='l1',
                        help='Loss type; l1/mse')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_min_rate', type=float, default=0.05)
    parser.add_argument('--warmup_lr_init_rate', type=float, default=0.001)
    parser.add_argument('--warmup_t_rate', type=float, default=0.03)

    parser.add_argument('--grad_clip_norm', default=1, type=float,
                        help='Maximum gradient norm in gradient clipping')
    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='Gradient accumulation steps')
    parser.add_argument('--summary_steps', default=200, type=int,
                        help='Training summary frequency')
    parser.add_argument('--validation_steps', default=20, type=int,
                        help='Validation and checkpoint saving frequency')
    
    parser.add_argument('--timeout', default=60, type=int,
                        help='Inactivity time to stop training')
    parser.add_argument('--save_val_output', default=False, action='store_true',
                        help='Saving output images from validation')
    
    return parser.parse_args()


def parse_args_test():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--', type=, default=,
#                         help='')
    
    # General options
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # Model options
    parser.add_argument('--lr_img_res', type=int, default=None,
                        help='Resolution of low-resolution image, None for maximum size automatically')
    parser.add_argument('--upscale', type=int, default=2)

    # Loading options
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--state_dict', type=str, default='')

    # Test options
    parser.add_argument('--dataset', type=str, default='Set5',
                        help='Test dataset; Urban100/manga109/Set5/Set14/BSD100')
    parser.add_argument('--output_results', default=False, action='store_true',
                        help='Output images saving option')
    parser.add_argument('--batch_size', type=int, default=16)
                        
    return parser.parse_args()