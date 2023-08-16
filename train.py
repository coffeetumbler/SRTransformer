import os, time, datetime

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import json

from models.msire import MultiScaleIREncoder, DilatedWindowedIREncoder, ExpandedWindowedIREncoder
from utils.dataloader import get_dataloader
from utils.options import parse_args_train
import utils.config as config
from test import test

from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler

import torch.distributed as dist
import torch.multiprocessing as mp
from threading import Timer as _timer



# Initialize the environment.
def init_process(rank, world_size, master_port, timeout=60):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(0, timeout))
    
# Timer object to stop a process
class Timer:
    def __init__(self, function=None, exception_str=''):
        self.function = function
        self.exception_str = exception_str

    def raise_exception(self, *args, **kwargs):
        if self.function != None:
            self.function(*args, **kwargs)
        raise Exception(self.exception_str)

    def start(self, time, *args, **kwargs):
        self.timer = _timer(time, self.raise_exception, args=args, kwargs=kwargs)
        self.timer.start()

    def cancel(self):
        self.timer.cancel()

    def reset(self, time, *args, **kwargs):
        self.timer.cancel()
        self.timer = _timer(time, self.raise_exception, args=args, kwargs=kwargs)
        self.timer.start()


# Kill this process.
def kill_process(message=None):
    print("Current process is killed.")
    if message != None:
        print(message)
    os.kill(os.getpid(), 9)

# Synchronize all parameters of models located in all processors.
def synchronize(checkpoint_dir_file, model):
    # Read a shared checkpoint directory.
    with open(checkpoint_dir_file, 'r') as f:
        checkpoint_dir = f.readline()
    model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')))

    


def main(rank, world_size, args):
    # GPU setting
    if world_size > 1:
        init_process(rank, world_size, args.master_port)
    cudnn.benchmark = True
    device = torch.device('cuda:'+args.gpu_id.split(',')[rank] if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.model_path, map_location=device)
    save_path_state_dict = args.save_path_state_dict
    save_path_output = args.save_path_output
    save_path_date = args.save_path_date
    
    # Log summary file
    if rank == 0:
        summary_writer = SummaryWriter(save_path_date)

    # Loss setting
    if args.loss == 'l1':
        criterion = nn.L1Loss().to(device)
    elif args.loss == 'mse':
        criterion = nn.MSELoss().to(device)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Dataset
    mixed_dataset = (args.train_dataset == 'mixed')
    dataloader_train = get_dataloader(lr_img_size=args.lr_img_res,
                                      batch_size=args.batch_size//world_size,
                                      setting='train',
                                      drop_last=True,
                                      num_workers=args.num_workers,
                                      upscale=args.upscale,
                                      dataset=args.train_dataset,
                                      channel_wise_noise=args.channelwise_noise_rate,
                                      data_merge=mixed_dataset)
    train_size = len(dataloader_train.dataset) * world_size
    batch_size = args.batch_size

    iters_per_epoch = len(dataloader_train.dataset) // (batch_size // world_size)
    accum_steps = args.accumulation_steps

    if args.max_steps == None:
        max_iter = iters_per_epoch * (args.num_epochs // world_size)
        num_epochs = args.num_epochs
        initial_epoch = 0 if args.start_epoch == None else args.start_epoch
        _iter = iters_per_epoch * (initial_epoch // world_size)
        summary_steps = args.summary_steps
        validation_steps = args.validation_steps
    else:
        max_iter = args.max_steps * accum_steps
        initial_step = 0 if args.start_epoch == None else args.start_epoch
        _iter = accum_steps * initial_step
        validation_steps = accum_steps * args.validation_steps
        summary_steps = accum_steps * args.summary_steps

    # Dataset for validation
    dataloader_val = get_dataloader(setting='test',
                                    dataset=args.valid_dataset,
                                    upscale=args.upscale)
    val_size = len(dataloader_val)

    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=max_iter,
                                  lr_min=args.learning_rate*args.lr_min_rate,
                                  warmup_lr_init=args.learning_rate*args.warmup_lr_init_rate,
                                  warmup_t=int(max_iter*args.warmup_t_rate),
                                  cycle_limit=1,
                                  t_in_epochs=False)
    scheduler.step_update(_iter)

    # Set a timer to stop process when an error occurs.
    timer = Timer(function=kill_process)
    timeout_str = 'Timeout in process {}'.format(rank)
    timeout_len = args.timeout
    timer.start(timeout_len, timeout_str)

    # Gradient settings
    grad_factor = accum_steps * world_size
    grad_clip_norm = args.grad_clip_norm

    upscale = args.upscale
    save_val_output = args.save_val_output

    # Single training step
    def train_step(items, update, return_loss):
        origin = items['origin']
        degraded = items['degraded']

        mid_result = model(degraded.to(device))
        loss = criterion(mid_result, origin.to(device)) / grad_factor
        loss.backward()

        # Update model parameters.
        if update:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            # Average all gradients in workers.
            if world_size > 1:
                for param in model.parameters():
                    if param.requires_grad:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            optimizer.step()
            optimizer.zero_grad()

        if return_loss:
            return loss.item() * grad_factor
        return
    
    # Validation
    prefix = save_path_output + '/epoch_' if args.max_steps == None else save_path_output + '/step_'
    suffix = ' (Epoch)' if args.max_steps == None else ' (Step)'
    def validate(step):
        if save_val_output:
            output_folder = prefix + str(step)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
        else:
            output_folder = None

        psnr_val, ssim_val = test(model, dataloader_val, 1, val_size, upscale, device,
                                  save_image=output_folder)
        summary_writer.add_scalar('PSNR/Set5'+suffix, psnr_val, step)
        summary_writer.add_scalar('SSIM/Set5'+suffix, ssim_val, step)
        print('Validation PSNR : {0}, SSIM : {1}\n'.format(psnr_val, ssim_val))

    
    # Start training.
    if args.max_steps == None:  # Training with maximum number of epochs
        # Validate first.
        if rank == 0:
            model.eval()
            validate(initial_epoch)

        for epoch in range(initial_epoch, num_epochs, world_size):
            # Train
            model.train()
            _epoch = epoch + world_size

            if rank == 0:
                with tqdm(total=train_size, desc='Epoch {}/{}'.format(_epoch, num_epochs)) as t:
                    for batch, items in enumerate(dataloader_train):
                        _iter += 1

                        if (batch + 1) % summary_steps != 0:
                            train_step(items, _iter % accum_steps == 0, False)
                        else:
                            loss = train_step(items, _iter % accum_steps == 0, True)
                            summary_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], _iter)
                            summary_writer.add_scalar('Training Loss', loss, _iter)
                            current = (batch+1) * items['origin'].shape[0] * world_size
                            tqdm.write(f"loss: {loss:>6f}, [{current:>5d}/{train_size:>5d}]")

                        scheduler.step_update(_iter)
                        t.update(batch_size)

            else:
                for items in dataloader_train:
                    _iter += 1
                    train_step(items, _iter % accum_steps == 0, False)
                    scheduler.step_update(_iter)

            # Reset a timer.
            timer.reset(timeout_len, timeout_str)
                    
            # Validate and save the checkpoint.
            if _epoch % validation_steps == 0:
                # Save and synchronize model parameters.
                if rank == 0:
                    torch.save(model.state_dict(), os.path.join(save_path_state_dict, 'state_dict_epoch_{}.pt'.format(_epoch)))
                    if world_size > 1:
                        dist.barrier()
                else:
                    dist.barrier()
                    model.load_state_dict(torch.load(os.path.join(save_path_state_dict, 'state_dict_epoch_{}.pt'.format(_epoch)),
                                                    map_location='cpu'))
                    
                # Validate
                if rank == 0:
                    model.eval()
                    validate(_epoch)
                    
                # Reset a timer.
                timer.reset(timeout_len, timeout_str)

        # Cancel a timer.
        timer.cancel()
        
        # End training.
        if rank == 0:
            if _epoch % validation_steps != 0:
                torch.save(model.state_dict(), os.path.join(save_path_state_dict, 'state_dict_epoch_{}.pt'.format(_epoch)))
                
                model.eval()
                validate(_epoch)

            summary_writer.close()


    else:  # Training with maximum number of steps
        # Validate first.
        if rank == 0:
            model.eval()
            validate(initial_step)

        # Train
        model.train()
        train_iterator = iter(dataloader_train)
        if rank == 0:
            t = tqdm(total=args.max_steps, initial=initial_step, desc='Training')

        while _iter < max_iter:
            _iter += 1

            # Get items from train dataloader.
            try:
                items = next(train_iterator)
            except StopIteration:
                train_iterator = iter(dataloader_train)
                items = next(train_iterator)

            # Get loss and update.
            if rank == 0:
                update = _iter % accum_steps == 0
                if _iter % summary_steps != 0:
                    train_step(items, update, False)
                else:
                    current = _iter // accum_steps
                    loss = train_step(items, update, True)
                    summary_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], current)
                    summary_writer.add_scalar('Training Loss', loss, current)
                    tqdm.write(f"loss: {loss:>6f}, [{current:>5d}/{args.max_steps:>5d}]")

                if update:
                    scheduler.step_update(_iter)
                    t.update(1)

            else:
                train_step(items, _iter % accum_steps == 0, False)
                scheduler.step_update(_iter)

            # Reset a timer.
            timer.reset(timeout_len, timeout_str)

            # Validate and save the checkpoint.
            if _iter % validation_steps == 0:
                # Save and synchronize model parameters.
                step = _iter // accum_steps
                if rank == 0:
                    torch.save(model.state_dict(), os.path.join(save_path_state_dict, 'state_dict_step_{}.pt'.format(step)))
                    if world_size > 1:
                        dist.barrier()
                else:
                    dist.barrier()
                    model.load_state_dict(torch.load(os.path.join(save_path_state_dict, 'state_dict_step_{}.pt'.format(step)),
                                                    map_location='cpu'))
                    
                # Validate
                if rank == 0:
                    model.eval()
                    validate(step)
                    model.train()
                    
                # Reset a timer.
                timer.reset(timeout_len, timeout_str)

        # Cancel a timer.
        timer.cancel()
        
        # End training.
        if rank == 0:
            if max_iter % validation_steps != 0:
                torch.save(model.state_dict(), os.path.join(save_path_state_dict, 'state_dict_step_{}.pt'.format(max_iter)))
                
                model.eval()
                validate(max_iter)

            summary_writer.close()
            t.close()


                                      
                                      
if __name__ == "__main__":
    args = parse_args_train()

    # Resume training if resume_training is not None.
    if args.resume_training != None:
        assert args.start_epoch != None
        ntime = args.resume_training.split('/')
        if '' in ntime:
            ntime.remove('')
        ntime = ntime[-1]

        if args.load_options:
            start_epoch = args.start_epoch
            args = torch.load(os.path.join(args.resume_training, 'options.pt'))
            args.start_epoch = start_epoch
        else:
            save_path_date = config.LOG_DIR + 'X{}/'.format(args.upscale) + ntime
            args.save_path_date = save_path_date
            args.model_path = save_path_date + '/model/model.pt'
            args.save_path_state_dict = save_path_date + '/state_dict'
            args.save_path_output = save_path_date + '/output'

        model = torch.load(args.model_path, map_location='cpu')
        if args.max_steps == None:
            model.load_state_dict(torch.load(args.save_path_state_dict + '/state_dict_epoch_{}.pt'.format(args.start_epoch),
                                             map_location='cpu'))
        else:
            model.load_state_dict(torch.load(args.save_path_state_dict + '/state_dict_step_{}.pt'.format(args.start_epoch),
                                             map_location='cpu'))
        torch.save(model, args.model_path)

    # model setting & save
    else:
        # Assertations
        assert args.batch_size % args.world_size == 0
        assert args.num_epochs % args.world_size == 0
        # assert args.validation_steps % args.world_size == 0

        encoder_n_layer = list(map(int, args.n_layer.split(',')))
        d_embed = list(map(int, args.d_embed.split(',')))
        if len(d_embed) == 1:
            d_embed = d_embed[0]

        if args.model == 'MSIRE':
            model = MultiScaleIREncoder(img_res=args.lr_img_res,
                                        d_embed=d_embed,
                                        n_layer=encoder_n_layer,
                                        n_head=args.n_head,
                                        hidden_dim_rate=args.hidden_dim_rate,
                                        window_size=args.window_size,
                                        dropout=args.dropout,
                                        path_dropout=args.path_dropout,
                                        sr_upscale=args.upscale)
        elif args.model == 'DWIRE':
            model = DilatedWindowedIREncoder(img_res=args.lr_img_res,
                                             d_embed=d_embed,
                                             n_layer=encoder_n_layer,
                                             n_head=args.n_head,
                                             hidden_dim_rate=args.hidden_dim_rate,
                                             conv_hidden_rate=args.conv_hidden_rate,
                                             window_size=args.window_size,
                                             dropout=args.dropout,
                                             path_dropout=args.path_dropout,
                                             sr_upscale=args.upscale)
        elif args.model == 'EWIRE':
            model = ExpandedWindowedIREncoder(img_res=args.lr_img_res,
                                              d_embed=d_embed,
                                              n_layer=encoder_n_layer,
                                              n_head=args.n_head,
                                              hidden_dim_rate=args.hidden_dim_rate,
                                              conv_hidden_rate=args.conv_hidden_rate,
                                              residual_hidden_rate=args.residual_hidden_rate,
                                              window_size=args.window_size,
                                              dropout=args.dropout,
                                              path_dropout=args.path_dropout,
                                              sr_upscale=args.upscale)
            
        if args.checkpoint != None:
            checkpoint_state_dict = torch.load(args.checkpoint, map_location='cpu')
            try:
                model.load_state_dict(checkpoint_state_dict)
            except:
                print('State dictionary does not match perfectly with model.')
                try:
                    model.load_state_dict(checkpoint_state_dict, strict=False)
                except:
                    print('Some layers do not have the same structure with the corresponding layers.')
                    print('Trying matching state dictionaries of encoders only.')
                    encoder_name = 'block_1'
                    prefix_len = len(encoder_name)
                    sub_dict = {key[prefix_len:] : value for key, value\
                                in checkpoint_state_dict.items() if key.startswith(encoder_name)}
                    getattr(model, encoder_name[:-1]).load_state_dict(sub_dict, strict=False)
        
        # Date
        now = time.localtime()
        ntime = "%04d%02d%02d_%02d%02d%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        # Save Path
        save_path = config.LOG_DIR
        save_path_upscale = save_path + 'X{}/'.format(args.upscale)
        save_path_date = save_path_upscale + ntime
        save_path_state_dict = save_path_date + '/state_dict'
        save_path_model = save_path_date + '/model'
        save_path_output = save_path_date + '/output'

        args.save_path_date = save_path_date
        args.model_path = save_path_model + '/model.pt'
        args.save_path_state_dict = save_path_state_dict
        args.save_path_output = save_path_output

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(save_path_upscale):
            os.mkdir(save_path_upscale)
        if not os.path.exists(save_path_date):
            os.mkdir(save_path_date)
        if not os.path.exists(save_path_state_dict):
            os.mkdir(save_path_state_dict)
        if not os.path.exists(save_path_model):
            os.mkdir(save_path_model)
        if not os.path.exists(save_path_output):
            os.mkdir(save_path_output)

        # Hyperparameters save
        with open(save_path_date+'/hyperparameters.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        torch.save(model, args.model_path)
        torch.save(args, save_path_date+'/options.pt')

    # Run multi-processing.
    if args.world_size > 1:
        mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        main(0, 1, args)