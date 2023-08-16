import os, time
import numpy as np
import cv2
import torch

import json

from utils.dataloader import get_dataloader, batch_making
from utils.options import parse_args_test
from utils.image_processing import PSNR, SSIM, denormalize, quantize
import utils.config as config



def test(model, dataloader_test, batch_size, test_size, trim_size, device, save_image=None):
    psnr_test = []
    ssim_test = []

    with torch.no_grad():
        for th, items in enumerate(dataloader_test):
            # Patched inputs
            if dataloader_test.dataset.sliding_window:
                origin = items['origin'].to(device)
                img = torch.zeros(origin.shape).to(device)
                degradeds = batch_making(items=items['degraded'], batch=batch_size)

                for keys, degraded in degradeds:
                    _img = model.evaluate(degraded.to(device))
                    _, _, img_h, img_w = _img.shape

                    for i, key in enumerate(keys):
                        p, q = key[0], key[1]
                        img[..., p:p+img_h, q:q+img_w] += _img[i] 

                mask = items["mask"].to(device)    
                img /= mask
            
            # Whole inputs
            else:
                origin = items["origin"].to(device)
                degraded = items["degraded"].to(device)
                img = model.evaluate(degraded)
            
            img = denormalize(img, device=device)
            img = quantize(img).to(torch.float64)
            origin = origin.to(torch.float64)

            psnr = PSNR(img, origin, trim_size, device).item()
            ssim = SSIM(img, origin, trim_size, device)

            if save_image != None:
                cv2.imwrite(save_image+'/final_{0}.png'.format(th), img[0].permute(1,2,0).cpu().numpy()*255)

            psnr_test.append(psnr)
            ssim_test.append(ssim)
            
        psnr_test = np.sum(psnr_test) / test_size
        ssim_test = np.sum(ssim_test) / test_size

    return psnr_test, ssim_test
     
    
    
if __name__ == '__main__':
    args = parse_args_test() 

    # Date
    now = time.localtime()
    ntime = "%04d%02d%02d_%02d%02d%02d"%(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

    # Save Path
    save_path = config.TEST_RESULTS_DIR
    save_path_upscale = save_path + 'X{}/'.format(args.upscale)
    save_path_date = save_path_upscale + ntime
    save_path_output = save_path_date + '/output'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_upscale):
        os.mkdir(save_path_upscale)
    if not os.path.exists(save_path_date):
        os.mkdir(save_path_date)
    if not os.path.exists(save_path_output):
        os.mkdir(save_path_output)

    # Hyperparameters save
    with open(save_path_date+'/hyperparameters.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # GPU setting
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    # Data Loading
    if args.lr_img_res == None:
        dataloader_test = get_dataloader(setting='test', upscale=args.upscale, dataset=args.dataset)
    else:
        dataloader_test = get_dataloader(setting='test', upscale=args.upscale, dataset=args.dataset,
                                         lr_img_size=args.lr_img_res, sliding_window=True)
    test_size = len(dataloader_test)

    # Load Model & Weight
    model = torch.load(args.model, map_location=device)
    model.load_state_dict(torch.load(args.state_dict, map_location='cpu'))
    model.eval()
    
    psnr_test, ssim_test = test(model, dataloader_test, args.batch_size, test_size, args.upscale, device,
                                save_path_output if args.output_results else None)
    print(f'average PSNR : {psnr_test}, average SSIM : {ssim_test}')
    
    file_rep = open(save_path_output+"/test_result.txt", "w")
    file_rep.write(f'PSNR : {psnr_test} \nSSIM : {ssim_test}')
    file_rep.close()