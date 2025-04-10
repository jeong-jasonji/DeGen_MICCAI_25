# use torch metrics to evaluate the generated images
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
# image metric libraries
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device id to use')
    parser.add_argument('--eval_df_dir', type=str, default='./eval', help='directory where the evaluation dataframes are')
    parser.add_argument('--n_compare', type=int, default=5000, help='number of samples to compare with')
    parser.add_argument('--desc', type=str, default=None, help='optional description for progress bar')
    
    return parser.parse_args()
    
def load_img(img_path, bit_depth='uint8', bit_max=255, transform=None):
    # load numpy or png images
    if img_path.endswith('.npy'):
        image = np.load(img_path)
    else:
        image = np.array(Image.open(img_path))
    # check bitdepth and change to correct bit depth as necessary
    if 'int' in str(image.dtype):
        if image.max() > 255:
            image = image / 65535.0
        else:
            image = image / 255.0
    image = (image * 255).astype('uint8')
    # return a PIL image for transforms
    if len(image.shape) > 2:
        image = image.squeeze()
    image = Image.fromarray(image).convert(mode='RGB')
    
    #FID, KID, LPIPS requires 8bit image tensors
    if transform is not None:
        image = transform(image)

    return image
    
def calculate_metrics(real_list, fake_list, device='cpu', metric_list=['KID', 'FID', 'LPIPS'], desc=None):
    # init lpips metrics
    lpips_real = []
    lpips_fake = []
    lpips = []
    
    transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            ])

    # load metric functions
    if 'FID' in metric_list:
        fid = FrechetInceptionDistance(normalize=True).to(device)
    if 'KID' in metric_list:
        kid = KernelInceptionDistance(normalize=True, subset_size= len(real_list) // (4)).to(device)
    if 'LPIPS' in metric_list:
        lpips_score = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    
    # evaluate all samples
    for i in tqdm(range(len(real_list)), desc='Eval' if desc is None else 'Eval {}'.format(desc)):
        # real - val
        real = load_img(real_list[i], transform=transform).unsqueeze(0).to(device)
        # fake - real or generated
        fake = load_img(fake_list[i], transform=transform).unsqueeze(0).to(device)
        # calculate
        if 'FID' in metric_list:
            fid.update(real, real=True)
            fid.update(fake, real=False)
        if 'KID' in metric_list:
            kid.update(real, real=True)
            kid.update(fake, real=False)
        if 'LPIPS' in metric_list:
            lpips.append(lpips_score(real, fake).item())
                
    # get the metrics
    fid_score = fid.compute() if 'FID' in metric_list else None
    kid_mean, kid_std = kid.compute() if 'KID' in metric_list else (None, None)
    # real vs fake lpips
    lpips_mean, lpips_std = np.mean(lpips), np.std(lpips) if 'LPIPS' in metric_list else (None, None)
    
    return fid_score, kid_mean, kid_std, lpips_mean, lpips_std

def eval_datasets(args):
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # data to check
    df_dir = args.eval_df_dir
    gen_img_dfs = os.listdir(df_dir) # should have two dataframes to compare against each other
    # eval datasets
    dset_log = open(args.log_path, 'a')
    comp_a = pd.read_csv(os.path.join(df_dir, gen_img_dfs[gen_img_df][0]))
    comp_b = pd.read_csv(os.path.join(df_dir, gen_img_dfs[gen_img_df][1]))
    # make sure both datasets sizes are the same for comparision
    if len(comp_a) > len(comp_b):
        comp_a = comp_a.sample(n=len(comp_b))
    elif len(comp_b) > len(comp_a):
        comp_b = comp_b.sample(n=len(comp_a))
    if args.n_compare is not None:
        comp_a = comp_a.sample(n=args.n_compare)
        comp_b = comp_b.sample(n=args.n_compare)
    comp_a = comp_a.reset_index()
    comp_b = comp_b.reset_index()
    # get the list of items
    real_list = comp_a.img_path.to_list()
    fake_list = comp_b.img_path.to_list()
    # get all metrics now
    fid_score, kid_mean, kid_std, lpips_mean, lpips_std = calculate_metrics(real_list, fake_list, device=dev, desc=args.desc)
    print('[{}]: FID: {:.4f}, KID: {:.4f} {:.4f}, LPIPS: {:.4f} {:.4f}'.format(gen_img_df, fid_score, kid_mean, kid_std, lpips_mean, lpips_std), file=dset_log)
    dset_log.close()


# python eval_quality.py
if __name__ == '__main__':
    args = parse_eval_args()
    eval_datasets_df(args)
            