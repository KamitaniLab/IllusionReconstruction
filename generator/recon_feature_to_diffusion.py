#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
updated by: Jin Hirano
updated by: fcheng Tue Jan 17 17:16:48 2023
"""


import argparse
import os
import glob
import scipy.io as sio

import numpy as np
import torch as th
import torch.distributed as dist


from diffusion.guided_diffusion import dist_util, logger

from diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


from bdpy.dataform import DecodedFeatures
from bdpy.feature import normalize_feature
from PIL import Image

def main():
    args = create_argparser().parse_args()
    device = "cuda"
    save_dir_root = './results/reconstruction/recon_images/diffusion'
    if not os.path.exists(save_dir_root):
        os.makedirs(save_dir_root)
    
    dist_util.setup_dist()
    logger.configure(dir=save_dir_root)

    logger.log(f"Model Path: {args.model_path}")
    logger.log(f"Diffusion steps: {args.diffusion_steps}")
    logger.log(f"Noise schedule: {args.noise_schedule}")
    logger.log(f"Target layer: {args.target_layer}")
    logger.log(f"Dataset: {args.data_dir}")
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    seed = args.seed
    th.manual_seed(seed)
    all_images = []

    # load test image
    dataset = args.data_dir
    subject = args.subject
    roi = args.roi
    network = args.network
    image_path_list = args.image_list
    print(image_path_list)


###############################################################################################
    # Please define the path to the decoded feature here
    decoded_feature_dir = os.path.join('./results/feature-decoding',dataset,'decoded_features', network)
    target_layer = args.target_layer

    # get image label
    matfiles = glob.glob(os.path.join(decoded_feature_dir, target_layer, subject, roi, '*.mat'))
    
    file_name_list = [os.path.splitext(os.path.basename(fl))[0] for fl in matfiles]
    # load DNN features
    features = DecodedFeatures(decoded_feature_dir, squeeze=False)
    
    feat = {
        f"{image_label}": features.get(layer=target_layer, subject=subject, roi=roi, image=image_label)
        for image_label in file_name_list
    }

    for image_label in file_name_list:
        decoded_feature_for_input = feat[f"{image_label}"]
        
        feature_std_file = './generator/GAN/estimated_cnn_feat_std_bvlc_reference_caffenet_ImgSize_227x227_chwise_dof1.mat'
        feat_std0 = sio.loadmat(feature_std_file)
        decoded_feature_for_input = normalize_feature(decoded_feature_for_input,
                                channel_wise_mean=True, channel_wise_std=True,
                                channel_axis=0,
                                shift='self', scale=feat_std0['relu6'],
                                std_ddof=1)

        decoded_feature_for_input = th.from_numpy(decoded_feature_for_input)

        feat[f"{image_label}"] = decoded_feature_for_input
###############################################################################################

    for label in image_path_list:

        model_kwargs = {}
        ##################################################
        if args.class_cond:
            feature = th.zeros([1, 4096])
            feature[0]= feat[label]    
            feature = feature.to(device)
            model_kwargs["y"] = feature 
        ##################################################
        
        # initialize sampling
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )


        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])


    arr = np.concatenate(all_images, axis=0)
    print(arr.shape)
    #arr = arr[: args.num_samples]
    
    # save images
    if dist.get_rank() == 0:
        
        out_path = os.path.join(save_dir_root, subject, roi)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for image, label in zip(arr, image_path_list):
            Image.fromarray(image).save(f"{out_path}/recon_image_normalized-{label}-seed{seed}.tiff")
        logger.log(f"saving to {out_path}")


    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=1,
        use_ddim=False,
        model_path="",
        target_layer="",
        data_dir="",
        shuffle=True,
        seed = 0,
        network="bvlc_reference_caffenet",
        image_list=['fillingin004_large_unionjack_lumi0p3_sat0p8_connected_trial03',
                    'fillingin004_large_unionjack_lumi0p3_sat0p8_connected_trial07',
                    'fillingin004_large_unionjack_lumi0p3_sat0p8_connected_trial08',
                    'fillingin004_large_unionjack_lumi0p3_sat0p8_connected_trial10',
                    'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected_trial03',
                    'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected_trial07',
                    'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected_trial08',
                    'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected_trial10'
                    ],
        roi="VC",
        subject = "",

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
