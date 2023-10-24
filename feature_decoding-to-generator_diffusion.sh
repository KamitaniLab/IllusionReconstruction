#!/bin/bash

### reconstruction  - feed decoded features to generator
## supp: diffusion

MODEL="--model_path generator/diffusion/generator.pt"

MODEL_FLAGS="--attention_resolutions 16,8 --class_cond True --dropout 0.1 
             --image_size 64 --learn_sigma True --num_channels 192 --num_head_channels 64 --num_res_blocks 3"
             
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear 
                 --resblock_updown True --use_new_attention_order True 
                 --use_fp16 True --use_scale_shift_norm True"
                 
SAMPLE_FLAGS="--timestep_respacing 400"

DATA_FLAGS="--data_dir Illusion_single_trial/derivatives/decoded_features/deeprecon-fmd-mscoco_fmriprep_500voxel_bvlc_reference_caffenet_allunits_fastl2lir_alpha100 
            --shuffle False"
            
RECON_FLAGS="--network caffe/bvlc_reference_caffenet --target_layer fc6 
             --roi VC --subject S1"
	    
seeds=(0 1 7 30 64 78 222 666 1111 2022 2023)
for s in "${seeds[@]}"; do
    python generator/recon_feature_to_diffusion.py $MODEL_FLAGS $DIFFUSION_FLAGS $MODEL $SAMPLE_FLAGS $DATA_FLAGS $RECON_FLAGS --seed $s 
done


### visualization

# output: results/plots/figsupp_diffusion.pdf
python visualization/make_figures_recon_images_diffusion.py 
