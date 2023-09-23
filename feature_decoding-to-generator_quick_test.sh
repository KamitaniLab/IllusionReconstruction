#!/bin/sh

### step 1: feature decoding 

PARAM1="./feature-decoding/config/deeprecon-fmd-mscoco_fmriprep_500voxel_caffe_bvlc_reference_caffenet_allunits_fastl2lir_alpha100_quick_test.yaml"
PARAM2="./feature-decoding/config/deepreconS6-fmd-mscoco_fmriprep_500voxel_caffe_bvlc_reference_caffenet_allunits_fastl2lir_alpha100_quick_test.yaml"

## decoder training
# output: results/feature-decoding/ImageNetTraining_FMD_MSCOCO

# subject S1-5, S7
python feature-decoding/featdec_fastl2lir_train.py $PARAM1
# subject S6
python feature-decoding/featdec_fastl2lir_train.py $PARAM2

## predict decoded features
# output: results/feature-decoding/Illusion_avg_trials

# subject S1-5, S7
python feature-decoding/featdec_fastl2lir_predict.py $PARAM1
# subject S6
python feature-decoding/featdec_fastl2lir_predict.py $PARAM2



### step 2: reconstruction - feed decoded features to generator
# output: results/reconstruction/recon_images

## GAN_quick_test
python generator/recon_feature_to_GAN_quick_test.py



### step 3: visualization

# output: results/plots/qucik_test
python visualization/make_figures_recon_images_quick_test.py
