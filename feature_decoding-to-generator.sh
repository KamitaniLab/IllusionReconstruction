#!/bin/sh

### step 1: feature decoding 

## decoder training

# subject S1-5, S7
python feature-decoding/featdec_fastl2lir_train.py feature-decoding/config/deeprecon-fmd-mscoco_fmriprep_5rep_500voxel_caffe_bvlc_reference_caffenet_allunits_fastl2lir_alpha100.yaml
# subject S6
python feature-decoding/featdec_fastl2lir_train.py feature-decoding/config/deepreconS6-fmd-mscoco_fmriprep_5rep_500voxel_caffe_bvlc_reference_caffenet_allunits_fastl2lir_alpha100.yaml


## predict decoded features

# subject S1-5, S7
python feature-decoding/featdec_fastl2lir_predict.py feature-decoding/config/deeprecon-fmd-mscoco_fmriprep_5rep_500voxel_caffe_bvlc_reference_caffenet_allunits_fastl2lir_alpha100.yaml
# subject S6
python feature-decoding/featdec_fastl2lir_predict.py feature-decoding/config/deepreconS6-fmd-mscoco_fmriprep_5rep_500voxel_caffe_bvlc_reference_caffenet_allunits_fastl2lir_alpha100.yaml



### step 2: reconstruction - feed decoded features to generator

## GAN
python generator/recon_GAN.py
