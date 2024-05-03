#!/bin/bash

#cd ./instant-nsr-pl
#python launch.py --config configs/avatar-ortho-wmask.yaml --gpu 0 --train dataset.scene=owl
#cd ../

accelerate launch --config_file 1gpu.yaml test_smplx_normal_ctrl.py \
            --config configs/test/mvdiffusion-joint-ortho-6views-controlnet.yaml