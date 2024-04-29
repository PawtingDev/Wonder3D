#CUDA_VISIBLE_DEVICES=0 \
# blenderproc run --blender-install-path /mnt/pfs/users/longxiaoxiao/workplace/blender \
# blenderProc_nineviews_ortho.py \
# --object_path /mnt/pfs/data/objaverse_lvis_glbs/c7/c70e8817b5a945aca8bb37e02ddbc6f9.glb --view 0 \
# --output_folder ./out_renderings/ \
# --object_uid c70e8817b5a945aca8bb37e02ddbc6f9 \
# --ortho_scale 1.35 \
# --resolution 512 \
##  --reset_object_euler


#CUDA_VISIBLE_DEVICES=0 \
# blenderproc run blenderProc_ortho.py \
# --object_path /media/pawting/SN640/Datasets/hf-objaverse-v1/glbs/000-000/bfc5ba51548b419c94ecf632c0aa9960.glb --view 0 \
# --output_folder ./out_renderings/ \
# --ortho_scale 1.35 \
# --resolution 512 \
##  --reset_object_euler

#CUDA_VISIBLE_DEVICES=0 \
# blenderproc run blenderProc_ortho.py \
# --object_path /media/pawting/SN640/Datasets/THuman2/0015/0015.obj --view 0 \
# --output_folder ./out_renderings/ \
# --ortho_scale 1.35 \
# --resolution 512 \
##  --reset_object_euler

CUDA_VISIBLE_DEVICES=0 \
 blenderproc run blenderProc_ortho.py \
 --object_path /media/pawting/SN640/Datasets/THuman2/scans/0000/0000.obj --view 0 \
 --output_folder /media/pawting/SN640/Datasets/wonder3d_dev/wonder3d_thuman2/ \
 --ortho_scale 1.35 \
 --resolution 1024