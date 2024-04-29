# python distributed.py \
#	--num_gpus 1 --gpu_list 0 --mode render_ortho    \
#	--workers_per_gpu 10 --view_idx $1 \
#	--start_i $2 --end_i $3 --ortho_scale 1.35 \
#	--input_models_path ../data_lists/lvis_uids_filter_by_vertex.json  \
#	--objaverse_root /media/pawting/SN640/Datasets/hf-objaverse-v1/ \
#	--save_folder /media/pawting/SN640/Datasets/obj_lvis_13views \
#	--blender_install_path /home/pawting/blender \
##	--random_pose

# python render_batch_ortho.py \
#	--num_gpus 1 --gpu_list 0 --mode render_ortho    \
#	--workers_per_gpu 6 --view_idx 0 \
#	--resolution 1024 \
#	--start_i 0 --end_i 526 --ortho_scale 1.35 \
#	--input_models_path /media/pawting/SN640/Datasets/THuman2/all.txt  \
#	--dataset_root /media/pawting/SN640/Datasets/THuman2/ \
#	--save_folder /media/pawting/SN640/Datasets/wonder3d_dev/wonder3d_thuman2 \
#	--blender_install_path /home/pawting/blender

# python render_batch_ortho.py \
#	--num_gpus 1 --gpu_list 0 --mode render_ortho    \
#	--workers_per_gpu 6 --view_idx 0 \
#	--resolution 1024 \
#	--start_i 525 --end_i 526 --ortho_scale 1.35 \
#	--input_models_path /media/pawting/SN640/Datasets/THuman2/all.txt  \
#	--dataset_root /media/pawting/SN640/Datasets/THuman2/ \
#	--save_folder /media/pawting/SN640/Datasets/wonder3d_dev/wonder3d_thuman2 \
#	--blender_install_path /home/pawting/blender

 python render_batch_ortho.py \
	--num_gpus 1 --gpu_list 0 --mode render_ortho \
	--workers_per_gpu 6 --view_idx 0 \
	--resolution 1024 \
	--start_i 0 --end_i 526 --ortho_scale 1.35 \
	--input_models_path /media/pawting/SN640/Datasets/THuman2/all.txt  \
	--dataset_root /media/pawting/SN640/Datasets/THuman2/ \
	--save_folder /media/pawting/SN640/Datasets/wonder3d_dev/thuman2/ortho135_res1024/ \
	--blender_install_path /home/pawting/blender