import numpy as np
import os
import smplx
import torch
import trimesh

# thuman_dir = '/media/pawting/SN640/Datasets/THuman2.1_Release'
# params = np.load(os.path.join(thuman_dir, 'smplx/0527/smplx_param.pkl'), allow_pickle=True)
# m_smplx = smplx.SMPLX('../data/smplx', gender='neutral', use_pca=False)
# smpl_out = m_smplx.forward(
# transl=torch.tensor(params['transl']),
# global_orient=torch.tensor(params['global_orient']),
# body_pose=torch.tensor(params['body_pose']),
# betas=torch.tensor(params['betas']),
# left_hand_pose=torch.tensor(params['left_hand_pose']),
# right_hand_pose=torch.tensor(params['right_hand_pose']),
# expression=torch.tensor(params['expression']),
# jaw_pose=torch.tensor(params['jaw_pose']),
# leye_pose=torch.tensor(params['leye_pose']),
# reye_pose=torch.tensor(params['reye_pose'])
# )
# vertices = smpl_out.vertices.detach().cpu().numpy()[0] * params['scale']
# m_model = trimesh.Trimesh(vertices=vertices, faces=m_smplx.faces, process=False)
# m_model.export('../data/smplx_0527.obj')

