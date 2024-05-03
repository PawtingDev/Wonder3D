import os
import numpy as np

render_dir = '/media/pawting/SN640/Datasets/wonder3d_dev/thuman2/ortho1_res1024_viewstep5'
output_dir = '/media/pawting/SN640/hello_worlds/Wonder3D/instant-nsr-pl/datasets/fixed_poses_thuman'
view_types = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
subjects_all = [f'{i:04d}' for i in range(2445)]
print(subjects_all)

for view in view_types:
    RT_all = []
    for sub in subjects_all:
        RT = np.loadtxt(os.path.join(render_dir, sub, f'{0:03d}_{view}_RT.txt'))
        print(RT)
        RT_all.append(RT)
    RT_avg = np.average(RT_all, axis=0)
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, f'{0:03d}_{view}_RT.txt'), RT_avg)
