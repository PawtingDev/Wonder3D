import blenderproc as bproc
import pickle

import argparse, sys, os, math, re
import bpy
from glob import glob

import matplotlib
from blenderproc.python.postprocessing import PostProcessingUtility

from mathutils import Vector, Matrix
import time
import urllib.request
import numpy as np
from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes

import cv2
import PIL.Image as Image

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--view', type=int, default=0, help='the index of view to be rendered')
parser.add_argument(
    "--object_path", type=str,
    default='/media/pawting/SN640/Datasets/THuman2/scans/0000/0000.obj',
    required=True, help="Path to the object file",
)
parser.add_argument('--output_folder', type=str, default='output', help='The path the output will be dumped to.')
parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
parser.add_argument('--ortho_scale', type=float, default=1.25, help='ortho rendering usage; how large the object is')
parser.add_argument('--random_pose', action='store_true', help='whether randomly rotate the poses to be rendered')

args = parser.parse_args()


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()

    dxyz = bbox_max - bbox_min
    dist = np.sqrt(dxyz[0] ** 2 + dxyz[1] ** 2 + dxyz[2] ** 2)
    #    print("dxyz: ",dxyz, "dist: ", dist)
    # scale = 1 / max(bbox_max - bbox_min)
    scale = 1. / dist
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    return scale, offset


def get_a_camera_location(loc):
    location = Vector([loc[0], loc[1], loc[2]])
    direction = - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rot_quat.to_euler()
    return location, rotation_euler


# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
    ))
    return RT


def get_calibration_matrix_K_from_blender(mode='simple'):
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    camdata = scene.camera.data

    if mode == 'simple':
        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()

    if mode == 'complete':

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        K = np.array([
            [alpha_u, skew, u_0],
            [0, alpha_v, v_0],
            [0, 0, 1]
        ], dtype=np.float32)

    return K


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=False)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        # bpy.ops.import_scene.obj(filepath=object_path, use_smooth_groups=False, use_split_objects=False,
        #                       use_split_groups=False, use_groups_as_vgroups=False, use_image_search=False,
        #                       split_mode='OFF', global_clamp_size=0.0, axis_forward='-Z', axis_up='Y')
        # bproc.loader.load_obj(filepath=object_path)
        bpy.ops.wm.obj_import(filepath=object_path)
    elif object_path.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def colorize_depth_maps(
        depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def dump_render_results(type, views, data, view_idx):
    for j in range(len(views)):
        index = j

        view = f"{view_idx:03d}" + VIEWS[j]

        # Nomralizes depth maps
        depth_map = data['depth'][index]

        # dis_map = data['distance'][index]
        # print("dis_max", np.max(dis_map))
        # mask
        # valid_mask = dis_map != np.max(dis_map)
        # invalid_mask = dis_map == np.max(dis_map)
        # # dis -> depth
        # depth_map = PostProcessingUtility.dist2depth(dis_map)
        # depth_map = dis_map

        depth_max = np.max(depth_map)
        valid_mask = depth_map != depth_max
        invalid_mask = depth_map == depth_max
        far = np.max(depth_map[valid_mask])
        depth_map[invalid_mask] = far  # far_bound
        # depth_map = np.uint16((depth_map / 10) * 65535)
        valid_mask = valid_mask.astype(np.int8) * 255

        depth_max = np.max(depth_map)
        depth_min = np.min(depth_map)

        depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-5)
        # depth_map = (depth_map - 1.5) / (2.7 - 1.5)
        depth_map.clip(0, 1)

        # w&g depth
        # depth_colored = colorize_depth_maps(
        #     depth_map, 0, 1, cmap="Spectral"
        # ).squeeze()
        # depth_map = np.uint8(depth_colored * 255)
        # depth_map = chw2hwc(depth_map)
        # depth_map = np.concatenate([depth_map, valid_mask[:, :, None]], axis=-1)

        # colored depth
        depth_map = np.uint8(depth_map * 255)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        depth_map = np.concatenate([depth_map, valid_mask[:, :, None]], axis=-1)

        normal_map = data['normals'][index] * 255

        # color_map = data['colors'][index]
        color_map = data['diffuse'][index]
        color_map = np.concatenate([color_map, valid_mask[:, :, None]], axis=-1)

        normal_map = np.concatenate([normal_map, valid_mask[:, :, None]], axis=-1)
        if type == 'model':
            dump_dir_model = os.path.join(args.output_folder, 'model')
            os.makedirs(dump_dir_model, exist_ok=True)
            Image.fromarray(color_map.astype(np.uint8)).save('{}/rgb_{}.png'.format(dump_dir_model, view))

            Image.fromarray(normal_map.astype(np.uint8)).save('{}/normals_{}.png'.format(dump_dir_model, view))
            cv2.imwrite('{}/mask_{}.png'.format(dump_dir_model, view), valid_mask)
            cv2.imwrite('{}/depth_{}.png'.format(dump_dir_model, view), depth_map)
        elif type == 'smplx':
            dump_dir_smplx = os.path.join(args.output_folder, 'smplx')
            os.makedirs(dump_dir_smplx, exist_ok=True)
            Image.fromarray(normal_map.astype(np.uint8)).save('{}/normals_{}.png'.format(dump_dir_smplx, view))
            cv2.imwrite('{}/mask_{}.png'.format(dump_dir_smplx, view), valid_mask)
            cv2.imwrite('{}/depth_{}.png'.format(dump_dir_smplx, view), depth_map)
        #
        # Image.fromarray(depth_map.astype(np.uint8)).save('{}/depth_{}.png'.format(args.output_folder, view))
        #
        # # cv2.imwrite('{}/{}/rgb_{}.png'.format(args.output_folder, object_uid, view), color_map)
        # # cv2.imwrite('{}/{}/depth_{}.png'.format(args.output_folder,object_uid, view), depth_map)
        # # cv2.imwrite('{}/{}/normals_{}.png'.format(args.output_folder,object_uid, view), normal_map)
        # cv2.imwrite('{}/mask_{}.png'.format(args.output_folder, view), valid_mask)


bproc.init()

# Place camera
bpy.data.cameras[0].type = "ORTHO"
bpy.data.cameras[0].ortho_scale = args.ortho_scale
print("ortho scale ", args.ortho_scale)

VIEWS = ["_front", "_back", "_right", "_left", "_front_right", "_front_left"]


# VIEWS = ["_front", "_back", "_right", "_left", "_front_right", "_front_left", "_back_right", "_back_left", "_top"]
# EXTRA_VIEWS = ["_front_right_top", "_front_left_top", "_back_right_top", "_back_left_top", ]

def save_images(object_file: str, viewidx: int) -> None:
    global VIEWS
    # global EXTRA_VIEWS
    # 1. init scene
    reset_scene()

    # 2. Rotate scan using smplx to face forward
    lst_object_path = args.object_path.split('/')
    obj_id = lst_object_path[-2]
    dataset_type = lst_object_path[-4]
    base_dir = '/'.join(lst_object_path[:-3])
    smplx_dir = 'smplx'

    smplx_path = os.path.join(base_dir, smplx_dir, obj_id, 'smplx_param.pkl')
    smplx_obj_file = os.path.join(base_dir, smplx_dir, obj_id, 'mesh_smplx.obj')
    # with open(smplx_path, 'rb') as f:
    #     smplx_para = np.load(f, allow_pickle=True)
    smplx_para = np.load(smplx_path, allow_pickle=True)

    y_orient = smplx_para['global_orient'][0][1]

    # load the object to bpy
    load_object(object_file)
    load_object(smplx_obj_file)
    # args.output_folder = os.path.join(args.output_folder, object_uid[:2])
    os.makedirs(args.output_folder, exist_ok=True)

    # if args.reset_object_euler:
    # if int(obj_id) > 525:
    print(f"{dataset_type} -> {y_orient}rad")
    if dataset_type == 'THuman2.1_Release':
        for obj in scene_root_objects():
            obj.rotation_euler[0] = 0  # don't know why
            obj.rotation_euler[2] = -y_orient  # rotate Z
        bpy.ops.object.select_all(action="DESELECT")
    elif dataset_type == 'THuman2':
        for obj in scene_root_objects():
            obj.rotation_euler[2] = -y_orient  # rotate Z
        bpy.ops.object.select_all(action="DESELECT")
    else:
        raise NotImplementedError

    scale, offset = normalize_scene()

    Scale_path = os.path.join(args.output_folder, "scale_offset_matrix.txt")
    np.savetxt(Scale_path, [scale] + list(offset) + [args.ortho_scale])

    try:
        # some objects' normals are affected by textures
        mesh_objects = convert_to_meshes([obj for obj in scene_meshes()])
        for obj in mesh_objects:
            print("removing invalid normals")
            for mat in obj.get_materials():
                mat.set_principled_shader_value("Normal", [1, 1, 1])
    except:
        print("don't know why")

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)

    radius = 2.0

    camera_locations = [
        np.array([0, -radius, 0]),  # camera_front
        np.array([0, radius, 0]),  # camera back
        np.array([radius, 0, 0]),  # camera right
        np.array([-radius, 0, 0]),  # camera left
        np.array([radius, -radius, 0]) / np.sqrt(2.),  # camera_front_right
        np.array([-radius, -radius, 0]) / np.sqrt(2.),  # camera front left
        np.array([radius, radius, 0]) / np.sqrt(2.),  # camera back right
        np.array([-radius, radius, 0]) / np.sqrt(2.),  # camera back left
        # np.array([0, 0, radius]),  # camera top
        # np.array([radius, -radius, radius]) / np.sqrt(3.),  # camera_front_right_top
        # np.array([-radius, -radius, radius]) / np.sqrt(3.),  # camera front left top
        # np.array([radius, radius, radius]) / np.sqrt(3.),  # camera back right top
        # np.array([-radius, radius, radius]) / np.sqrt(3.),  # camera back left top
    ]

    for location in camera_locations:
        _location, _rotation = get_a_camera_location(location)
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=_location, rotation=_rotation,
                                  scale=(1, 1, 1))
        _camera = bpy.context.selected_objects[0]
        _constraint = _camera.constraints.new(type='TRACK_TO')
        _constraint.track_axis = 'TRACK_NEGATIVE_Z'
        _constraint.up_axis = 'UP_Y'
        _camera.parent = cam_empty
        _constraint.target = cam_empty
        _constraint.owner_space = 'LOCAL'

    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action='DESELECT')
    cam_empty.select_set(True)

    if args.random_pose:
        print("random poses")
        delta_z = np.random.uniform(-60, 60, 1)  # left right rotate
        delta_x = np.random.uniform(-15, 30, 1)  # up and down rotate
        delta_y = 0
    else:
        print("fix poses")
        delta_z = 0
        delta_x = 0
        delta_y = 0

    bpy.ops.transform.rotate(value=math.radians(viewidx), orient_axis='Z', orient_type='VIEW')

    bpy.ops.transform.rotate(value=math.radians(delta_z), orient_axis='Z', orient_type='VIEW')
    bpy.ops.transform.rotate(value=math.radians(delta_y), orient_axis='Y', orient_type='VIEW')
    bpy.ops.transform.rotate(value=math.radians(delta_x), orient_axis='X', orient_type='VIEW')

    bpy.ops.object.select_all(action='DESELECT')

    # VIEWS = VIEWS + EXTRA_VIEWS
    for j in range(len(VIEWS)):
        view = f"{viewidx:03d}" + VIEWS[j]
        # set camera
        cam = bpy.data.objects[f'Camera.{j + 1:03d}']
        location, rotation = cam.matrix_world.decompose()[0:2]

        print(j, rotation)

        cam_pose = bproc.math.build_transformation_mat(location, rotation.to_matrix())
        bproc.camera.set_resolution(args.resolution, args.resolution)
        bproc.camera.add_camera_pose(cam_pose)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(cam)
        # print(np.linalg.inv(cam_pose))  # the same
        # print(RT)
        RT_path = os.path.join(args.output_folder, view + "_RT.txt")
        np.savetxt(RT_path, RT)

    # activate normal and depth rendering
    # must be here
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    # bproc.renderer.enable_distance_output(activate_antialiasing=True, antialiasing_distance_max=3)

    # Render the scene
    bproc.renderer.enable_diffuse_color_output()
    bproc.renderer.set_noise_threshold(0.01)
    bproc.renderer.set_light_bounces(diffuse_bounces=4, glossy_bounces=4, transmission_bounces=12,
                                     transparent_max_bounces=8, volume_bounces=0)

    # render model
    # Manually set metallic to 1.0
    bpy.data.materials["material0"].node_tree.nodes["Principled BSDF"].inputs[1].default_value = 1
    bpy.data.objects['mesh_smplx'].hide_render = True
    render_results_model = bproc.renderer.render()
    dump_render_results(type='model', views=VIEWS, data=render_results_model, view_idx=viewidx)
    # render smplx
    bpy.data.objects[f'{obj_id}'].hide_render = True
    bpy.data.objects['mesh_smplx'].hide_render = False
    # for smplx normal smoothing
    bpy.data.objects['mesh_smplx'].select_set(True)
    bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action='DESELECT')
    render_results_smplx = bproc.renderer.render()
    dump_render_results(type='smplx', views=VIEWS, data=render_results_smplx, view_idx=viewidx)


if __name__ == "__main__":
    tic = time.time()
    # render
    if not os.path.exists(args.object_path):
        print("object does not exists")
    else:
        try:
            save_images(args.object_path, args.view)
        except Exception as e:
            print("Failed to render", args.object_path)
            print(e)

    toc = time.time()

    print("Finished", args.object_path, "in", toc - tic, "seconds")

"""
blenderproc run --blender-install-path /home/pawting/blender render_thuman.py \
    --object_path '/media/pawting/SN640/Datasets/THuman2/model/0000/0000.obj' \
     --ortho_scale 1.35 \
     --resolution 512 \
# face left
blenderproc run --blender-install-path /home/pawting/blender render_thuman.py \
    --object_path '/media/pawting/SN640/Datasets/THuman2/model/0006/0006.obj' \
     --ortho_scale 0.4 \
     --resolution 512 \
     --thuman 2.0 \
     --output_folder '/media/pawting/SN640/Datasets/wonder3d_dev/thuman2/debug' \
     --view 30
# THuman 2.1
blenderproc run --blender-install-path /home/pawting/blender render_thuman.py \
    --object_path '/media/pawting/SN640/Datasets/THuman2.1_Release/model/2444/2444.obj' \
     --ortho_scale 1.0 \
     --resolution 512 \
     --thuman 2.1 \
     --output_folder '/media/pawting/SN640/Datasets/wonder3d_dev/thuman2/debug' \
     --view 0
"""
