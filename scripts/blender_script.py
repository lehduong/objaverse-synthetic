"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""
import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple
import json
import bpy
import numpy as np
from mathutils import Vector


parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="objaverse_synthetic")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=300)
parser.add_argument("--camera_dist", type=int, default=4)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

scene.render.use_compositing = True
scene.use_nodes = True
scene.view_layers["ViewLayer"].use_pass_z = True

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 800
render.resolution_y = 800
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 64
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting(height=3) -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    locations = [[-1, -1, height], [-1, 1, height], [1, 1, height], [1, -1, height], [-1, -1, -height], [-1, 1, -height], [1, 1, -height], [1, -1, -height]]
    bpy.ops.object.light_add(type="AREA")
    for i, location in enumerate(locations):
        bpy.ops.object.light_add(type="AREA")
        light_name = "Area.{}".format(str(i+1).zfill(3))
        light = bpy.data.lights[light_name]
        light.energy = 30000
        bpy.data.objects[light_name].location = location
        bpy.data.objects[light_name].scale[0] = 100
        bpy.data.objects[light_name].scale[1] = 100
        bpy.data.objects[light_name].scale[2] = 100
    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects["Area"].select_set(True)
    bpy.ops.object.delete()


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


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


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


def normalize_scene(scale):
    bbox_min, bbox_max = scene_bbox()
    scale = scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def setup_depth_viewer(max_depth=10):
    tree = scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')      
    map = tree.nodes.new('CompositorNodeMapValue') 
    map.min[0] = 0
    map.max[0] = max_depth
    map.use_min = False
    map.use_max = False
    map.size[0] = 1
    set_output_extension('exr')
    output_file = tree.nodes.new('CompositorNodeOutputFile')
    links.new(rl.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], output_file.inputs[0])
    return rl, output_file

def set_output_extension(type='png'):
    if type == 'png':
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode = "RGBA"
    elif type == 'exr':
        render.image_settings.file_format = "OPEN_EXR"
        render.image_settings.color_mode = "BW"
    else:
        raise ValueError("Expect type to be png or exr")
    
def save_images(object_file: str, save_mesh=True) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    object_uid = os.path.basename(object_file).split(".")[0]
    # already render
    if os.path.exists(os.path.join(args.output_dir, object_uid, 'transforms_train.json')):
        return
    
    reset_scene()
    # load the object
    load_object(object_file)
    normalize_scene(scale=2.5) #TODO: how large is the object
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    # # setup nodes for depth rendering
    # max_depth = 1000000000 # clamp depth to be in the range of [0; max_depth]
    # _, output_file = setup_depth_viewer(max_depth=max_depth)
    # output_file.base_path = os.path.join(args.output_dir, object_uid, 'depth')
    # list of camera pose
    frames = {'train': [], 'val': [], 'test': []}
    to_export = {
        "camera_angle_x": bpy.data.cameras[0].angle_x,
        "width": render.resolution_x,
        "height": render.resolution_y,
        # "clamp_depth": max_depth
    }
    for i in range(args.num_images):
        # bpy.context.scene.frame_set(i)
        # output_file.base_path = os.path.join(args.output_dir, object_uid, 'depth_'+str(i))
        if i % 3 == 2:
            mode = 'test'
        elif i % 3 == 1:
            mode = 'val'
        else:
            mode = 'train'
        # set the camera position
        theta = (i / args.num_images) * math.pi * 2
        phi = math.radians(random.randint(35, 90))
        point = (
            args.camera_dist * math.sin(phi) * math.cos(theta),
            args.camera_dist * math.sin(phi) * math.sin(theta),
            args.camera_dist * math.cos(phi),
        )
        cam.location = point
        # render the image
        render_path = os.path.join(args.output_dir, object_uid, mode, f"{i:03d}.png")
        scene.render.filepath = render_path
        set_output_extension('png')
        bpy.ops.render.render(write_still=True)
        # get depth image
        # depth_image = np.array(bpy.data.images['Viewer Node'].pixels).reshape(render.resolution_y, render.resolution_x, 4)
        # disp = 1 / depth_image[:, :, 0]
        # with open(os.path.join(args.output_dir, object_uid, mode, f"{i:03d}_depth.npz"), 'w') as f:
            # np.savez(f, disp)

        # store camera pose for this frame
        pos, rt, scale = cam.matrix_world.decompose()
        bpy.context.view_layer.update()
        to_add = get_frame_poses(pos, rt, i, mode)
        frames[mode].append(to_add)
    # save camera pose
    for mode in ['train', 'val', 'test']:   
        with open(f'{args.output_dir}/{object_uid}/transforms_{mode}.json', 'w') as f:
            to_export['frames'] = frames[mode]
            json.dump(to_export, f,indent=4)
    if save_mesh:
        while bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[0], do_unlink=True)
        load_object(object_file)
        normalize_scene(scale=2.5) #TODO: how large is the object
        mesh = join_meshes()
        mesh.select_set(True)
        #bpy.ops.export_mesh.ply(filepath=os.path.join(args.output_dir, object_uid, "mesh.ply"), use_selection=True)
        bpy.ops.wm.obj_export(filepath=os.path.join(args.output_dir, object_uid, "mesh.obj"), export_selected_objects=True, forward_axis='Y', up_axis='Z')

def join_meshes() -> bpy.types.Object:
    """Joins all the meshes in the scene into one mesh."""
    # get all the meshes in the scene
    meshes = scene_meshes()
    # join all of the meshes
    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
    # join the meshes
    bpy.ops.object.join()
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    assert len(meshes) == 1
    mesh = meshes[0]
    return mesh

def get_frame_poses(pos, rt, i, mode):
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0.0, 0.0, 0.0, 1.0])
    to_add = {\
        "file_path":f'{mode}/{str(i).zfill(3)}',
        "transform_matrix":matrix
    }
    return to_add    

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path, save_mesh=True)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
