import glob
import json
import os
import time
import torchvision.transforms
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from gaussian_renderer import render_fn_dict
from scene import GaussianModel
from utils.general_utils import safe_state
from utils.camera_utils import Camera, JSON_to_camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from utils.system_utils import searchForMaxIteration
from scene.direct_light_map import DirectLightMap
from utils.graphics_utils import focal2fov, rgb_to_srgb
from scene.envmap import EnvLight


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

def update_relighting(pose):
    flatten_pose = pose.flatten()
    traject_config_file = "configs/dtu/trajectory.json"
    traject_config = load_json_config(traject_config_file)
    traject_config["trajectory"]["0"] = flatten_pose.tolist()
    with open(traject_config_file, 'w') as f:
        json.dump(traject_config, f)

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


class OrbitCamera:
    def __init__(self, W, H, fovy=60, near=0.1, far=10, rot=None, translate=None, center=None):
        self.W = W
        self.H = H
        if translate is None:
            self.radius = 1
        else:
            self.radius = np.linalg.norm(translate)
        self.radius *= 2
        self.fovy = fovy  # in degree
        self.near = near
        self.far = far

        if center is None:
            self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        else:
            self.center = center

        if rot is None:
            self.rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))  # looking back to z axis
        else:
            self.rot = R.from_matrix(rot)

        # self.up = np.array([0, -1, 0], dtype=np.float32)  # need to be normalized!
        self.up = -self.rot.as_matrix()[:3, 1]

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]  # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])


class GUI:
    def __init__(self, H, W, fovy, c2w, center, render_fn, render_kwargs, 
                 mode="render", debug=True):
        self.W = W
        self.H = H
        self.debug = debug
        rot = c2w[:3, :3]
        translate = c2w[:3, 3] - center
        self.render_fn = render_fn
        self.render_kwargs = render_kwargs

        self.cam = OrbitCamera(self.W, self.H, fovy=fovy * 180 / np.pi, rot=rot, translate=translate, center=center)

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.resize_fn = torchvision.transforms.Resize((self.H, self.W), antialias=True)
        self.downsample = 1
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.envmaps = ["original"] + ["env_map/"+p for p in os.listdir("env_map") if (p.endswith(".exr") or p.endswith(".hdr"))]
        self.menu = None
        self.mode = None

        # 环境光旋转角（单位：度）
        self.env_yaw = 0.0    # 左右转动太阳
        self.env_pitch = 0.0  # 上下抬升太阳

        self.env_intensity = 1.0

        self.need_update = True

        self.full_pc = self.render_kwargs.get("pc", None)
        self.crop_enable = False
        if self.full_pc is not None and hasattr(self.full_pc, "get_xyz") and self.full_pc.get_xyz.numel() > 0:
            xyz = self.full_pc.get_xyz.detach().cpu()
            mins = xyz.min(0).values.numpy()
            maxs = xyz.max(0).values.numpy()
            self.crop_x_min_limit = float(mins[0])
            self.crop_x_max_limit = float(maxs[0])
            self.crop_y_min_limit = float(mins[1])
            self.crop_y_max_limit = float(maxs[1])
            self.crop_z_min_limit = float(mins[2])
            self.crop_z_max_limit = float(maxs[2])
            self.crop_x_min = self.crop_x_min_limit
            self.crop_x_max = self.crop_x_max_limit
            self.crop_y_min = self.crop_y_min_limit
            self.crop_y_max = self.crop_y_max_limit
            self.crop_z_min = self.crop_z_min_limit
            self.crop_z_max = self.crop_z_max_limit
        else:
            self.crop_x_min_limit = -1.0
            self.crop_x_max_limit = 1.0
            self.crop_y_min_limit = -1.0
            self.crop_y_max_limit = 1.0
            self.crop_z_min_limit = -1.0
            self.crop_z_max_limit = 1.0
            self.crop_x_min = self.crop_x_min_limit
            self.crop_x_max = self.crop_x_max_limit
            self.crop_y_min = self.crop_y_min_limit
            self.crop_y_max = self.crop_y_max_limit
            self.crop_z_min = self.crop_z_min_limit
            self.crop_z_max = self.crop_z_max_limit

        self.step()
        self.mode = "pbr_env"
        dpg.create_context()
        self.register_dpg()

    def update_env_rotation(self):
        env_light = self.render_kwargs["dict_params"].get("env_light", None)
        if env_light is None or not hasattr(env_light, "transform"):
            return

        yaw = np.radians(self.env_yaw)
        pitch = np.radians(self.env_pitch)

        Ry = torch.tensor([
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0,        1.0, 0.0        ],
            [-np.sin(yaw), 0.0, np.cos(yaw)]
        ], dtype=torch.float32, device="cuda")

        Rx = torch.tensor([
            [1.0, 0.0,            0.0           ],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch),  np.cos(pitch)]
        ], dtype=torch.float32, device="cuda")

        R_env = Ry @ Rx
        env_light.transform = R_env

    def __del__(self):
        dpg.destroy_context()

    def get_buffer(self, render_results, mode=None):
        if render_results is None or mode is None:
            output = torch.ones(self.H, self.W, 3, dtype=torch.float32, device='cuda').detach().cpu().numpy()
        else:
            output = render_results[mode]

            if mode == "depth":
                output = (output - output.min()) / (output.max() - output.min())
            elif mode == "num_contrib":
                output = output.clamp_max(1000) / 1000

            if len(output.shape) == 2:
                output = output[None]
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)
            if "normal" in mode:
                opacity = render_results["opacity"]
                output = output * 0.5 + 0.5 * opacity
            if (self.H, self.W) != tuple(output.shape[1:]):
                output = self.resize_fn(output)

            output = output.permute(1, 2, 0).contiguous().detach().cpu().numpy()
        return output

    @property
    def custom_cam(self):
        w2c = self.cam.view
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        down = self.downsample
        H, W = self.H // down, self.W // down
        fovy = self.cam.fovy * np.pi / 180
        fovx = fovy * W / H
        custom_cam = Camera(colmap_id=0, R=R, T=-T,
                            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                            image=torch.zeros(3, H, W), image_name=None, uid=0)
        return custom_cam

    def apply_crop(self):
        if self.full_pc is None:
            return
        if not self.crop_enable:
            self.render_kwargs["pc"] = self.full_pc
            return
        pc = self.full_pc
        xyz = pc.get_xyz
        if xyz.numel() == 0:
            self.render_kwargs["pc"] = pc
            return
        bounds_min = torch.tensor(
            [self.crop_x_min, self.crop_y_min, self.crop_z_min],
            dtype=xyz.dtype,
            device=xyz.device,
        )
        bounds_max = torch.tensor(
            [self.crop_x_max, self.crop_y_max, self.crop_z_max],
            dtype=xyz.dtype,
            device=xyz.device,
        )
        mask = (xyz >= bounds_min) & (xyz <= bounds_max)
        mask = mask.all(dim=1)
        if not mask.any():
            self.render_kwargs["pc"] = pc
            return
        new_pc = GaussianModel(pc.max_sh_degree, render_type=pc.render_type)
        for name in pc.attribute_names:
            attr = getattr(pc, "_" + name)
            if attr.numel() == 0:
                setattr(new_pc, "_" + name, attr)
            else:
                setattr(new_pc, "_" + name, attr[mask].clone())
        if hasattr(pc, "_visibility_tracing") and pc._visibility_tracing is not None:
            new_pc._visibility_tracing = pc._visibility_tracing[mask].clone()
        if hasattr(pc, "_incident_dirs") and pc._incident_dirs is not None:
            new_pc._incident_dirs = pc._incident_dirs[mask].clone()
        if hasattr(pc, "_incident_areas") and pc._incident_areas is not None:
            new_pc._incident_areas = pc._incident_areas[mask].clone()
        new_pc.base_color_scale = pc.base_color_scale
        self.render_kwargs["pc"] = new_pc

    @torch.no_grad()
    def render(self):
        self.step()
        dpg.render_dearpygui_frame()

    def step(self):
        if self.need_update:
            self.apply_crop()
            self.need_update = False
        self.start.record()
        render_pkg = self.render_fn(viewpoint_camera=self.custom_cam, **self.render_kwargs)
        self.end.record()
        torch.cuda.synchronize()
        t = self.start.elapsed_time(self.end)

        buffer1 = self.get_buffer(render_pkg, self.mode)
        self.render_buffer = buffer1

        if t == 0:
            fps = 0
        else:
            fps = int(1000 / t)

        if self.menu is None:
            self.menu = [k for k, v in render_pkg.items() if
                         isinstance(v, torch.Tensor) and np.array(v.shape).prod() % (self.H * self.W) == 0]
        else:
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({fps} FPS)')
            dpg.set_value("_texture", self.render_buffer)

    def register_dpg(self):

        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=200):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # envmap combo
                def callback_change_envmap(sender, app_data):
                    if app_data == "original":
                        self.render_kwargs["dict_params"]["env_light"] = pbr_kwargs.get("env_light", None)
                    else:
                        self.render_kwargs["dict_params"]["env_light"] = EnvLight(path=app_data, scale=self.env_intensity)
                    # 切换环境图时重置旋转
                    self.env_yaw = 0.0
                    self.env_pitch = 0.0
                    self.update_env_rotation()
                    self.need_update = True

                dpg.add_combo(self.envmaps, label='envmap', default_value=self.envmaps[0], callback=callback_change_envmap)

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(self.menu, label='mode', default_value=self.mode, callback=callback_change_mode)

                def callback_set_downsample(sender, app_data):
                    self.downsample = app_data
                    self.need_update = True

                dpg.add_slider_int(label="Downsample", min_value=1, max_value=8, format="x%d",
                                   default_value=self.downsample, callback=callback_set_downsample)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg",
                                   default_value=self.cam.fovy, callback=callback_set_fovy)

                def callback_set_env_intensity(sender, app_data):
                    env = self.render_kwargs["dict_params"].get("env_light", None)
                    self.env_intensity = app_data
                    if isinstance(env, EnvLight) and getattr(env, "envmap_path", None) is not None:
                        env.scale = self.env_intensity
                        env.envmap = env.load(env.envmap_path, scale=self.env_intensity, device=env.device)
                    self.update_env_rotation()
                    self.need_update = True

                dpg.add_slider_float(label="Env Intensity", min_value=0.1, max_value=5.0,
                                     default_value=self.env_intensity, callback=callback_set_env_intensity)

                def callback_set_crop_enable(sender, app_data):
                    self.crop_enable = app_data
                    self.need_update = True

                dpg.add_checkbox(label="Enable Crop Box", default_value=self.crop_enable, callback=callback_set_crop_enable)

                def callback_set_crop_x_min(sender, app_data):
                    if app_data > self.crop_x_max:
                        app_data = self.crop_x_max
                    self.crop_x_min = app_data
                    self.need_update = True

                def callback_set_crop_x_max(sender, app_data):
                    if app_data < self.crop_x_min:
                        app_data = self.crop_x_min
                    self.crop_x_max = app_data
                    self.need_update = True

                def callback_set_crop_y_min(sender, app_data):
                    if app_data > self.crop_y_max:
                        app_data = self.crop_y_max
                    self.crop_y_min = app_data
                    self.need_update = True

                def callback_set_crop_y_max(sender, app_data):
                    if app_data < self.crop_y_min:
                        app_data = self.crop_y_min
                    self.crop_y_max = app_data
                    self.need_update = True

                def callback_set_crop_z_min(sender, app_data):
                    if app_data > self.crop_z_max:
                        app_data = self.crop_z_max
                    self.crop_z_min = app_data
                    self.need_update = True

                def callback_set_crop_z_max(sender, app_data):
                    if app_data < self.crop_z_min:
                        app_data = self.crop_z_min
                    self.crop_z_max = app_data
                    self.need_update = True

                dpg.add_slider_float(label="Crop X Min", min_value=self.crop_x_min_limit, max_value=self.crop_x_max_limit,
                                     default_value=self.crop_x_min, callback=callback_set_crop_x_min)
                dpg.add_slider_float(label="Crop X Max", min_value=self.crop_x_min_limit, max_value=self.crop_x_max_limit,
                                     default_value=self.crop_x_max, callback=callback_set_crop_x_max)
                dpg.add_slider_float(label="Crop Y Min", min_value=self.crop_y_min_limit, max_value=self.crop_y_max_limit,
                                     default_value=self.crop_y_min, callback=callback_set_crop_y_min)
                dpg.add_slider_float(label="Crop Y Max", min_value=self.crop_y_min_limit, max_value=self.crop_y_max_limit,
                                     default_value=self.crop_y_max, callback=callback_set_crop_y_max)
                dpg.add_slider_float(label="Crop Z Min", min_value=self.crop_z_min_limit, max_value=self.crop_z_max_limit,
                                     default_value=self.crop_z_min, callback=callback_set_crop_z_min)
                dpg.add_slider_float(label="Crop Z Max", min_value=self.crop_z_min_limit, max_value=self.crop_z_max_limit,
                                     default_value=self.crop_z_max, callback=callback_set_crop_z_max)

                # 环境光旋转：左右（Yaw）
                def callback_set_env_yaw(sender, app_data):
                    self.env_yaw = app_data
                    self.update_env_rotation()
                    self.need_update = True

                dpg.add_slider_float(label="Env Yaw (deg)", min_value=-180.0, max_value=180.0,
                                     default_value=self.env_yaw, callback=callback_set_env_yaw)

                # 环境光旋转：上下（Pitch）
                def callback_set_env_pitch(sender, app_data):
                    self.env_pitch = app_data
                    self.update_env_rotation()
                    self.need_update = True

                dpg.add_slider_float(label="Env Pitch (deg)", min_value=-90.0, max_value=90.0,
                                     default_value=self.env_pitch, callback=callback_set_env_pitch)

                def callback_save_image(sender, app_data):
                    img = torch.from_numpy(self.render_buffer).permute(2, 0, 1)
                    img = img.clamp(0.0, 1.0)
                    os.makedirs("env_gui_output", exist_ok=True)
                    filename = os.path.join("env_gui_output", f"screenshot_{int(time.time())}.png")
                    save_image(img, filename)

                dpg.add_button(label="Save Image", callback=callback_save_image)

                def callback_save_cropped_ply(sender, app_data):
                    if self.full_pc is None:
                        return
                    current_pc = self.render_kwargs.get("pc", self.full_pc)
                    os.makedirs("env_gui_output", exist_ok=True)
                    filename = os.path.join("env_gui_output", f"cropped_point_cloud_{int(time.time())}.ply")
                    current_pc.save_ply(filename)

                dpg.add_button(label="Save Cropped Point Cloud", callback=callback_save_cropped_ply)

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
                update_relighting(self.cam.pose)

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
                update_relighting(self.cam.pose)

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
                update_relighting(self.cam.pose)

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

        dpg.create_viewport(title='3D Gaussian Rendering Viewer', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument('-t', '--type', choices=['render','neilf'], default='render')
    parser.add_argument('--bake', action='store_true', default=False, help="Bake the visibility and refine.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="resume from checkpoint")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument('--hdr2ldr', action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--emissivity", type=float, default=None)
    parser.add_argument("--emissivity_mode", choices=["external", "auto"], default="external")
    parser.add_argument("--thermal_scale", type=float, default=1e-8)

    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    
    env_light = None
    pbr_kwargs = dict()
    pbr_kwargs['sample_num'] = pipe.sample_num
    checkpoints = glob.glob(os.path.join(args.model_path, "chkpnt*.pth"))
    if args.checkpoint is not None or len(checkpoints) > 0:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
        else:
            checkpoint = sorted(checkpoints, key=lambda x: int(x.split("chkpnt")[-1].split(".")[0]))[-1]
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.create_from_ckpt(checkpoint, restore_optimizer=False)

        env_checkpoint = checkpoint.split("chkpnt")[0] + "env_light_chkpnt" + checkpoint.split("chkpnt")[-1]
        if os.path.exists(env_checkpoint):
            env_light = DirectLightMap(dataset.global_shs_degree)
            env_light.create_from_ckpt(env_checkpoint, restore_optimizer=False)

            pbr_kwargs["env_light"] = env_light
        else:
            print("cannot find env_light_checkpoint at {}, and env light will be ignore.".format(env_checkpoint))
    else:
        if args.iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
        else:
            loaded_iter = args.loaded_iter
        gaussians.load_ply(
            os.path.join(args.model_path, "point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))

        env_ckpts = glob.glob(os.path.join(args.model_path, "env_light_chkpnt*.pth"))
        if len(env_ckpts) > 0:
            env_checkpoint = sorted(env_ckpts, key=lambda x: int(x.split("env_light_chkpnt")[-1].split(".")[0]))[-1]
            env_light = DirectLightMap(dataset.global_shs_degree)
            env_light.create_from_ckpt(env_checkpoint, restore_optimizer=False)
            pbr_kwargs["env_light"] = env_light
        else:
            print("cannot find env_light_checkpoint in {}, and env light will be ignore.".format(args.model_path))

    gaussians.update_visibility(pipe.sample_num)

    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if os.path.exists(os.path.join(args.model_path, "cameras.json")):
        with open(os.path.join(args.model_path, "cameras.json"), 'r') as file:
            cam = JSON_to_camera(json.load(file)[0])
        c2w = cam.c2w.detach().cpu().numpy()
        H, W = int(cam.image_height / args.scale), int(cam.image_width / args.scale)
        fovy = cam.FoVy

        if fovy is None:
            fovy = focal2fov(cam.fy, cam.image_height)
    else:
        H, W = 800, 800
        fovy = 50 * np.pi / 180
        c2w = np.array([
            [0.0, 0.0, -1.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    center = gaussians.get_xyz.mean(dim=0).detach().cpu().numpy()

    dict_params = {
        "env_light": env_light,
        "sample_num": args.sample_num,
    }
    if args.temperature is not None:
        dict_params["object_temperature"] = args.temperature
    if args.emissivity is not None:
        dict_params["emissivity"] = args.emissivity
    dict_params["emissivity_mode"] = args.emissivity_mode
    dict_params["thermal_scale"] = args.thermal_scale

    render_kwargs = {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": background,
        "is_training": False,
        "dict_params": dict_params,
        "bake": args.bake
    }

    windows = GUI(H, W, fovy,
                  c2w=c2w, center=center,
                  render_fn=render_fn, render_kwargs=render_kwargs,
                  mode='pbr')

    while True:
        windows.render()
