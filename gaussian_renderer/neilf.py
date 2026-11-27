import math
import os
import torch
import numpy as np
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, bilateral_smooth_loss, second_order_edge_aware_loss, tv_loss, first_order_edge_aware_loss, first_order_loss, first_order_edge_aware_norm_loss
from utils.image_utils import psnr
from utils.graphics_utils import fibonacci_sphere_sampling, rgb_to_srgb, srgb_to_rgb
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    direct_light_env_light = dict_params.get("env_light")
    # 中文注释：从外部参数读取物体温度（摄氏度）与发射率，用于在 PBR 方程中显式加入热辐射项
    object_temperature = dict_params.get("object_temperature", None)
    emissivity = dict_params.get("emissivity", None)
    emissivity_mode = dict_params.get("emissivity_mode", "external")
    # 中文注释：热辐射强度的标定系数（简化 Stefan–Boltzmann），用于把 MWIR 能量对齐到图像数值范围
    #可进行修改，只是一个近似值，https://www.spectralcalc.com/blackbody_calculator/blackbody.php可算辐射值
    #然后除以T^4得到thermal_scale
    thermal_scale = dict_params.get("thermal_scale", 1e-11)
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    base_color = pc.get_base_color
    N = means3D.shape[0]
    if dict_params.get("metallic_from_model", False) and getattr(pc, "use_pbr", False):
        metallic = pc.get_metalness.view(N, 1)
    elif dict_params.get("metal_prob", None) is not None:
        metal_prob = dict_params["metal_prob"]
        assert metal_prob.shape[0] == N, \
            f"metal_prob length {metal_prob.shape[0]} != num gaussians {N}; " \
            f"checkpoint/mask mismatch or densification is on"
        metallic = metal_prob.to(base_color.device, base_color.dtype).reshape(N, 1)
    else:
        metallic = torch.full((N, 1), float(dict_params.get("metallic", 0.0)),
                              device=base_color.device, dtype=base_color.dtype)
    roughness = pc.get_roughness
    normal = pc.get_normal
    incidents = pc.get_incidents  # incident shs
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)
    
    if is_training:
        brdf_color, extra_results = rendering_equation(
            base_color, roughness, normal.detach(), viewdirs, incidents,
            direct_light_env_light, visibility_precompute=pc._visibility_tracing, 
            object_temperature=object_temperature, emissivity=emissivity, thermal_scale=thermal_scale, emissivity_mode=emissivity_mode,
            incident_dirs_precompute=pc._incident_dirs, incident_areas_precompute=pc._incident_areas, metallic=metallic)
    else:
        chunk_size = 100000
        brdf_color = []
        extra_results = []
        for i in range(0, means3D.shape[0], chunk_size):
            _brdf_color, _extra_results = rendering_equation(
                base_color[i:i + chunk_size], roughness[i:i + chunk_size], 
                normal[i:i + chunk_size].detach(), viewdirs[i:i + chunk_size], incidents[i:i + chunk_size],
                direct_light_env_light, 
                object_temperature=object_temperature, emissivity=emissivity, thermal_scale=thermal_scale, emissivity_mode=emissivity_mode,
                visibility_precompute=pc._visibility_tracing[i:i + chunk_size], 
                incident_dirs_precompute=pc._incident_dirs[i:i + chunk_size], 
                incident_areas_precompute=pc._incident_areas[i:i + chunk_size],
                metallic=metallic[i:i + chunk_size])
            brdf_color.append(_brdf_color)
            extra_results.append(_extra_results)
        brdf_color = torch.cat(brdf_color, dim=0)
        extra_results = {k: torch.cat([_extra_results[k] for _extra_results in extra_results], dim=0) for k in extra_results[0]}
        torch.cuda.empty_cache()

    # 材质先验：逐点 eps 的区间软约束，金属/介电分支按 metalness 软加权（背景 metalness=0，走介电区间）
    material_prior = None
    if is_training and ("emissivity" in extra_results):
        eps_pt = extra_results["emissivity"].mean(dim=-1, keepdim=True)  # [N,1]

        def _interval_hinge(e, lo, hi):
            return F.relu(lo - e).pow(2) + F.relu(e - hi).pow(2)

        material_prior = (
            metallic * _interval_hinge(eps_pt, dict_params.get("eps_metal_min", 0.05),
                                       dict_params.get("eps_metal_max", 0.65))
            + (1 - metallic) * _interval_hinge(eps_pt, dict_params.get("eps_diel_min", 0.65),
                                               dict_params.get("eps_diel_max", 0.98))
        ).mean()

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()
    
    # 中文注释：在特征中加入“热辐射”作为独立通道，便于栅格化后输出 emission-only 图像
    if is_training:
        features = torch.cat([depths, depths2, brdf_color, normal, base_color, roughness, 
                              extra_results["diffuse_light"],
                              extra_results["emission"],  # 中文注释：新增热辐射特征（3通道）
                              extra_results["incident_visibility"].mean(-2)], dim=-1)
    else:
        features = torch.cat([depths, depths2, brdf_color, normal, base_color, roughness,
                              extra_results["diffuse_light"],
                              extra_results["emission"],  # 中文注释：新增热辐射特征（3通道）
                              extra_results["specular"], 
                              extra_results["incident_lights"].mean(-2),
                              extra_results["local_incident_lights"].mean(-2),
                              extra_results["global_incident_lights"].mean(-2),
                              extra_results["incident_visibility"].mean(-2)], dim=-1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask
    feature_dict = {}
    if is_training:
        rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, \
            rendered_roughness, rendered_diffuse, rendered_emission, rendered_visibility \
            = rendered_feature.split([1, 1, 3, 3, 3, 1, 3, 3, 1], dim=0)
        feature_dict.update({"base_color": rgb_to_srgb(rendered_base_color),
                             "roughness": rendered_roughness,
                             "diffuse": rgb_to_srgb(rendered_diffuse),
                             "emission": rgb_to_srgb(rendered_emission),  # 中文注释：输出热辐射图（仅物体）
                             "visibility": rendered_visibility
                             })
    else:
        rendered_depth, rendered_depth2, rendered_pbr, rendered_normal, rendered_base_color, rendered_roughness, \
            rendered_diffuse, rendered_emission, rendered_specular, rendered_light, rendered_local_light, rendered_global_light, rendered_visibility \
            = rendered_feature.split([1, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1], dim=0)
        feature_dict.update({
                             "base_color": rgb_to_srgb(rendered_base_color),
                             "roughness": rendered_roughness,
                             "diffuse": rgb_to_srgb(rendered_diffuse),
                             "emission": rgb_to_srgb(rendered_emission),  # 中文注释：输出热辐射图（仅物体）
                             "specular": rgb_to_srgb(rendered_specular),
                             "lights": rgb_to_srgb(rendered_light),
                             "local_lights": rgb_to_srgb(rendered_local_light),
                             "global_lights": rgb_to_srgb(rendered_global_light),
                             "visibility": rendered_visibility,
                             })
    rendered_var = rendered_depth2 - rendered_depth.square()

    pbr = rendered_pbr
    rendered_pbr = pbr * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]
    # 中文注释：仅热辐射图像（不含反射），只对物体区域进行合成，背景保持原背景色
    rendered_emission = rendered_emission * rendered_opacity + (1 - rendered_opacity) * bg_color[:, None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_image,
               "depth": rendered_depth,
               "depth_var": rendered_var,
               "pbr": rgb_to_srgb(rendered_pbr),
               "emission": rgb_to_srgb(rendered_emission),  # 中文注释：提供仅自发辐射的渲染结果
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib
               }

    results.update(feature_dict)
    results["diffuse_light"] = extra_results["diffuse_light"]
    if material_prior is not None:
        results["material_prior_loss"] = material_prior
    try:
        results["env"] = direct_light_env_light.get_env
    except:
        pass
    
    if not is_training:
        directions = viewpoint_camera.get_world_directions()
        direct_env = direct_light_env_light.direct_light(directions.permute(1, 2, 0)).permute(2, 0, 1)
        results["render_env"] = rendered_image + (1 - rendered_opacity) * rgb_to_srgb(direct_env)
        results["pbr_env"] = rgb_to_srgb(pbr * rendered_opacity + (1 - rendered_opacity) * direct_env)
        # 训练时下方要注释掉
        pbr_env = pbr * rendered_opacity + (1 - rendered_opacity) * direct_env
        weights = torch.tensor([0.299, 0.587, 0.114], device=pbr_env.device, dtype=pbr_env.dtype).view(3, 1, 1)
        gray = (pbr_env * weights).sum(dim=0, keepdim=True)
        pbr_env_gray = gray.repeat(3, 1, 1)
        results["pbr_env"] = rgb_to_srgb(pbr_env_gray)
        results["env_only"] = rgb_to_srgb(direct_env)
        
    return results


def calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_pbr = results["pbr"]
    rendered_opacity = results["opacity"]
    rendered_base_color = results["base_color"]
    rendered_roughness = results["roughness"]
    rendered_diffuse = results["diffuse"]
    rendered_emission = results.get("emission", None)

    gt_image = viewpoint_camera.original_image.cuda()
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
    ssim_val_pbr = ssim(rendered_pbr, gt_image)
    tb_dict["l1_pbr"] = Ll1_pbr.item()
    tb_dict["ssim_pbr"] = ssim_val_pbr.item()
    tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
    loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
    loss = loss + opt.lambda_pbr * loss_pbr
    
    # 中文注释：分离损失（可选）。约束“反射部分”与“去除热辐射后的 GT”一致，促进反射/发射的分解
    #若要关闭该损失，将lambda_emission_sep吧0.03设为0即可
    lambda_emission_sep = getattr(opt, "lambda_emission_sep", 0.03)
    if (lambda_emission_sep > 0) and (rendered_emission is not None):
        reflect_only_pred = (rendered_pbr - rendered_emission).clamp(0, 1)
        reflect_only_gt = (gt_image - rendered_emission).clamp(0, 1)
        loss_emission_sep = F.l1_loss(reflect_only_pred, reflect_only_gt)
        tb_dict["loss_emission_sep"] = loss_emission_sep.item()
        loss = loss + lambda_emission_sep * loss_emission_sep

    lambda_material_prior = getattr(opt, "lambda_material_prior", 0.0)
    material_prior = results.get("material_prior_loss", None)
    if (lambda_material_prior > 0) and (material_prior is not None):
        tb_dict["loss_material_prior"] = material_prior.item()
        loss = loss + lambda_material_prior * material_prior

    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['pseudo_normal']
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_mvs_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        depth_mask = (gt_depth > 0).float()
        mvs_normal = viewpoint_camera.normal.cuda()
        loss_normal_mvs_depth = F.mse_loss(
            rendered_normal * depth_mask, mvs_normal * depth_mask)
        tb_dict["loss_normal_mvs_depth"] = loss_normal_mvs_depth.item()
        loss = loss + opt.lambda_normal_mvs_depth * loss_normal_mvs_depth

    if opt.lambda_light > 0:
        diffuse_light = results["diffuse_light"]
        mean_light = diffuse_light.mean(-1, keepdim=True).expand_as(diffuse_light)
        loss_light = F.l1_loss(diffuse_light, mean_light)
        tb_dict["loss_light"] = loss_light.item()
        loss = loss + opt.lambda_light * loss_light

    if opt.lambda_base_color_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        # loss_base_color_smooth = second_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

    if opt.lambda_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        # loss_roughness_smooth = second_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
        tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
        loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth
    
    
    if opt.lambda_light_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_light_smooth = first_order_edge_aware_loss(rendered_diffuse * image_mask, rendered_normal)
        # loss_light_smooth = second_order_edge_aware_loss(rendered_diffuse * image_mask, gt_image)
        tb_dict["loss_light_smooth"] = loss_light_smooth.item()
        loss = loss + opt.lambda_light_smooth * loss_light_smooth
        
    if opt.lambda_env_smooth > 0:
        env = direct_light_env_light.get_env
        loss_env_smooth = tv_loss(env[0].permute(2, 0, 1))
        tb_dict["loss_env_smooth"] = loss_env_smooth.item()
        loss = loss + opt.lambda_env_smooth * loss_env_smooth
    
    if opt.lambda_normal_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        # loss_normal_smooth = second_order_edge_aware_loss(rendered_normal * image_mask, gt_image)
        loss_normal_smooth = tv_loss(rendered_normal * image_mask)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth
    
    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render_neilf(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None, **kwargs):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, direct_light_env_light=dict_params['env_light'])
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results


_SPEC_LUT_CACHE = {}


def _get_spec_lut(device):
    if device not in _SPEC_LUT_CACHE:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "computebrdf", "spec_lut.npz")
        data = np.load(path)
        _SPEC_LUT_CACHE[device] = tuple(
            torch.as_tensor(data[k], dtype=torch.float32, device=device)
            for k in ("A", "B", "roughness", "ndv"))
    return _SPEC_LUT_CACHE[device]


def lookup_spec_ab(roughness, ndv):
    """Bilinear lookup of the offline Karis A/B LUT, rho_spec = F0 * A + B.

    Generated by computebrdf/gen_spec_lut.py from the same simplified Disney BRDF as
    GGX_specular. A, B are tabulated on (N.V, roughness).
    """
    A, B, r_ax, v_ax = _get_spec_lut(roughness.device)
    shape = roughness.shape
    r = roughness.reshape(-1).clamp(r_ax[0], r_ax[-1])
    v = ndv.reshape(-1).clamp(v_ax[0], v_ax[-1])
    iv = (torch.searchsorted(r_ax, r, right=True) - 1).clamp_(0, r_ax.numel() - 2)
    iu = (torch.searchsorted(v_ax, v, right=True) - 1).clamp_(0, v_ax.numel() - 2)
    fr = ((r - r_ax[iv]) / (r_ax[iv + 1] - r_ax[iv])).clamp_(0, 1)
    fv = ((v - v_ax[iu]) / (v_ax[iu + 1] - v_ax[iu])).clamp_(0, 1)
    out = []
    for M in (A, B):
        m = (M[iu, iv] * (1 - fv) * (1 - fr) + M[iu, iv + 1] * (1 - fv) * fr
             + M[iu + 1, iv] * fv * (1 - fr) + M[iu + 1, iv + 1] * fv * fr)
        out.append(m.view(shape))
    return out


def rendering_equation(base_color, roughness, normals, viewdirs,
                              incidents, direct_light_env_light=None,
                              visibility_precompute=None, incident_dirs_precompute=None, incident_areas_precompute=None,
                              object_temperature=None, emissivity=None, thermal_scale=1e-8, emissivity_mode="external",
                              metallic=None):
    incident_dirs, incident_areas = incident_dirs_precompute, incident_areas_precompute

    deg = int(np.sqrt(incidents.shape[1]) - 1)
    #全局光
    global_incident_lights = direct_light_env_light.direct_light(incident_dirs)
    #周围反射光，可设为0 
    local_incident_lights = eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    # local_incident_lights = torch.zeros_like(global_incident_lights)
    incident_visibility = visibility_precompute
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = local_incident_lights + global_incident_lights

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    # 计算漫反射项
    #可以选为固定值来简化中波红外下的漫反射，如0.05或是其他较小的值，剩下的将由自发辐射和镜面反射来拟合
    # rho = 0.05  # 例如 5% 反射率
    # f_d = rho / np.pi
    #rho = 0.5
    # 或者f_d = (rho * base_color[:, None]) / np.pi
    #除以pi是因为我们假设漫反射率是与入射角度无关的常数，而不是与入射角度相关的函数
    if metallic is None:
        metallic = torch.zeros_like(base_color[:, :1])
    F0 = 0.04 * (1 - metallic) + base_color * metallic      # [N,3]
    f_d = (base_color * (1 - metallic))[:, None] / np.pi    # [N,1,3]
    f_s = GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=F0)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    specular = ((f_s) * transport).mean(dim=-2)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)
    diffuse_light = transport.mean(dim=-2)

    # 中文注释：显式加入热辐射项（Stefan–Boltzmann 简化），避免把热能量错误归因到 base_color 或环境光
    # L_emit = emissivity * thermal_scale * (T_K)^4，其中 T_K = T_C + 273.15
    emit = torch.zeros_like(pbr)
    eps = None
    if emissivity_mode == "analytic" and emissivity is None:
        # 集总近似：漫反射率取 base_color，镜面反射率直接取 F0=0.04，忽略粗糙度与视角
        rho = (base_color + 0.04 * (1 - metallic)).clamp(0, 1)
        eps = 1.0 - rho
    elif emissivity_mode == "lut" and emissivity is None:
        # 镜面反射率查表 rho_spec = F0*A + B；N.V 固定为 1，保证每个高斯点只有一个发射率
        A, B = lookup_spec_ab(roughness, torch.ones_like(roughness))
        rho = (base_color * (1 - metallic) + F0 * A + B).clamp(0, 1)
        eps = 1.0 - rho
    elif emissivity_mode == "auto" and emissivity is None:
        transport_unit = incident_areas * n_d_i
        R_approx = ((f_d + f_s) * transport_unit).mean(dim=-2).clamp(0, 1)
        eps = 1.0 - R_approx
    else:
        eps = torch.tensor(emissivity, dtype=base_color.dtype, device=base_color.device) if emissivity is not None else None
    if (object_temperature is not None) and (eps is not None):
        temp_kelvin = torch.tensor(object_temperature + 273.15, dtype=base_color.dtype, device=base_color.device)
        emit_mono = eps * thermal_scale * torch.pow(temp_kelvin, 4)
        # 将单通道的热辐射强度扩展成一个三通道的 RGB 颜色。


        # # 如果效果不好把下面注释掉，换成上面的emit_mono
        #  # 中文注释：第一步——根据温度和发射率计算“辐射亮度”L_emit（相对物理量）
        # L_emit = eps * thermal_scale * torch.pow(temp_kelvin, 4)

        # # 中文注释：第二步——利用相机标定关系 L = p1 * DN + p2 反推灰度 DN
        # # 注意：p1, p2 需要根据具体相机/标定实验设置，这里是占位示例。
        # p1 = torch.tensor(0.4091, dtype=base_color.dtype, device=base_color.device)
        # p2 = torch.tensor(6.381, dtype=base_color.dtype, device=base_color.device)
        # DN = (L_emit - p2) / p1
        # DN = DN.clamp(0.0, 255.0)

        # # 中文注释：第三步——把 DN 归一化到 [0,1]，作为热辐射的灰度强度
        # emit_mono = DN / 255.0

        # # 将单通道的热辐射强度扩展成一个三通道的灰度颜色（与 PNG 输入保持一致）



        emit = emit_mono.expand_as(pbr)
        pbr = pbr + emit  # 中文注释：总辐射 = 反射 + 热辐射

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        "incident_visibility": incident_visibility,
        "diffuse_light": diffuse_light,
        "specular": specular,
        "emission": emit,  # 中文注释：为后续栅格化提供仅热辐射特征（3通道）
        "metalness": metallic,
    }
    if eps is not None and eps.dim() == 2:
        extra_results["emissivity"] = eps

    return pbr, extra_results

#  Cook-Torrance 镜面反射 BRDF (双向反射分布函数)
def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    F0 = fresnel[:, None, :] if isinstance(fresnel, torch.Tensor) and fresnel.dim() == 2 else fresnel
    frac0 = F0 + (1 - F0) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    # 下限只防除零：r>=0.09 时 nom 真值最小约 5e-8，1e-12 不会触发；原 1e-6 会压掉粗糙度 <0.15 的镜面反射瓣核心
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-12, 4 * np.pi)
    spec = frac / nom
    return spec
