import numpy as np

def integrate_slice(theta, brdf):
    """这是从 slice_reflectance.py 提取的积分核心函数"""
    keep = theta >= 0
    theta, brdf = theta[keep], brdf[keep]
    order = np.argsort(theta)
    theta, brdf = theta[order], brdf[order]
    integrand = brdf * np.cos(theta)[:, None] * np.sin(theta)[:, None]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    rho = trapz(integrand, theta, axis=0)
    rho = np.clip(2 * np.pi * rho, 0, 1)
    return rho

def main():
    # ================= 配置区 =================
    input_file = 'brdf_outgoing.txt'          # 你的原始文件名
    output_file = 'brdf_outgoing.txt'    # 输出的新文件名
    target_eps_increase = 0.05                # 想要每次增加的 eps 值
    # ==========================================

    # 1. 读取原始数据
    try:
        data = np.loadtxt(input_file, delimiter=',')
    except FileNotFoundError:
        print(f"❌ 找不到文件 {input_file}，请确认文件名和路径。")
        return

    theta_i, theta_o = data[:, 0], data[:, 2]
    
    # 自动判断使用哪个角度作为积分变量
    if np.ptp(theta_i) > np.ptp(theta_o):
        theta = theta_i
    else:
        theta = theta_o
        
    brdf_original = data[:, 4:7]

    # 2. 计算当前的 rho
    rho_current = integrate_slice(theta, brdf_original)
    
    # 取 RGB 平均来计算目标
    rho_current_avg = np.mean(rho_current)
    eps_current_avg = 1.0 - rho_current_avg

    print(f"📊 当前数据: 平均 rho = {rho_current_avg:.4f}, 平均 eps = {eps_current_avg:.4f}")

    # 3. 计算目标 rho
    target_eps_avg = eps_current_avg + target_eps_increase
    
    # 边界保护：eps 不能超过 1.0 (即 rho 不能小于 0)
    if target_eps_avg > 1.0:
        target_eps_avg = 1.0
        print("⚠️ 警告: 目标 eps 超过 1.0，已自动限制为 1.0 (纯黑吸收体)。")

    target_rho_avg = 1.0 - target_eps_avg

    # 4. 计算缩放系数 k
    # 因为 rho 和 BRDF 是线性关系，直接把 BRDF 乘以这个比例即可
    if rho_current_avg > 0:
        scale_factor = target_rho_avg / rho_current_avg
    else:
        scale_factor = 0.0

    print(f"🔧 为了将 eps 提高 {target_eps_increase}，BRDF 数据需要乘以系数: {scale_factor:.4f}")

    # 5. 缩放 BRDF 并保存
    brdf_new = brdf_original * scale_factor
    
    # 确保数值合法
    brdf_new = np.clip(brdf_new, 0, 1.0)

    new_data = np.hstack((data[:, :4], brdf_new))
    np.savetxt(output_file, new_data, delimiter=', ', fmt='%.5f')

    # 6. 验证结果
    rho_new = integrate_slice(theta, brdf_new)
    eps_new = 1.0 - rho_new
    print("-" * 40)
    print(f"✅ 已生成新文件: {output_file}")
    print(f"📈 新的 rho: {rho_new}")
    print(f"📈 新的 eps: {eps_new}")
    print("-" * 40)

if __name__ == "__main__":
    main()