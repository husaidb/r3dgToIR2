import numpy as np

def add_specular_to_brdf(input_file, output_file, 
                         theta_i_sign_flip=True, 
                         peak_extra=0.07, 
                         sigma=0.25):
    """
    在现有漫反射BRDF数据上添加高斯镜面反射峰
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        theta_i_sign_flip: 是否将theta_i取反（根据您的坐标约定）
        peak_extra: 高光峰值额外增加的BRDF值
        sigma: 高斯峰的标准差（弧度），越大变化越缓慢
    """
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            if len(parts) != 7:
                continue
            theta_i = float(parts[0])
            phi_i = float(parts[1])
            theta_o = float(parts[2])
            phi_o = float(parts[3])
            r, g, b = map(float, parts[4:7])
            data.append([theta_i, phi_i, theta_o, phi_o, r, g, b])
    
    # 实际入射角（根据约定取反）
    theta_i_actual = -theta_i if theta_i_sign_flip else theta_i
    
    # 镜面反射峰的位置（出射角）
    specular_theta_o = -theta_i_actual   # 因为反射角 = -入射角（法线为z轴）
    
    with open(output_file, 'w') as f:
        for row in data:
            theta_i, phi_i, theta_o, phi_o, r_orig, g_orig, b_orig = row
            
            # 计算高光增量（高斯函数）
            delta = peak_extra * np.exp(-((theta_o - specular_theta_o)**2) / (2 * sigma**2))
            
            # 新BRDF值（保持RGB比例不变，高光为白色）
            r_new = r_orig + delta
            g_new = g_orig + delta
            b_new = b_orig + delta
            
            f.write(f"{theta_i:.5f}, {phi_i:.5f}, {theta_o:.5f}, {phi_o:.5f}, "
                    f"{r_new:.6f}, {g_new:.6f}, {b_new:.6f}\n")
    
    print(f"处理完成，输出文件：{output_file}")
    print(f"高光中心 theta_o = {specular_theta_o:.5f} rad，峰值增量 {peak_extra}，标准差 {sigma} rad")

if __name__ == "__main__":
    # ===== 请根据您的实际情况修改以下参数 =====
    INPUT_FILE = "14353.txt"      # 原始数据文件名
    OUTPUT_FILE = "brdf_specular.txt"     # 输出文件名
    THETA_I_SIGN_FLIP = True              # 是否翻转theta_i符号（您的数据需要翻转）
    PEAK_EXTRA = 0.07                     # 高光峰值额外增量（总反射率最高约0.141+0.07=0.211）
    SIGMA = 0.25                          # 标准差（弧度），越大峰越宽越平滑
    # =======================================
    
    add_specular_to_brdf(INPUT_FILE, OUTPUT_FILE, 
                         theta_i_sign_flip=THETA_I_SIGN_FLIP,
                         peak_extra=PEAK_EXTRA, 
                         sigma=SIGMA)