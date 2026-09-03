import json
import numpy as np
import argparse

def look_at(eye, target, up):
    """
    计算一个“look at”视图矩阵（从相机到世界）。
    我们实际需要的是它的逆矩阵（从世界到相机），但由于 relighting.py 的历史原因，
    它期望接收一个从相机到世界的矩阵（C2W），然后在内部将其转换为W2C。
    因此，我们直接构建并返回C2W矩阵。
    """
    # 计算前向、右向和上向向量
    forward = eye - target
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    
    new_up = np.cross(forward, right)
    new_up = new_up / np.linalg.norm(new_up)

    # 构建从相机到世界的变换矩阵 (C2W)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = new_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    
    return c2w

def generate_circular_trajectory(num_frames, radius, height, target):
    """
    生成一个围绕目标点的圆形相机轨迹。
    """
    trajectory = {}
    up_vector = np.array([0, 1, 0]) # 定义“上”方向

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        # 计算相机在XZ平面上的位置
        eye_x = target[0] + radius * np.sin(angle)
        eye_y = target[1] + height
        eye_z = target[2] + radius * np.cos(angle)
        
        eye_position = np.array([eye_x, eye_y, eye_z])
        
        # 计算C2W矩阵
        c2w_matrix = look_at(eye_position, target, up_vector)
        
        # relighting.py 需要一个稍微不同的格式，它将C2W矩阵的Y和Z轴翻转
        # 这是为了匹配它内部从Colmap格式转换的逻辑
        c2w_matrix[:3, 1:3] *= -1

        trajectory[str(i)] = c2w_matrix.flatten().tolist()
        
    return trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a circular camera trajectory for relighting.")
    parser.add_argument('--output', type=str, default='configs/chair/trajectory.json', help='Output path for the trajectory JSON file.')
    parser.add_argument('--num_frames', type=int, default=300, help='Number of frames in the trajectory.')
    parser.add_argument('--radius', type=float, default=3.5, help='Radius of the circular path.')
    parser.add_argument('--height', type=float, default=2.0, help='Height of the camera from the target.')
    parser.add_argument('--target', type=float, nargs=3, default=[0, 0, 0], help='The 3D point the camera looks at.')
    parser.add_argument('--width', type=int, default=800, help='Camera width.')
    parser.add_argument('--height_cam', type=int, default=800, help='Camera height.')
    parser.add_argument('--fov', type=int, default=50, help='Camera field of view.')

    args = parser.parse_args()

    print(f"Generating a circular trajectory with {args.num_frames} frames...")
    
    # 生成轨迹
    trajectory_data = generate_circular_trajectory(args.num_frames, args.radius, args.height, np.array(args.target))
    
    # 准备最终的JSON结构
    output_dict = {
        "camera": {
            "width": args.width,
            "height": args.height_cam,
            "fov": args.fov
        },
        "trajectory": trajectory_data
    }
    
    # 写入文件
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=2)
        
    print(f"Trajectory successfully saved to {args.output}")
