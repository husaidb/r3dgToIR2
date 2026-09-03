import os
import argparse
from PIL import Image
import rembg
from tqdm import tqdm
import io

def create_masks(source_path):
    """
    为给定的源路径中的图像生成掩码。
    读取 'images' 文件夹，并将生成的掩码保存到 'masks' 文件夹。
    """
    # 定义输入和输出文件夹的路径
    image_dir = os.path.join(source_path, "images")
    mask_dir = os.path.join(source_path, "masks")

    # 检查输入文件夹是否存在
    if not os.path.exists(image_dir):
        print(f"错误：找不到图像文件夹 '{image_dir}'")
        return

    # 创建输出文件夹（如果不存在）
    os.makedirs(mask_dir, exist_ok=True)
    print(f"掩码将保存到: '{mask_dir}'")

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"在 '{image_dir}' 中没有找到图像文件。")
        return

    print(f"找到 {len(image_files)} 张图像，开始生成掩码...")

    # 遍历所有图像并生成掩码
    for filename in tqdm(image_files, desc="正在生成掩码"):
        input_path = os.path.join(image_dir, filename)
        output_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + ".png") # 确保掩码为png格式

        # 读取输入图像
        with open(input_path, "rb") as i:
            input_bytes = i.read()

            # 使用 rembg 移除背景
            output_bytes = rembg.remove(input_bytes)

            # 将输出转换为 PIL Image 对象
            result_image = Image.open(io.BytesIO(output_bytes))

            mask = result_image.getchannel('A')
            mask.save(output_path)

    print("所有掩码文件已成功生成！")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="使用 rembg 为图像数据集生成掩码。")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="包含 'images' 文件夹的数据集源路径。")
    
    args = parser.parse_args()
    
    # 运行主函数
    create_masks(args.source_path)
