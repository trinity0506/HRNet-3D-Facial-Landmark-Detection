import json
import csv
import SimpleITK as sitk
import numpy as np
import pandas as pd
from PIL import Image

def load_nifti(nifti_path):
    img = sitk.ReadImage(nifti_path)
    arr = sitk.GetArrayFromImage(img)  # shape (z, y, x)
    return img, arr

def xy_to_voxel(x_img, y_img, img_w, img_h, nifti_x, nifti_y, crop_offset=(294, 113)):
    """
    将图片坐标 (x_img, y_img) 映射为 NIfTI 的 voxel (x, y)（不含 z）。
    img_w, img_h：图片宽高（JSON 中 x, y 的参考尺寸）
    nifti_x, nifti_y：NIfTI 切片的像素尺寸 (x, y) 对应 arr.shape[2], arr.shape[1]
    crop_offset：图片左上角黑边的像素偏移量 (x_offset, y_offset)
    """
    # 计算裁剪后的图片尺寸（去除黑边）
    cropped_w = img_w - crop_offset[0] - 296  # 左黑边294，右黑边296
    cropped_h = img_h - crop_offset[1] - 116  # 上黑边113，下黑边116

    # 映射比例
    scale_x = nifti_x / cropped_w
    scale_y = nifti_y / cropped_h

    # 映射到 NIfTI voxel 坐标
    vx = (x_img - crop_offset[0]) * scale_x
    vy = (y_img - crop_offset[1]) * scale_y
    return int(round(vx)), int(round(vy))

import SimpleITK as sitk
import json
import pandas as pd
import numpy as np

def process_keypoints_json(json_path, original_nifti_path, preprocessed_nifti_path, out_csv="output.csv"):
    """
    改进版：
    1. 利用回归公式计算预处理图像坐标系的 (x_prep, z_prep)。
    2. 将其转换为物理世界坐标。
    3. 将物理坐标映射回原始 CT 图像的体素坐标 (x_orig, z_orig)。
    4. 在原始 CT 上沿 Y 轴扫描提取数据。
    """
    
    # 1. 加载两个 NIfTI 图像
    print(f"正在加载原始NIfTI文件: {original_nifti_path}")
    print(f"正在加载预处理NIfTI文件 (用于坐标参照): {preprocessed_nifti_path}")
    
    try:
        orig_image = sitk.ReadImage(original_nifti_path)
        prep_image = sitk.ReadImage(preprocessed_nifti_path)
    except Exception as e:
        print(f"错误: 无法加载NIfTI文件. {e}")
        return

    # 获取原始图像的尺寸，这是我们最终扫描的范围
    orig_size = orig_image.GetSize()
    dim_x_orig, dim_y_orig, dim_z_orig = orig_size
    print(f"原始图像尺寸: {orig_size}")

    # 2. 加载 JSON 关键点
    print(f"正在加载JSON文件: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        keypoints_2d = data.get("keypoints", [])
    except Exception as e:
        print(f"错误: 无法加载JSON文件. {e}")
        return

    # 3. 定义回归参数 (基于预处理图像坐标系)
    kx, bx = -0.322, 208.23
    kz, bz = -0.385, 199.10

    all_points_data = []
    print(f"开始处理 {len(keypoints_2d)} 个关键点 (跨坐标系转换)...")
    
    for kp in keypoints_2d:
        x_2d, y_2d = kp['x'], kp['y']

        # -------------------------------------------------------
        # Step A: 计算在 "预处理图像" 中的体素索引
        # -------------------------------------------------------
        # 注意：这里我们并不需要它一定在预处理图像边界内，因为物理空间是连续的
        voxel_x_prep = int(round(kx * x_2d + bx))
        voxel_z_prep = int(round(kz * y_2d + bz))
        # Y轴随便取一个值(例如0)，因为我们通过物理坐标转换时，主要关注 X和Z 的定位
        voxel_y_prep = 0 

        # -------------------------------------------------------
        # Step B: 坐标转换 (Prep Voxel -> Physical -> Orig Voxel)
        # -------------------------------------------------------
        try:
            # 1. 转为物理坐标 (mm)
            # TransformIndexToPhysicalPoint 接受 (x,y,z)
            # 即使索引超出 prep_image 的边界，SITK通常也能算出物理位置
            phys_point = prep_image.TransformIndexToPhysicalPoint((voxel_x_prep, voxel_y_prep, voxel_z_prep))
            
            # 2. 转为原始图像体素索引
            # TransformPhysicalPointToIndex 接受物理点，返回 (x,y,z)
            voxel_idx_orig = orig_image.TransformPhysicalPointToIndex(phys_point)
            
            # 提取原始图像中的 X 和 Z
            x_orig = voxel_idx_orig[0]
            z_orig = voxel_idx_orig[2]
            
            # 注意：y_orig 这里我们不直接使用，因为我们要扫描整个 Y 轴
            
        except Exception as e:
            print(f"坐标转换错误 (Kp: {kp['index']}): {e}")
            continue

        # -------------------------------------------------------
        # Step C: 原始图像边界检查
        # -------------------------------------------------------
        # 确保转换后的 X 和 Z 在原始图像范围内
        if not (0 <= x_orig < dim_x_orig and 0 <= z_orig < dim_z_orig):
            print(f"跳过: 关键点 {kp['index']} 映射到原始图像外 ({x_orig}, {z_orig})")
            continue

        # -------------------------------------------------------
        # Step D: 在原始 NIfTI 上沿 Y 轴扫描
        # -------------------------------------------------------
        for j in range(dim_y_orig):
            try:
                # 访问原始数据
                value = orig_image.GetPixel(x_orig, j, z_orig)
                
                all_points_data.append({
                    "kp_index": kp['index'],
                    "x": x_orig,   # 原始图像 X
                    "y": j,        # 原始图像 Y (即使是空气也记录，方便后续查看)
                    "z": z_orig,   # 原始图像 Z
                    "value": value
                })
            except Exception as e:
                pass

    # 4. 保存结果
    if not all_points_data:
        print("未生成任何数据点。")
        return
        
    df = pd.DataFrame(all_points_data)
    df.to_csv(out_csv, index=False)
    print(f"\n处理完成！共提取 {len(df)} 个原始数据点，已保存至: {out_csv}")
    print("CSV文件预览 (原始体素坐标):")
    print(df.head())

# 示例用法
# if __name__ == "__main__":
#     # --- 请在这里配置你的文件路径 ---
    
#     # 输入的2D关键点JSON文件
#     json_file = "keypoints.json"
    
#     # 对应的3D NIfTI文件（必须是世界坐标系）
#     nifti_file = "H004.nii.gz"  
    
#     # 输出的CSV文件名
#     output_file = "output_all_y_values.csv"

#     # --- 执行函数 ---
#     process_keypoints_json(
#         json_path=json_file,
#         nifti_path=nifti_file,
#         out_csv=output_file
#     )
