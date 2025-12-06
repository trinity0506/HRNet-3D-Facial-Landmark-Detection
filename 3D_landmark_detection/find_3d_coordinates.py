import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import csv
import SimpleITK as sitk
import json

def map_2d_point_to_nifti(nifti_path, point_2d, canvas_size=(1184,649)):
    """
    针对四周有黑边的情况，将2D点还原为3D NIfTI坐标
    """
    img = sitk.ReadImage(nifti_path)
    
    # 1. 读取 NIfTI 的物理“盒子”尺寸
    # 这是关键！我们是用这个“隐形的盒子”去匹配画布，而不是用人头去匹配
    size = img.GetSize()      # (x, y, z) 像素数量
    spacing = img.GetSpacing() # (dx, dy, dz) 像素间距
    origin = img.GetOrigin()
    
    # 计算 NIfTI 在真实世界里的物理宽高 (mm)
    # 注意：SimpleITK 的 GetSize 顺序通常是 (X, Y, Z)
    # 但在正视图(Front View)中，我们看到的是 X轴(宽) 和 Z轴(高)
    vol_width_mm = size[0] * spacing[0]
    vol_height_mm = size[2] * spacing[2] # 假设Z轴是高度
    
    # NIfTI 物理中心点
    center_x = origin[0] + vol_width_mm / 2.0
    center_z = origin[2] + vol_height_mm / 2.0
    center_y = origin[1] + (size[1] * spacing[1]) / 2.0 # 深度中心

    # 2. 模拟渲染器的“缩放 (Scale)”逻辑
    # 渲染器会将最大的那个物理边，缩放到适应屏幕
    canvas_w, canvas_h = canvas_size
    
    # 分别计算宽和高撑满屏幕所需的缩放比
    scale_x = canvas_w / vol_width_mm
    scale_y = canvas_h / vol_height_mm
    
    # 既然四周都有黑边，说明渲染器取了较小的那个缩放比来保证完全显示
    # 也就是 scale = min(...)
    scale = min(scale_x, scale_y)

    # *重要修正*：如果渲染器有默认的 margin (例如10%留白)，
    # 这里可能需要乘以一个系数，如 0.9。
    # 但通常标准 render 是紧贴至少一边的。先按紧贴计算。
    
    # 3. 计算有效显示内容的大小 (Display Box)
    display_w = vol_width_mm * scale
    display_h = vol_height_mm * scale
    
    # 4. 计算画布上的偏移量 (即黑边的宽度/高度，单位：像素)
    # 这步会自动算出你图片里看到的上下左右黑边
    offset_x_px = (canvas_w - display_w) / 2.0
    offset_y_px = (canvas_h - display_h) / 2.0
    
    # 5. 坐标逆变换
    u, v = point_2d # 输入的检测坐标
    
    # 5.1 减去黑边，得到相对于“有效画面”左上角的像素坐标
    rel_u = u - offset_x_px
    rel_v = v - offset_y_px
    
    # 5.2 转回物理尺寸 (mm)
    # 从中心点算起
    delta_pixel_x = rel_u - (display_w / 2.0)
    delta_pixel_z = rel_v - (display_h / 2.0) # 屏幕Y向下增加
    
    # 转为 mm
    delta_mm_x = delta_pixel_x / scale
    delta_mm_z = -delta_pixel_z / scale # 注意 Y轴反转 (屏幕向下 vs 世界向上)
    
    # 5.3 加上原点
    # X 轴方向注意：
    # 屏幕右侧通常对应世界坐标 X 的负方向（Radiological View: 病人左侧在屏幕右侧）
    # 或者正方向（Neurological View）。NIfTI 默认通常是 RAS。
    # 如果发现左右反了，把下面的 + 号改成 - 号
    final_x = center_x - delta_mm_x  # 假设是 Radiological View (屏幕右=病人左)
    final_z = center_z + delta_mm_z
    
    
    return (final_x, final_z)

def voxel_to_ras(voxel_coord, affine, origin, spacing):
    """
    将 voxel 坐标转换为 RAS 坐标。
    voxel_coord: (x, y, z) 的 voxel 坐标。
    affine: NIfTI 的方向矩阵。
    origin: NIfTI 的原点。
    spacing: NIfTI 的体素间距。
    """
    ras_coord = origin + np.dot(affine, voxel_coord * spacing)
    return ras_coord

def hu_after_ok(idx, hu_s, window=5, low=-300, high=300):
    """
    检查 idx 之后若干点是否稳定处于组织范围。
    """
    if idx is None:
        return False
    start = max(0, idx)
    end = min(len(hu_s), idx + window)
    m = hu_s[start:end].mean()
    return low < m < high

def convert_to_hu(nifti_img, values):

    hu_values = values
    return hu_values

import pandas as pd
import numpy as np
import os
def find_3d_points(json_path, original_nifti_path, preprocessed_nifti_path, out_csv):
    """
    一步到位函数：
    1. 计算回归坐标 (x, z)。
    2. 在原始 CT 提取 Y 轴数据。
    3. 利用梯度+阈值确定皮肤表面 Y。
    4. 兜底策略：如果检测失败，强制使用第一个成功点的 Y 值。
    """
    
import SimpleITK as sitk
import json
import pandas as pd
import numpy as np
import os

def find_3d_points_direct_with_fallback(json_path, original_nifti_path, preprocessed_nifti_path, out_csv="output/keypoints.csv"):
    """
    [回归逻辑: First Hit + Local Search]
    1. 从外部向内部扫描 (Y轴大->小)。
    2. 找到第一个数值超过 -200 的点 (粗定位)。
    3. 在该点前后 10 个像素范围内，找梯度最大的位置 (精定位)。
    4. 包含对齐回填 (Fallback) 机制。
    """
    print(f"Loading data...")
    try:
        orig_image = sitk.ReadImage(original_nifti_path)
        prep_image = sitk.ReadImage(preprocessed_nifti_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    dim_x, dim_y, dim_z = orig_image.GetSize()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keypoints_2d = data.get("keypoints", [])

    kx, bx = -0.322, 208.23
    kz, bz = -0.385, 199.10
    
    # 阈值：第一次碰到这个值，就认为大概撞到皮肤了
    TISSUE_THRESHOLD = -200 
    # 局部搜索范围：在粗定位点的上下多少层内寻找最陡峭的边缘
    SEARCH_RADIUS = 10     
    MIN_GRADIENT = 30 # 稍微降低梯度要求，确保容易检测

    final_results = []
    reference_y = None 

    print(f"Processing {len(keypoints_2d)} keypoints (Method: First Hit > {TISSUE_THRESHOLD} -> Max Grad in ±{SEARCH_RADIUS})...")

    for i, kp in enumerate(keypoints_2d):
        kp_idx = kp['index']
        
        # --- 1. 坐标映射 ---
        voxel_x_prep = int(round(kx * kp['x'] + bx))
        voxel_z_prep = int(round(kz * kp['y'] + bz))
        
        try:
            phys_pt = prep_image.TransformIndexToPhysicalPoint((voxel_x_prep, 0, voxel_z_prep))
            idx_orig = orig_image.TransformPhysicalPointToIndex(phys_pt)
            x_orig, z_orig = int(idx_orig[0]), int(idx_orig[2])
            
            x_orig = max(0, min(x_orig, dim_x - 1))
            z_orig = max(0, min(z_orig, dim_z - 1))
        except:
            # 越界跳过
            continue

        # --- 2. 提取数据 (从外向内扫描: Y轴从大到小) ---
        y_values = []
        for j in range(dim_y - 1, -1, -1):  
            y_values.append(orig_image.GetPixel(x_orig, j, z_orig))
        y_values = np.array(y_values) 
        
        # 计算梯度 (空气->皮肤是数值上升，所以正梯度是边缘)
        # 注意：这里不需要加负号，因为数值是从-1000变到-100，梯度本身就是正的
        gradients = np.gradient(y_values) 

        # --- 3. 核心逻辑: 首个阈值点 + 局部搜索 ---
        detected_y = None
        
        # 3.1 找到第一次 "撞击" 组织的索引
        # (因为是从外向内扫，第一个 > -200 的点就是皮肤最外层附近)
        hit_indices = np.where(y_values > TISSUE_THRESHOLD)[0]

        if len(hit_indices) > 0:
            first_hit_idx = hit_indices[0] # 获取第一个碰到的点
            
            # 3.2 定义局部搜索窗口 (防止搜到脑壳里面去)
            start_search = max(0, first_hit_idx - SEARCH_RADIUS)
            end_search = min(len(y_values), first_hit_idx + SEARCH_RADIUS + 1)
            
            # 3.3 在窗口内找梯度最大的点 (最陡峭的空气-皮肤交界面)
            local_gradients = gradients[start_search:end_search]
            if len(local_gradients) > 0:
                # 注意：argmax返回的是相对窗口的索引
                local_best_idx = np.argmax(local_gradients)
                best_idx_in_array = start_search + local_best_idx
                
                # 检查梯度强度
                if gradients[best_idx_in_array] >= MIN_GRADIENT:
                    # 转换回原始CT坐标 (因为y_values是倒序提取的)
                    # y_values[k] 对应 dim_y - 1 - k
                    detected_y = int(dim_y - 1 - best_idx_in_array)
                    detected_val = y_values[best_idx_in_array]

        # --- 4. 决策与回填 ---
        final_y = 0
        final_val = -1000
        method = "Unknown"

        if detected_y is not None:
            # 成功找到
            final_y = detected_y
            final_val = detected_val
            method = "Detected (FirstHit)"
            
            # 更新参考平面
            if reference_y is None:
                reference_y = final_y
                print(f"  [Kp:{kp_idx}] Locked Reference Plane Y = {reference_y}")
            # 如果偏离参考平面太远(如超过30像素)，认为是误检，强行拉回
            elif abs(final_y - reference_y) > 30:
                final_y = reference_y
                final_val = orig_image.GetPixel(x_orig, final_y, z_orig)
                method = "Corrected (Outlier)"
                
        else:
            # 没找到 (比如射线射偏了，完全在空气中)
            if reference_y is not None:
                final_y = reference_y
                final_val = orig_image.GetPixel(x_orig, final_y, z_orig)
                method = "Fallback (Ref)"
                print(f"  [Kp:{kp_idx}] Missed -> Using Reference Y")
            else:
                final_y = int(dim_y // 2) # 无奈之举
                method = "Failed"

        final_results.append({
            "kp_index": kp_idx,
            "x": x_orig,
            "y": final_y,
            "z": z_orig,
            "value": final_val,
            "method": method
        })

    # 保存
    df = pd.DataFrame(final_results)
    if 'kp_index' in df.columns:
        df = df.sort_values(by='kp_index')
    df.to_csv(out_csv, index=False)
    print(f"Done! Saved to {out_csv}")

def find_3d_points_simple(json_path, original_nifti_path, preprocessed_nifti_path, out_csv="output/keypoints.csv"):
    """
    [回归逻辑: First Hit + Local Search]
    1. 从外部向内部扫描 (Y轴大->小)。
    2. 找到第一个数值超过 -200 的点 (粗定位)。
    3. 在该点前后 10 个像素范围内，找梯度最大的位置 (精定位)。
    4. 包含对齐回填 (Fallback) 机制。
    """
    print(f"Loading data...")
    try:
        orig_image = sitk.ReadImage(original_nifti_path)
        prep_image = sitk.ReadImage(preprocessed_nifti_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    dim_x, dim_y, dim_z = orig_image.GetSize()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keypoints_2d = data.get("keypoints", [])

    kx, bx = -0.322, 208.23
    kz, bz = -0.385, 199.10
    
    # 阈值：第一次碰到这个值，就认为大概撞到皮肤了
    TISSUE_THRESHOLD = 100
    # 局部搜索范围：在粗定位点的上下多少层内寻找最陡峭的边缘
    SEARCH_RADIUS = 10     
    MIN_GRADIENT = 30 # 稍微降低梯度要求，确保容易检测

    final_results = []
    reference_y = None 

    print(f"Processing {len(keypoints_2d)} keypoints (Method: First Hit > {TISSUE_THRESHOLD} -> Max Grad in ±{SEARCH_RADIUS})...")

    for i, kp in enumerate(keypoints_2d):
        kp_idx = kp['index']
        
        # --- 1. 坐标映射 ---
        voxel_x_prep = int(round(kx * kp['x'] + bx))
        voxel_z_prep = int(round(kz * kp['y'] + bz))
        
        try:
            phys_pt = prep_image.TransformIndexToPhysicalPoint((voxel_x_prep, 0, voxel_z_prep))
            idx_orig = orig_image.TransformPhysicalPointToIndex(phys_pt)
            x_orig, z_orig = int(idx_orig[0]), int(idx_orig[2])
            
            x_orig = max(0, min(x_orig, dim_x - 1))
            z_orig = max(0, min(z_orig, dim_z - 1))
        except:
            # 越界跳过
            continue

        # --- 2. 提取数据 (从外向内扫描: Y轴从大到小) ---
        y_values = []
        for j in range(dim_y - 1, -1, -1):  
            y_values.append(orig_image.GetPixel(x_orig, j, z_orig))
        y_values = np.array(y_values) 
        
        # 计算梯度 (空气->皮肤是数值上升，所以正梯度是边缘)
        # 注意：这里不需要加负号，因为数值是从-1000变到-100，梯度本身就是正的
        gradients = np.gradient(y_values) 

        # --- 3. 核心逻辑: 首个阈值点 + 局部搜索 ---
        detected_y = None
            
            # 遍历Y轴数据（从外向内扫描：大→小）
        for i in range(len(y_values)):
            if y_values[i] > TISSUE_THRESHOLD:  # 找到第一个 > -200 的点
                # 将数组索引转换回原始CT的Y坐标（因y_values是倒序提取）
                detected_y = int(dim_y - 1 - i)  
                detected_val = y_values[i]
                break  # 找到后立即退出循环，不再继续搜索

            # --- 4. 决策逻辑（保持不变） ---
        if detected_y is not None:
            final_y = detected_y
            method = "First Hit (> -200)"
            final_val = detected_val
            method = "Detected (FirstHit)"
            
            # 更新参考平面
            if reference_y is None:
                reference_y = final_y
                print(f"  [Kp:{kp_idx}] Locked Reference Plane Y = {reference_y}")
            # 如果偏离参考平面太远(如超过30像素)，认为是误检，强行拉回
            elif abs(final_y - reference_y) > 30:
                final_y = reference_y
                final_val = orig_image.GetPixel(x_orig, final_y, z_orig)
                method = "Corrected (Outlier)"
                
        else:
            # 没找到 (比如射线射偏了，完全在空气中)
            if reference_y is not None:
                final_y = reference_y
                final_val = orig_image.GetPixel(x_orig, final_y, z_orig)
                method = "Fallback (Ref)"
                print(f"  [Kp:{kp_idx}] Missed -> Using Reference Y")
            else:
                final_y = int(dim_y // 2) # 无奈之举
                method = "Failed"

        final_results.append({
            "kp_index": kp_idx,
            "x": x_orig,
            "y": final_y,
            "z": z_orig,
            "value": final_val,
            "method": method
        })

    # 保存
    df = pd.DataFrame(final_results)
    if 'kp_index' in df.columns:
        df = df.sort_values(by='kp_index')
    df.to_csv(out_csv, index=False)
    print(f"Done! Saved to {out_csv}")

def find_face_points_from_csv(input_csv, output_csv, tissue_thresh=-300, min_gradient=100, search_radius=5):
    """
    修改版：基于体素坐标的深度搜索。
    读取包含沿Y轴所有点的CSV，为每个 2D Keypoint 找到最可能的面部表面点 (Y坐标)。

    Args:
        input_csv (str): 输入的CSV文件路径 (包含 kp_index, x, y, z, value)。
        output_csv (str): 输出的CSV文件路径。
        tissue_thresh (int): 组织HU值阈值 (建议 -300 到 -100 之间)。
        min_gradient (int): 最小梯度阈值。
        search_radius (int): 局部搜索半径(单位:体素)。
    """
    # 1. 加载包含所有扫描线数据的CSV
    print(f"正在加载数据: {input_csv}")
    if not os.path.exists(input_csv):
        print(f"错误: 文件未找到 '{input_csv}'")
        return

    df = pd.read_csv(input_csv)

    # 存储最终筛选出的表面点
    surface_points = []
    
    # 2. 分组处理
    # 优先使用 'kp_index' 分组 (确保每个关键点都有一个输出)
    # 如果没有 'kp_index' (旧版数据)，则回退到按 'x', 'z' 分组
    if 'kp_index' in df.columns:
        print("检测到 kp_index，按关键点索引分组处理...")
        grouped = df.groupby('kp_index')
    else:
        print("未检测到 kp_index，按 (x, z) 坐标分组处理...")
        grouped = df.groupby(['x', 'z'])
    
    # 遍历每个关键点的扫描线数据
    for name, group_df in grouped:
        
        # -------------------------------------------------------------
        # 关键设置：扫描方向
        # 假设我们想从“空气”侧扫描到“皮肤”侧。
        # 如果您的 NIfTI Y轴是 "后->前" (Posterior-Anterior)，脸在 Y 值较大的地方，
        # 那么需要从大到小扫描，即 ascending=False。
        # 如果 Y=0 是脸前方的空气，则 ascending=True。
        # 这里默认保留您原来的 ascending=True，如果发现找到了后脑勺，请改为 False。
        # -------------------------------------------------------------
        group_df = group_df.sort_values(by='y', ascending=False).reset_index(drop=True)

        hu_values = group_df['value'].to_numpy()
        
        # 数据点太少无法计算梯度
        if len(hu_values) < 2:
            continue

        # 计算HU值沿Y轴的梯度 (Gradient)
        gradient = np.gradient(hu_values)

        # ---------------------------------------------------------
        # Step 1: 粗定位 - 找到第一个超过组织阈值的点 (Anchor)
        # ---------------------------------------------------------
        region_anchor_idx = None
        for i, hu in enumerate(hu_values):
            if hu >= tissue_thresh:
                region_anchor_idx = i
                break
        
        # ---------------------------------------------------------
        # Step 2: 精定位 - 在Anchor点附近寻找梯度最大值
        # ---------------------------------------------------------
        chosen_idx = None
        
        if region_anchor_idx is not None:
            # 定义局部搜索窗口
            start_search = max(0, region_anchor_idx - search_radius)
            end_search = min(len(gradient), region_anchor_idx + search_radius + 1)

            local_grad_segment = gradient[start_search:end_search]

            if len(local_grad_segment) > 0:
                # 找局部梯度最大值
                local_peak_idx = np.argmax(local_grad_segment)
                global_peak_idx = start_search + local_peak_idx
                
                # 决策：梯度是否足够大？
                if gradient[global_peak_idx] >= min_gradient:
                    chosen_idx = global_peak_idx
                else:
                    chosen_idx = region_anchor_idx
            else:
                chosen_idx = region_anchor_idx

        # 保存结果
        if chosen_idx is not None:
            chosen_point = group_df.iloc[chosen_idx]
            
            # 构建结果字典
            result_dict = {
                'x': int(chosen_point['x']), # 强制转为整数索引
                'y': int(chosen_point['y']),
                'z': int(chosen_point['z']),
                'HU': chosen_point['value']
            }
            
            # 如果有 kp_index，保留它以便后续对应
            if 'kp_index' in chosen_point:
                result_dict['kp_index'] = int(chosen_point['kp_index'])
                
            surface_points.append(result_dict)

    # 3. 保存
    if not surface_points:
        print("处理完成，但未能找到任何有效的表面点。")
        return

    output_df = pd.DataFrame(surface_points)
    
    # 重新排列列顺序，让 kp_index 排在前面更好看
    cols = ['kp_index', 'x', 'y', 'z', 'HU'] if 'kp_index' in output_df.columns else ['x', 'y', 'z', 'HU']
    output_df = output_df[cols]

    output_df.to_csv(output_csv, index=False)
    
    print(f"\n处理完成！共找到 {len(output_df)} 个面部表面点 (体素坐标)，已保存至: {output_csv}")
    print(output_df.head())

def find_3d_points_from_preprocessed(
    json_path, 
    original_nifti_path, 
    preprocessed_nifti_path, 
    out_csv="output/keypoints_fixed.csv"
):
    """
    [更新思路]
    1. 在预处理CT中，沿Y轴从大到小扫描。
    2. 找到第一个像素值 > SURFACE_THRESHOLD 的点（设为皮肤/骨骼表面）。
    3. 修复 TypeError，确保所有传给 SimpleITK 的坐标均为 Python 原生 int。
    4. 将找到的空间点映射回原始 CT 坐标。
    5. 新增：输出4个点在预处理CT中沿Y轴所有值至3D_y_all.csv便于调试。
    """
    print(f"=== 新思路：预处理CT定位 (阈值 > 1) ===")
    try:
        orig_image = sitk.ReadImage(original_nifti_path)
        prep_image = sitk.ReadImage(preprocessed_nifti_path)
    except Exception as e:
        print(f"图像加载失败: {e}")
        return

    prep_size = prep_image.GetSize()
    orig_size = orig_image.GetSize()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keypoints_2d = data.get("keypoints", [])
    print(f"共读取到 {len(keypoints_2d)} 个关键点")

    kx, bx = -0.322, 208.23
    kz, bz = -0.385, 199.10
    SURFACE_THRESHOLD = 1  

    final_results = []
    reference_y_prep = None

    # --- 准备保存4个点沿Y轴所有像素值以便输出 ---
    debug_lines = []
    debug_points_to_log = 4
    points_logged = 0

    print(f"处理关键点...")

    for i, kp in enumerate(keypoints_2d):
        kp_idx = kp['index']
        try:
            raw_x = kx * kp['x'] + bx
            raw_z = kz * kp['y'] + bz
            
            prep_x = int(round(raw_x))
            prep_z = int(round(raw_z))
            prep_x = int(max(0, min(prep_x, prep_size[0] - 1)))
            prep_z = int(max(0, min(prep_z, prep_size[2] - 1)))
        except Exception as e:
            print(f"[Kp:{kp_idx}] 坐标计算错误: {e}")
            continue

        hit_y_prep = None
        hit_val = -1000
        
        # 采集该点沿Y轴的所有体素值，用于调试
        y_values = []
        for j in range(prep_size[1] - 1, -1, -1):
            val = prep_image.GetPixel(int(prep_x), int(j), int(prep_z))
            y_values.append({'kp_index': kp_idx, 'prep_x': prep_x, 'prep_z': prep_z, 'y': j, 'value': val})
            if hit_y_prep is None and val > SURFACE_THRESHOLD:
                hit_y_prep = int(j)
                hit_val = val

        if hit_y_prep is None:
            status = "Fallback(Ref)"
            if reference_y_prep is not None:
                hit_y_prep = reference_y_prep
                print(f"[Kp:{kp_idx}] 未达阈值，回退至参考Y: {hit_y_prep}")
            else:
                hit_y_prep = int(prep_size[1] // 2) 
        else:
            status = "Detected"
            if reference_y_prep is None:
                reference_y_prep = hit_y_prep
            
            if abs(hit_y_prep - reference_y_prep) > 30:
                pass
        
        try:
            idx_tuple = (int(prep_x), int(hit_y_prep), int(prep_z))
            phys_pt = prep_image.TransformIndexToPhysicalPoint(idx_tuple)
            orig_idx_tuple = orig_image.TransformPhysicalPointToIndex(phys_pt)
            
            orig_x = int(orig_idx_tuple[0])
            orig_y = int(orig_idx_tuple[1])
            orig_z = int(orig_idx_tuple[2])
            
            orig_x = int(np.clip(orig_x, 0, orig_size[0] - 1))
            orig_y = int(np.clip(orig_y, 0, orig_size[1] - 1))
            orig_z = int(np.clip(orig_z, 0, orig_size[2] - 1))
        except Exception as e:
            print(f"[Kp:{kp_idx}] 坐标映射异常: {e}")
            orig_x, orig_y, orig_z = -1, -1, -1
            status = "Error"

        final_results.append({
            "kp_index": kp_idx,
            "orig_x": orig_x,
            "orig_y": orig_y,
            "orig_z": orig_z,
            "prep_x": prep_x,
            "prep_y": hit_y_prep,
            "prep_z": prep_z,
            "hit_val": hit_val,
            "status": status
        })

        # 只记录前4个点的y轴value以便debug输出文件
        if points_logged < debug_points_to_log:
            debug_lines.extend(y_values)
            points_logged += 1

    # 保存关键点的最终位置结果csv
    df_final = pd.DataFrame(final_results).sort_values(by="kp_index")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    print(f"关键点3D坐标已保存至: {out_csv}")

    # 保存4个关键点沿Y轴所有像素值的调试文件
    debug_df = pd.DataFrame(debug_lines)
    debug_out_path = os.path.join(os.path.dirname(out_csv), "3D_y_all.csv")
    debug_df.to_csv(debug_out_path, index=False)
    print(f"前{debug_points_to_log}个关键点沿Y轴值已保存至: {debug_out_path}")

    # 保存
    df = pd.DataFrame(final_results).sort_values(by="kp_index")
    df.to_csv(out_csv, index=False)
    print(f"Done! 结果已保存至 {out_csv}")
