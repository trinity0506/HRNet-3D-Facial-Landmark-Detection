import nibabel as nib
import numpy as np
import SimpleITK as sitk

def load_and_inspect_nifti(file_path):
    """
    加载NIFTI文件并检查其基本信息。
    """
    # 使用SimpleITK加载，因为它提供丰富的元数据
    sitk_image = sitk.ReadImage(file_path)
    data_array = sitk.GetArrayFromImage(sitk_image)
    
    print("=== NIFTI文件信息 ===")
    print(f"数据维度 (Z, Y, X): {data_array.shape}")
    print(f"体素间距 (mm): {sitk_image.GetSpacing()}")
    print(f"图像原点 (mm): {sitk_image.GetOrigin()}")
    print(f"方向矩阵: {sitk_image.GetDirection()}")
    print(f"数据类型: {data_array.dtype}")
    print(f"强度值范围: [{np.min(data_array):.2f}, {np.max(data_array):.2f}]")
    
    return sitk_image, data_array

def resample_image(sitk_image, new_spacing=(1.0, 1.0, 1.0)):
    """
    将图像重采样到指定的体素间距。
    """
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    
    # 计算新尺寸
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    
    # 设置重采样滤波器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayFromImage(sitk_image))))  # 设置背景值为最小值
    resampler.SetInterpolator(sitk.sitkLinear)  # 线性插值
    
    # 执行重采样
    resampled_image = resampler.Execute(sitk_image)
    
    print(f"原始间距: {original_spacing}, 新间距: {resampled_image.GetSpacing()}")
    return resampled_image

def intensity_normalization(sitk_image, method='zscore'):
    """
    对图像强度进行归一化。
    """
    data_array = sitk.GetArrayFromImage(sitk_image)
    
    # 创建掩码：排除背景（假设背景HU值 < -500）
    background_threshold = -200
    mask = data_array > background_threshold
    
    if method == 'zscore':
        # Z-score归一化: (x - mean) / std
        mean_val = np.mean(data_array[mask])
        std_val = np.std(data_array[mask])
        normalized_array = (data_array - mean_val) / std_val
        print(f"Z-score归一化: Mean={mean_val:.2f}, Std={std_val:.2f}")
    elif method == 'minmax':
        # Min-Max缩放至 [0, 1]
        min_val = np.min(data_array[mask])
        max_val = np.max(data_array[mask])
        normalized_array = (data_array - min_val) / (max_val - min_val)
        print(f"Min-Max归一化: Min={min_val:.2f}, Max={max_val:.2f}")
    else:
        raise ValueError("方法请选择 'zscore' 或 'minmax'")
    
    # 将归一化后的数组转换回SimpleITK图像
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(sitk_image)  # 保留元数据
    return normalized_image

def smooth_image(sitk_image, sigma=2.0):
    """
    应用高斯滤波平滑图像。
    """
    smoother = sitk.DiscreteGaussianImageFilter()
    smoother.SetVariance(sigma ** 2)  # 设置方差（标准差平方）
    smoothed_image = smoother.Execute(sitk_image)
    print(f"应用高斯滤波: Sigma={sigma}")
    return smoothed_image

def skull_stripping_ct(sitk_image, bone_threshold=800, background_threshold=-200):
    """
    改进版：包含形态学处理以去除头枕/板子
    """
    # 1. 获取原始数据的Mask（基于阈值）
    # 注意：SimpleITK的二值化滤镜比numpy操作更容易进行后续的形态学处理
    binary_mask = sitk.BinaryThreshold(
        sitk_image, 
        lowerThreshold=background_threshold, 
        upperThreshold=2000 # 上限设高一点以包含所有骨骼
    )

    # 2. 形态学开运算 (Opening) - 关键步骤！
    # 作用：先腐蚀后膨胀。它可以切断头部和板子之间细微的连接。
    # 如果板子和大头粘在一起，增加 KernelRadius (如 2 或 3)
    opener = sitk.BinaryMorphologicalOpeningImageFilter()
    opener.SetKernelRadius(3) 
    opener.SetKernelType(sitk.sitkBall) # 可选：使用球形核
    binary_mask = opener.Execute(binary_mask)

    # 3. 提取最大连通域
    # 作用：只保留体积最大的一块（头部），丢弃板子和噪点
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_mask = cc_filter.Execute(binary_mask)
    
    # 按各区域大小(体积)重新编号，1号代表最大区域
    relabel_filter = sitk.RelabelComponentImageFilter()
    labeled_mask = relabel_filter.Execute(labeled_mask)
    
    # 只保留 Label 为 1 的区域
    final_mask = sitk.BinaryThreshold(labeled_mask, lowerThreshold=1, upperThreshold=1)

    # 4. 稍微膨胀回来 (可选)
    # 作用：因为第2步腐蚀了一点边缘，现在补回来，以免损伤皮肤表面
    dilater = sitk.BinaryDilateImageFilter()
    dilater.SetKernelRadius(2)
    final_mask = dilater.Execute(final_mask)
    
    # ==========================================
    # 应用掩膜
    # ==========================================
    # 将 mask 转换回 array 进行乘法，或者使用 MaskImageFilter
    mask_filter = sitk.MaskImageFilter()
    mask_filter.SetOutsideValue(-1000) # 背景设为空气
    brain_image = mask_filter.Execute(sitk_image, final_mask)
    
    print("颅骨剥离完成: 已移除背景及头枕。")
    return brain_image

def enhance_contrast(sitk_image, alpha=1.5):
    """
    适度的对比度增强，避免过度处理
    """
    data_array = sitk.GetArrayFromImage(sitk_image)
    
    # 使用更保守的对比度拉伸
    min_val = np.percentile(data_array[data_array > -500], 5)  # 使用5%分位数而非最小值
    max_val = np.percentile(data_array[data_array > -500], 95) # 使用95%分位数而非最大值
    
    enhanced_array = (data_array - min_val) / (max_val - min_val) * 255
    enhanced_array = np.clip(enhanced_array, 0, 255)
    
    enhanced_image = sitk.GetImageFromArray(enhanced_array.astype(np.uint8))
    enhanced_image.CopyInformation(sitk_image)
    
    print(f"对比度增强: 拉伸范围[{min_val:.2f}, {max_val:.2f}]")
    return enhanced_image

def extract_face_roi(sitk_image):
    """
    智能提取ROI：自动判断图像类型并选择阈值，同时保留正确的空间坐标。
    """
    # 1. 自动检测图像像素类型，设定正确的阈值
    pixel_id = sitk_image.GetPixelID()
    
    # 如果是 uint8 (经过了对比度增强/归一化，范围0-255)
    if pixel_id == sitk.sitkUInt8:
        # 背景通常是0，只要大于1即视为有人体组织
        lower_th = 0
        upper_th = 255
        print("检测到 uint8 图像，使用阈值 [1, 255] 提取 ROI")
        
    # 如果是 int16/float (原始CT数据，范围HU值)
    else:
        # 使用软组织阈值 (HU > -300 包含皮肤和肌肉)
        lower_th = -500
        upper_th = 3000
        print(f"检测到原始 CT 数据，使用 HU 阈值 [{lower_th}, {upper_th}] 提取 ROI")

    # 2. 生成二值掩码
    mask = sitk.BinaryThreshold(sitk_image, lowerThreshold=lower_th, upperThreshold=upper_th)
    
    # 3. 计算掩码的边界框 (LabelShapeStatisticsImageFilter)
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(mask)
    
    if not shape_filter.GetLabels():
        print("警告: 未找到有效区域，返回原图像。")
        return sitk_image
        
    # 获取 Label 1 的 Bounding Box
    # bbox: (x_min, y_min, z_min, x_size, y_size, z_size)
    bbox = shape_filter.GetBoundingBox(1)
    
    # 4. 使用 RegionOfInterestfilter 提取
    # 注意：这能完美保留之前提到的 Origin 和 Direction 空间信息
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetRegionOfInterest(bbox)
    roi_image = roi_filter.Execute(sitk_image)
    
    print(f"ROI提取完成: {sitk_image.GetSize()} -> {roi_image.GetSize()}")
    
    return roi_image

def facial_ct_preprocessing_pipeline(input_file_path, output_file_path):
    """
    面部CT预处理完整管道。
    """
    print("步骤1/7: 加载与验证数据...")
    sitk_image, data_array = load_and_inspect_nifti(input_file_path)
    
    print("步骤2/7: 重采样至1mm各向同性...")
    resampled_img = resample_image(sitk_image)
    print("步骤4/7: 图像平滑...")
    smoothed_img = smooth_image(resampled_img, sigma=1)


    stripped_img = skull_stripping_ct(
        smoothed_img, 
        bone_threshold=800,      
        background_threshold=-500
    )
    print("步骤3/7: 强度归一化...")
    normalized_img = intensity_normalization(stripped_img, method='minmax')


    
    print("步骤5/7: 对比度增强...")
    enhanced_img = enhance_contrast(normalized_img)
    
    print("步骤6/7: (可选) 面部ROI提取...")
    final_img = extract_face_roi(enhanced_img) 

    
    # 保存结果
    sitk.WriteImage(enhanced_img, output_file_path)
    print(f"预处理完成！结果已保存至: {output_file_path}")
    return enhanced_img
# 示例
if __name__ == '__main__':
    input_path = "data/ct/images/NIFTI/H006.nii.gz"
    output_path = "output/preprocessed.nii.gz"
    preprocessed_image = facial_ct_preprocessing_pipeline(input_path, output_path)