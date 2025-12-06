import os
from detect_keypoints import parse_args, save_last_cfg, load_last_cfg, auto_find_cfg, load_model, preprocess_image, get_max_preds
from map_keypoints_to_voxel import process_keypoints_json
from find_3d_coordinates import find_face_points_from_csv, voxel_to_ras, hu_after_ok, find_3d_points,find_3d_points_direct_with_fallback, find_3d_points_simple,find_3d_points_from_preprocessed
from generate_json import convert_xyz_to_ras_json, extract_nifti_parameters
from lib.config import config, update_config
import torch
import json
import cv2
def detect_keypoints(image_file, json_out,detect_image):
    """
    捕捉面部关键点并保存为 JSON 文件。
    """
    # 配置文件路径
    cfg_path = os.path.join("experiments", "wflw", "face_alignment_wflw_hrnet_w18.yaml")

    # 封装配置对象
    class Args:
        cfg = cfg_path

    args = Args()

    # 更新配置
    update_config(config, args)
    save_last_cfg(cfg_path)

    # 加载模型
    model_file = config.INFER.MODEL_FILE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, model_file, device)

    # 预处理图片
    input_size = config.MODEL.IMAGE_SIZE
    input_tensor, orig_w, orig_h, (inp_w, inp_h) = preprocess_image(image_file, input_size)

    # 推理
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        heatmaps = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        heatmaps_np = heatmaps.cpu().numpy()

    # 提取关键点
    preds, maxvals = get_max_preds(heatmaps_np)
    scale_x = float(orig_w) / heatmaps_np.shape[-1]
    scale_y = float(orig_h) / heatmaps_np.shape[-2]
    preds[:, :, 0] *= scale_x
    preds[:, :, 1] *= scale_y

    # 筛选指定的关键点
    kp_indices = config.INFER.KP_INDICES  # 从配置文件中读取 KP_INDICES
    kp_indices = [int(i) for i in kp_indices if 0 <= int(i) < preds.shape[1]]
    if len(kp_indices) == 0:
        raise ValueError("No valid kp indices after filtering; check KP_INDICES in cfg or --kp-indices arg")
    print("Using kp_indices =", kp_indices)

    b = 0
    selected = []
    for idx in kp_indices:
        x = float(preds[b, idx, 0])
        y = float(preds[b, idx, 1])
        conf = float(maxvals[b, idx, 0]) if maxvals is not None else None
        # 1. 定义 ROI 参数 (根据“下方中间”的描述设置)
        # x轴: 只要中间 50% 的区域 (假设头在正中间)
        # y轴: 只要下半部分 (假设眼睛在图片高度的 40% 到 90% 之间)
        
        x_start_ratio = 0  # 左边去掉 10%
        x_end_ratio   = 1  # 右边去掉 10%
        y_start_ratio = 0.38 # 去掉顶部 40% (背景/额头以上)
        y_end_ratio   = 0.85 # 去掉底部 10% (最底边缘)

        roi_x_min = orig_w * x_start_ratio
        roi_x_max = orig_w * x_end_ratio
        roi_y_min = orig_h * y_start_ratio
        roi_y_max = orig_h * y_end_ratio

        # 2. 执行过滤
        # 如果点不在 ROI 矩形框内，直接跳过，不加入 selected 列表
        if not (roi_x_min < x < roi_x_max and roi_y_min < y < roi_y_max):
            # 可选：打印日志方便调试
            # print(f"Debug:点 {idx} 落点 ({x:.1f}, {y:.1f}) 在 ROI 之外被过滤")
            continue
        if conf is not None and getattr(args, 'conf_thresh', None) is not None:
            try:
                if conf < float(args.conf_thresh):
                    continue
            except Exception:
                pass
        selected.append({"index": int(idx), "x": x, "y": y, "conf": (float(conf) if conf is not None else None)})
    img_cv = cv2.imread(image_file)
    if img_cv is None:
        raise RuntimeError(f"Cannot read image: {image_file}")
    H_img, W_img = img_cv.shape[:2]
    circle_color = (0, 255, 0)
    text_color = (0, 255, 0)
    radius = max(1, int(round(min(H_img, W_img) / 200)))
    font_scale = 0.4 if max(H_img, W_img) < 1000 else 0.6
    thickness = 1
    img_cv = cv2.imread(image_file)
    for kp in selected:
        x = int(round(kp['x'])); y = int(round(kp['y']))
        cv2.circle(img_cv, (x, y), 1, circle_color, -1)
        cv2.putText(img_cv, str(kp['index']), (x + radius + 2, y - radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

    out_dir = os.path.dirname(detect_image)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(detect_image, img_cv)
    print("Saved visualization to:", detect_image)


        # 保存 JSON
    if json_out:
        json_dir = os.path.dirname(json_out)
        if json_dir and not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump({"image": image_file, "keypoints": selected}, f, ensure_ascii=False, indent=2)
        print(f"Saved keypoints JSON to {json_out}")

def run_pipeline(image_file, nii_file, output_dir,ori_nii_file):
    """
    集成后四步流程，完成从面部图片到 JSON 文件的生成。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 捕捉照片中的面部关键点
    print("Step 1: 捕捉面部关键点...")
    keypoints_json = os.path.join(output_dir, "keypoints.json")
    detect_image = os.path.join(output_dir, "keypoints_vis.png")
    detect_keypoints(image_file, keypoints_json,detect_image)

    # Step 2: 找到 2D 关键点对应的3D坐标
    # print("Step 2: 映射关键点到 voxel...")
    # all_csv = os.path.join(output_dir, "3D_y_all.csv")
    # process_keypoints_json(json_path=keypoints_json, preprocessed_nifti_path=nii_file, out_csv=all_csv,original_nifti_path=ori_nii_file, search_radius=3)

    # Step 3: 找到符合的 3D 坐标
    print("Step 2: 找到符合的 3D 坐标...")
    face_csv = os.path.join(output_dir, "keypoints.csv")
    # find_face_points_from_csv(input_csv=all_csv, output_csv=face_csv, 
    #                           tissue_thresh=3, min_gradient=100, search_radius=5)
    find_3d_points_from_preprocessed(json_path=keypoints_json, original_nifti_path=ori_nii_file, preprocessed_nifti_path=nii_file, out_csv=face_csv)
    # Step 4: 生成 JSON 文件
    # print("Step 4: 生成 JSON 文件...")
    # final_json = os.path.join(output_dir, "landmarks.json")
    # affine, origin, spacing = extract_nifti_parameters(nii_file)
    # affine[0, :] *= -1
    # affine[1, :] *= -1
    # affine[2, :] *= -1
    # affine[:, [2, 1]] = affine[:, [1, 2]]
    # origin[2] += 250 
    # origin[0] += 250 
    # origin[1] += -20 
    # spacing[2] *= 0.9
    # convert_xyz_to_ras_json(face_csv, final_json, affine, origin, spacing)


# 示例用法
if __name__ == "__main__":
    image_file = "out.png"  # 输入的面部图片路径
    nii_file = "output/preprocessed.nii.gz"  # 输入的 NIfTI 文件路径
    output_dir = "./output"  # 输出文件存储的目录
    ori_nii_file="data/ct/images/NIFTI/H006.nii.gz"
    run_pipeline(image_file, nii_file, output_dir,ori_nii_file)