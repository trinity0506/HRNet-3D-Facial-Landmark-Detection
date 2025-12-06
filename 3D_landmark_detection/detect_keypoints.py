

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pprint
import json
import numpy as np
from PIL import Image
import cv2
import torch
import numbers
import glob

from lib.config import config, update_config
import lib.models as models
from lib.models.hrnet import HighResolutionNet

LAST_CFG_PATH = os.path.join(os.path.dirname(__file__), '.last_infer_cfg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=False, default=None, type=str, help='config file (optional). If omitted, use last used or auto-find.')
    parser.add_argument('--model-file', required=False, default=None, type=str, help='model file path (optional, can be set in cfg.INFER.MODEL_FILE)')
    parser.add_argument('--image-file', required=False, default=None, type=str, help='input image path (optional, can be set in cfg.INFER.IMAGE_FILE)')
    parser.add_argument('--out', required=False, default=None, type=str, help='output visualization path (optional, can be set in cfg.INFER.OUT)')
    parser.add_argument('--json-out', required=False, default=None, type=str, help='output json path (optional, can be set in cfg.INFER.JSON_OUT)')
    parser.add_argument('--gpu', required=False, default=None, type=str, help='gpu id or -1 for cpu (optional, can be set in cfg.INFER.GPU)')
    parser.add_argument('--conf-thresh', required=False, default=None, type=float, help='confidence threshold to include kp (optional, can be set in cfg.INFER.CONF_THRESH)')
    parser.add_argument('--one-based', action='store_true', help='display indices as 1-based instead of 0-based')
    parser.add_argument('--kp-indices', default=None, type=str,
                    help='comma-separated kp indices to visualize, e.g. "36,39,42,45,30,48". Overrides cfg.INFER.KP_INDICES.')
    return parser.parse_args()

def save_last_cfg(cfg_path):
    try:
        with open(LAST_CFG_PATH, 'w', encoding='utf-8') as f:
            f.write(cfg_path)
    except Exception:
        pass

def load_last_cfg():
    if os.path.exists(LAST_CFG_PATH):
        try:
            with open(LAST_CFG_PATH, 'r', encoding='utf-8') as f:
                p = f.read().strip()
                if p:
                    return p
        except Exception:
            return None
    return None

def auto_find_cfg():
    # 尝试在 experiments/ 下查找第一个 yaml 文件
    exp_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments')
    exp_dir = os.path.abspath(exp_dir)
    if not os.path.exists(exp_dir):
        exp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    patterns = ['**/*.yaml', '**/*.yml']
    for pat in patterns:
        matches = glob.glob(os.path.join(exp_dir, pat), recursive=True)
        if matches:
            # 返回第一个
            return matches[0]
    return None

def load_model(cfg, model_file, device):
    model = models.get_face_alignment_net(cfg)
    model.to(device)

    # 先尝试以 weights_only=True 安全加载（适用于常见只保存 state_dict 的 checkpoint）
    try:
        checkpoint = torch.load(model_file, map_location=device, weights_only=True)
    except Exception as e:
        # 如果抛出 UnpicklingError / Weights only load failed，尝试用 safe_globals 回退加载
        print("Safe-load failed, attempting full load with safe_globals. Only do this if you trust the checkpoint.")
        # 临时允许 HighResolutionNet 反序列化
        with torch.serialization.safe_globals([HighResolutionNet]):
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    # checkpoint 可能是 dict(state_dict) / 直接 state_dict / nn.Module 对象等
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 该 dict 可能就是 state_dict
            state_dict = checkpoint
    else:
        # 如果保存的是整个 module 对象
        try:
            state_dict = checkpoint.state_dict()
        except Exception:
            raise RuntimeError("Unable to extract state_dict from checkpoint object")

    # 去掉 module. 前缀（如果有 DataParallel 保存）
    new_state = {}
    for k, v in state_dict.items():
        new_k = k[len('module.'):] if k.startswith('module.') else k
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model

def preprocess_image(image_path, input_size, mean=None, std=None):
    img = Image.open(image_path).convert('RGB')
    open_cv_image = np.array(img) 
    lab = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    img = Image.fromarray(final_img)
    orig_w, orig_h = img.size
    w = int(input_size[0]); h = int(input_size[1])
    img_resized = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0

    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean.reshape(1,1,3)) / std.reshape(1,1,3)

    # 如果你有通道反转（RGB->BGR）请在这里做，但会产生负 stride
    # arr = arr[:, :, ::-1]

    # 确保是连续内存，避免负 stride 导致 torch.from_numpy 报错
    arr = np.ascontiguousarray(arr)

    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    return tensor, orig_w, orig_h, (w, h)

def get_max_preds(batch_heatmaps):

    assert isinstance(batch_heatmaps, np.ndarray), "batch_heatmaps must be a numpy array"
    if batch_heatmaps.ndim != 4:
        raise ValueError("batch_heatmaps must have 4 dims (B,N,H,W) or (B,H,W,N)")

    # 如果是 B x H x W x N，尝试转置为 B x N x H x W
    B, C1, C2, C3 = batch_heatmaps.shape
    # 通过猜测 N 的位置来决定是否需要转置（若 N 很小或等于常见关键点数可判断）
    # 但最安全的做法是若第二维不是常见 heatmap 高度时尝试转置
    # 这里简单检测：如果第二维很大且第四维较小，可能是 BxHxWxN
    if C1 > C3 and C3 <= 4:  # heuristic: many关键点数量通常 <= 100，这里以 4 作为简单区分阈（可根据需要调整）
        # 可能是 B x H x W x N -> 转为 B x N x H x W
        batch_heatmaps = batch_heatmaps.transpose(0, 3, 1, 2)

    B, N, H, W = batch_heatmaps.shape
    preds = np.zeros((B, N, 2), dtype=np.float32)
    maxvals = np.zeros((B, N, 1), dtype=np.float32)

    for b in range(B):
        for j in range(N):
            hm = batch_heatmaps[b, j, :, :]
            # 找到最大位置
            idx = int(np.argmax(hm))
            y, x = divmod(idx, W)
            maxval = float(hm.flatten()[idx])
            preds[b, j, 0] = x
            preds[b, j, 1] = y
            maxvals[b, j, 0] = maxval

            # 次像素修正（基于局部梯度的简单偏移）
            # 确保索引在合法范围内（避免越界）
            if 1 < x < W - 2 and 1 < y < H - 2:
                # 注意：hm 的索引必须为整数
                dx = float(hm[int(y), int(x + 1)]) - float(hm[int(y), int(x - 1)])
                dy = float(hm[int(y + 1), int(x)]) - float(hm[int(y - 1), int(x)])
                if dx != 0:
                    preds[b, j, 0] += np.sign(dx) * 0.25
                if dy != 0:
                    preds[b, j, 1] += np.sign(dy) * 0.25

    return preds, maxvals

def parse_kp_indices(s):
    import re
    if s is None:
        return []

    # 若已经是 list/tuple，逐项转换为 int（忽略空项）
    if isinstance(s, (list, tuple)):
        res = []
        for item in s:
            if item is None:
                continue
            if isinstance(item, str) and item.strip() == '':
                continue
            res.append(int(item))
        return res

    # 若是单个整数
    if isinstance(s, (int, np.integer)):
        return [int(s)]

    # 若是字符串，使用正则抽取所有整数（支持负数，但 kp 索引通常非负）
    if isinstance(s, str):
        s_strip = s.strip()
        if s_strip == '':
            return []
        # 使用正则找出所有整数子串
        nums = re.findall(r'-?\d+', s_strip)
        return [int(x) for x in nums]

    # 其他类型尝试直接强制转换为 int 列表
    try:
        return [int(s)]
    except Exception:
        raise ValueError(f"Unsupported type for kp indices: {type(s)}")

def main():
    args = parse_args()

    # 1) 解析 cfg 路径：命令行 -> 上次记录 -> 自动搜索
    cfg_path = args.cfg
    if cfg_path is None:
        last = load_last_cfg()
        if last and os.path.exists(last):
            cfg_path = last
        else:
            found = auto_find_cfg()
            if found:
                cfg_path = found

    if cfg_path is None or not os.path.exists(cfg_path):
        raise RuntimeError("No cfg specified and cannot find one automatically. Provide --cfg PATH or place a yaml under experiments/ or run once with --cfg to save as default.")

    # 更新全局 config 并保存为 last
    update_config(config, args=type('X', (), {'cfg': cfg_path}))
    save_last_cfg(cfg_path)
    pprint.pprint(config)

    # 2) 命令行参数若为空，则从 cfg.INFER 读取默认
    infer_cfg = {}
    if hasattr(config, 'INFER'):
        infer_cfg = config.INFER

    model_file = args.model_file or getattr(infer_cfg, 'MODEL_FILE', None)
    image_file = args.image_file or getattr(infer_cfg, 'IMAGE_FILE', None)
    out_path = args.out or getattr(infer_cfg, 'OUT', 'vis_result.jpg')
    json_out = args.json_out or getattr(infer_cfg, 'JSON_OUT', 'keypoints.json')
    gpu = args.gpu if args.gpu is not None else getattr(infer_cfg, 'GPU', None)
    conf_thresh = args.conf_thresh if args.conf_thresh is not None else getattr(infer_cfg, 'CONF_THRESH', None)

    if model_file is None:
        raise RuntimeError("model-file not provided and not found in cfg.INFER.MODEL_FILE")
    if image_file is None:
        raise RuntimeError("image-file not provided and not found in cfg.INFER.IMAGE_FILE")

    # 设备
    if gpu == '' or gpu == '-1' or gpu is None:
        device = torch.device('cpu')
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = load_model(config, model_file, device)
    model.to(device)
    model.eval()

    # 读取预处理参数
    input_size = getattr(config.MODEL, 'IMAGE_SIZE', [256,256])
    mean = None
    std = None
    if hasattr(config, 'DATASET') and hasattr(config.DATASET, 'MEAN') and hasattr(config.DATASET, 'STD'):
        try:
            mean = np.array(config.DATASET.MEAN, dtype=np.float32)
            std = np.array(config.DATASET.STD, dtype=np.float32)
        except Exception:
            mean, std = None, None

    # 预处理并 forward
    input_tensor, orig_w, orig_h, (inp_w, inp_h) = preprocess_image(image_file, input_size, mean, std)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, (list, tuple)):
            heatmaps = outputs[0]
        else:
            heatmaps = outputs
        if isinstance(heatmaps, torch.Tensor):
            heatmaps_np = heatmaps.cpu().numpy()
        else:
            heatmaps_np = np.asarray(heatmaps)

    # 处理 heatmaps -> preds
    preds, maxvals = get_max_preds(heatmaps_np)
    hm_h, hm_w = heatmaps_np.shape[-2], heatmaps_np.shape[-1]
    scale_x = float(orig_w) / float(hm_w)
    scale_y = float(orig_h) / float(hm_h)
    preds[:, :, 0] *= scale_x
    preds[:, :, 1] *= scale_y

    # 读取 KP_INDICES 优先级：命令行 > cfg.INFER > 默认（全部）
    raw_kp = None
    if getattr(args, 'kp_indices', None):
        raw_kp = args.kp_indices
    else:
        infer_cfg = getattr(config, 'INFER', None)
        if infer_cfg is not None and hasattr(infer_cfg, 'KP_INDICES'):
            raw_kp = getattr(infer_cfg, 'KP_INDICES')

    kp_indices = parse_kp_indices(raw_kp)
    # 如果为空则默认全部（如果你明确只要 yaml 指定的，改为 raise 或警告）
    if not kp_indices:
        print("KP_INDICES not specified or empty -> default to all keypoints")
        kp_indices = list(range(preds.shape[1]))

    # 如果你的 YAML 使用的是 1-based 索引，请在这里转换：
    # kp_indices = [i-1 for i in kp_indices]

    # 过滤越界并保持给定顺序
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
        y_start_ratio = 0 # 去掉顶部 40% (背景/额头以上)
        y_end_ratio   = 1 # 去掉底部 10% (最底边缘)

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
    one_based = bool(args.one_based)
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

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, img_cv)
    print("Saved visualization to:", out_path)


        # 保存 JSON
    if json_out:
        json_dir = os.path.dirname(json_out)
        if json_dir and not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump({"image": image_file, "keypoints": selected}, f, ensure_ascii=False, indent=2)
        print(f"Saved keypoints JSON to {json_out}")

if __name__ == '__main__':
    main()