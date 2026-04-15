import io
import os
import cv2
import numpy as np
import imagehash
import threading
import urllib.parse
import torch
import torch.nn as nn
import math
import pillow_heif  # <--- 新增：引入 HEIC 支持库
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageStat, ExifTags
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file # <--- 修改这里，加上 send_file
from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN

# <--- 新增：注册 HEIC 打开器，让 PIL 能读取 HEIC
pillow_heif.register_heif_opener()

app = Flask(__name__)

# ==================== 1. 系统配置区域 ====================
# 这个默认路径现在只是作为前端输入框的初始值
# DEFAULT_WINDOWS_PATH = r'Y:' # <--- Run in my PC, map nas to Y drive
DEFAULT_WINDOWS_PATH = '/data' # <--- Run in Docker, NAS 挂载在 /data 目录下

PHOTO_DIR = os.environ.get('NAS_PHOTO_DIR', DEFAULT_WINDOWS_PATH)
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.webp'}
# 自动设置最大线程数 (CPU核心数 + 2)
MAX_WORKERS = (os.cpu_count() or 4) + 2

# ==================== 2. AI 模型初始化 ====================
# 检测计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"🔄 正在加载 AI 模型 (运行设备: {device}, 线程数: {MAX_WORKERS})...")

# --- 模型 A: MTCNN (人脸检测 & 关键点) ---
# 用于：人脸位置、5个关键点(眼/鼻/嘴)、人脸置信度
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7], margin=20)

# --- 模型 B: ResNet18 (语义特征提取) ---
# 用于：计算图片之间的“内容相似度”，判断是否为同一场景
weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)
resnet.fc = nn.Identity() # 移除最后的分类层，只取特征
resnet.to(device)
resnet.eval()

# 图片预处理 (ResNet 标准输入)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("✅ 模型加载完成，系统就绪。")

# ==================== 3. 扫描会话管理 (流式处理核心) ====================
class ScanSession:
    """
    管理文件列表的单例类。
    为了处理数万张照片，我们需要把文件列表缓存在内存中，
    前端每次只请求一小批 (Batch) 进行计算。
    """
    def __init__(self):
        self.all_files = []
        self.lock = threading.Lock()

    def load_files(self, root_dir):
        """递归扫描目录，加载所有图片路径"""
        file_list = []
        # 安全检查
        if not os.path.exists(root_dir):
            print(f"❌ 路径不存在: {root_dir}")
            return 0
            
        print(f"📂 正在扫描目录: {root_dir}")
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if '@eadir' not in d.lower()] # 过滤群晖缩略图
            for file in files:
                if os.path.splitext(file)[1].lower() in EXTENSIONS:
                    full_path = os.path.join(root, file)
                    if not self._is_system_file(full_path):
                        file_list.append(full_path)
        
        # 按修改时间排序，这样相似的照片（连拍）会物理相邻，提高分组效率
        file_list.sort(key=lambda x: os.path.getmtime(x))
        
        with self.lock:
            self.all_files = file_list
        return len(file_list)

    def _is_system_file(self, path):
        if '@eadir' in path.lower() or 'synofile_thumb' in os.path.basename(path).lower():
            return True
        return False

    def get_batch(self, start_index, count):
        with self.lock:
            if start_index >= len(self.all_files): return []
            return self.all_files[start_index : start_index + count]

scan_session = ScanSession()

# ==================== 4. 核心算法库 (数学/图像处理) ====================

def calculate_laplacian_variance(img_gray):
    """计算拉普拉斯方差 (清晰度评价标准)"""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def analyze_rule_of_thirds(width, height, center_x, center_y):
    """
    三分法构图评分
    计算物体中心点距离画面黄金分割线(1/3, 2/3)的距离。
    """
    third_w = width / 3
    third_h = height / 3
    
    # 也就是画面的4个黄金交叉点
    points = [
        (third_w, third_h), (third_w * 2, third_h),
        (third_w, third_h * 2), (third_w * 2, third_h * 2)
    ]
    
    # 找最近的一个点
    min_dist = float('inf')
    for px, py in points:
        dist = math.sqrt((center_x - px)**2 + (center_y - py)**2)
        min_dist = min(min_dist, dist)
    
    # 归一化：距离越近分越高。最大容忍距离为宽度的 1/4
    max_dist = width / 4
    score = max(0, 1 - (min_dist / max_dist))
    return score * 100

def analyze_hsv_stats(cv_img):
    """
    色彩分析：计算 HSV 空间的统计数据
    """
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # 饱和度均值 (色彩是否丰富)
    sat_mean = np.mean(s)
    # 亮度对比度 (亮度的标准差)
    contrast = np.std(v)
    # 色彩均匀性 (色相的标准差，越低越纯，越高越杂)
    hue_std = np.std(h)
    
    return sat_mean, contrast, hue_std

def estimate_saliency_center(cv_img):
    """
    估算视觉重心 (简化版 Saliency)
    使用高斯模糊后的亮度+饱和度加权重心，模拟人眼关注点。
    """
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    # 简单的显著性图 = 饱和度 + 亮度 (人眼趋向于亮和鲜艳的地方)
    saliency_map = cv2.addWeighted(s, 0.5, v, 0.5, 0)
    saliency_map = cv2.GaussianBlur(saliency_map, (21, 21), 0)
    
    # 计算重心 (Moments)
    M = cv2.moments(saliency_map)
    if M["m00"] == 0:
        return cv_img.shape[1]/2, cv_img.shape[0]/2
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

# ==================== 5. 人物照片评分逻辑 (分项加权) ====================

def score_person_image(cv_img, pil_img, box, landmarks, global_sharpness):
    """
    人物照片评分模型
    总分 = 0.25清晰度 + 0.20表情 + 0.15眼睛 + 0.10构图 + 0.10光照 + 0.10无遮挡 + 0.10情绪
    """
    scores = {}
    
    # 1. 基础数据准备
    h, w, _ = cv_img.shape
    x1, y1, x2, y2 = [int(b) for b in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_w = x2 - x1
    face_h = y2 - y1
    face_area = face_w * face_h
    face_roi = cv_img[y1:y2, x1:x2]
    
    # 如果人脸太小或切出去了，给低分
    if face_roi.size == 0 or face_w < 20:
        return 0, {}

    # ----------------------------------------------------
    # 维度 1: 清晰度 (25%) - Laplacian Variance
    # ----------------------------------------------------
    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
    face_sharpness = calculate_laplacian_variance(face_gray)
    # 归一化：通常 > 500 算极度清晰， < 50 算模糊
    s_sharpness = min(face_sharpness / 5.0, 100) 
    scores['sharpness'] = s_sharpness

    # ----------------------------------------------------
    # 维度 2: 表情自然度 & 7. 情绪强度 (共 30%)
    # ----------------------------------------------------
    # MTCNN 关键点: [左眼, 右眼, 鼻, 左嘴, 右嘴]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    nose = landmarks[2]
    
    # 计算嘴巴宽度与脸宽的比例 (微笑时嘴角会上扬且变宽)
    mouth_width = math.sqrt((left_mouth[0]-right_mouth[0])**2 + (left_mouth[1]-right_mouth[1])**2)
    mouth_ratio = mouth_width / face_w
    
    # 简单的微笑启发式算法：嘴宽比例适中且嘴角高于平均水平
    # 情绪强度：嘴巴张开程度 (模拟)
    s_emotion = min(mouth_ratio * 200, 100) # 嘴越宽情绪越强
    scores['emotion'] = s_emotion
    
    # 表情自然度：这里用正脸对称性模拟，越对称看起来越舒服
    diff_y = abs(left_mouth[1] - right_mouth[1])
    s_natural = max(0, 100 - (diff_y / face_h * 500))
    scores['expression'] = s_natural

    # ----------------------------------------------------
    # 维度 3: 眼睛状态 (15%) - 模拟 EAR
    # ----------------------------------------------------
    # 由于只有5点，无法算精准EAR。我们切出眼睛区域，算对比度。
    # 睁眼时：黑瞳孔+白眼球 -> 对比度高 / 边缘能量高
    # 闭眼时：皮肤 -> 对比度低
    eye_center = landmarks[0] # 左眼
    ex, ey = int(eye_center[0]), int(eye_center[1])
    er = int(face_w * 0.15) # 眼睛半径
    eye_roi = cv_img[max(0, ey-er):min(h, ey+er), max(0, ex-er):min(w, ex+er)]
    
    if eye_roi.size > 0:
        eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2GRAY)
        eye_contrast = eye_gray.std() # 亮度标准差
        s_eye = min(eye_contrast * 2, 100)
    else:
        s_eye = 50
    scores['eye'] = s_eye

    # ----------------------------------------------------
    # 维度 4: 构图 (10%) - 三分法
    # ----------------------------------------------------
    face_cx = (x1 + x2) / 2
    face_cy = (y1 + y2) / 2
    s_comp = analyze_rule_of_thirds(w, h, face_cx, face_cy)
    scores['composition'] = s_comp

    # ----------------------------------------------------
    # 维度 5: 光照质量 (10%)
    # ----------------------------------------------------
    # 理想的光照：亮度适中 (V通道 100-200)，且不是死黑或过曝
    hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2HSV)
    brightness = hsv_roi[:,:,2].mean()
    # 距离 150 (中间亮度) 越近越好
    dist_light = abs(brightness - 150)
    s_light = max(0, 100 - (dist_light * 0.8))
    scores['light'] = s_light

    # ----------------------------------------------------
    # 维度 6: 无遮挡 (10%) - 简单模拟
    # ----------------------------------------------------
    # 如果人脸检测置信度极高，说明特征明显，无遮挡。
    # 这里我们用人脸面积占比来辅助判断，面积太小视为不可辨识
    face_ratio = face_area / (w * h)
    s_occlusion = min(face_ratio * 500, 100) # 占比 > 20% 满分
    scores['occlusion'] = s_occlusion

    # === 加权汇总 ===
    final_score = (
        0.25 * s_sharpness +
        0.20 * s_natural +
        0.15 * s_eye +
        0.10 * s_comp +
        0.10 * s_light +
        0.10 * s_occlusion +
        0.10 * s_emotion
    )
    
    return final_score, scores

# ==================== 6. 景物照片评分逻辑 (全局分析) ====================

def score_scenery_image(cv_img, global_sharpness):
    """
    景物照片评分模型
    总分 = 0.30构图 + 0.20色彩 + 0.20光影 + 0.15清晰度 + 0.15视觉平衡
    """
    scores = {}
    h, w, _ = cv_img.shape
    
    # 1. 基础统计
    sat_mean, contrast, hue_std = analyze_hsv_stats(cv_img)
    
    # ----------------------------------------------------
    # 维度 1: 构图评分 (30%) - 视觉重心与三分线
    # ----------------------------------------------------
    # 估算视觉重心
    cx, cy = estimate_saliency_center(cv_img)
    s_comp = analyze_rule_of_thirds(w, h, cx, cy)
    scores['composition'] = s_comp
    
    # ----------------------------------------------------
    # 维度 2: 色彩和谐度 (20%)
    # ----------------------------------------------------
    # 饱和度适中 (50-150) 且 色相不要太杂乱
    s_color = min(sat_mean, 100) 
    # 如果颜色太杂 (hue_std 高)，扣分
    if hue_std > 80: s_color *= 0.8
    scores['color'] = s_color
    
    # ----------------------------------------------------
    # 维度 3: 光影层次 (20%) - 对比度
    # ----------------------------------------------------
    # 对比度越高，画面越通透
    s_light = min(contrast * 1.5, 100)
    scores['light'] = s_light
    
    # ----------------------------------------------------
    # 维度 4: 清晰度 (15%)
    # ----------------------------------------------------
    s_sharpness = min(global_sharpness / 3.0, 100) # 风景通常细节多，分母小点
    scores['sharpness'] = s_sharpness
    
    # ----------------------------------------------------
    # 维度 5: 视觉复杂度平衡 (15%)
    # ----------------------------------------------------
    # 简单的边缘密度检测。边缘太多=杂乱，太少=空洞
    edges = cv2.Canny(cv_img, 100, 200)
    edge_ratio = np.count_nonzero(edges) / (w * h)
    # 期望边缘占比在 5% - 15% 之间
    dist_edge = abs(edge_ratio - 0.10)
    s_balance = max(0, 100 - (dist_edge * 1000))
    scores['balance'] = s_balance
    
    # === 加权汇总 ===
    final_score = (
        0.30 * s_comp +
        0.20 * s_color +
        0.20 * s_light +
        0.15 * s_sharpness +
        0.15 * s_balance
    )
    
    return final_score, scores

# ==================== 7. 主流程逻辑 ====================

def analyze_image_task(path):
    """单张图片处理入口"""
    try:
        # 读取
        pil_img = Image.open(path).convert('RGB')
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # 1. 基础特征 (Hash & ResNet) - 用于分组
        phash = imagehash.phash(pil_img)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature_vector = resnet(img_tensor).cpu().numpy().flatten()
            
        # 2. 转 OpenCV 格式用于评分
        cv_img = np.array(pil_img)
        # 全局清晰度
        global_sharpness = calculate_laplacian_variance(cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY))

        # 3. 智能检测 (人脸)
        boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
        
        ai_data = {
            "type": "scenery", # 默认为风景
            "details": {}      # 详细得分表
        }
        
        final_score = 0
        
        if boxes is not None:
            # --- 判定为人物照 ---
            ai_data["type"] = "person"
            ai_data["face_count"] = len(boxes)
            
            # 找主角 (最大的人脸)
            max_area = 0
            best_idx = -1
            for i, box in enumerate(boxes):
                area = (box[2]-box[0]) * (box[3]-box[1])
                if area > max_area:
                    max_area = area
                    best_idx = i
            
            # 对主角进行人物评分
            if best_idx >= 0:
                final_score, score_details = score_person_image(
                    cv_img, pil_img, boxes[best_idx], landmarks[best_idx], global_sharpness
                )
                ai_data["details"] = score_details
        else:
            # --- 判定为风景照 ---
            final_score, score_details = score_scenery_image(cv_img, global_sharpness)
            ai_data["details"] = score_details

        return {
            "path": path,
            "hash": phash,
            "hash_str": str(phash),
            "vector": feature_vector,
            "score": round(final_score, 1),
            "ai_data": ai_data
        }

    except Exception as e:
        print(f"❌ Error processing {path}: {e}")
        return None

def compute_cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def process_batch(file_paths, threshold=10):
    """批量处理并聚类"""
    results = []
    
    # 修改点：使用根据 CPU 核心数计算出的 MAX_WORKERS
    # 这将大幅提升 CPU 利用率
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for res in executor.map(analyze_image_task, file_paths):
            if res: results.append(res)
            
    if not results: return []

    # 按 Hash 预排序
    results.sort(key=lambda x: x['hash_str'])
    
    groups = []
    processed = set()
    SEMANTIC_THRESHOLD = 0.92 
    
    for i in range(len(results)):
        if results[i]['path'] in processed: continue
        current_group = [results[i]]
        processed.add(results[i]['path'])
        
        vec_a = results[i]['vector']
        
        for j in range(i + 1, len(results)):
            if results[j]['path'] in processed: continue
            
            vec_b = results[j]['vector']
            similarity = compute_cosine_similarity(vec_a, vec_b)
            hash_diff = results[i]['hash'] - results[j]['hash']
            
            if similarity > SEMANTIC_THRESHOLD or hash_diff <= threshold:
                current_group.append(results[j])
                processed.add(results[j]['path'])
        
        # 即使只有1张，为了不漏掉照片，如果是前端请求的，还是建议返回
        # 但为了逻辑统一，这里还是保持 >1，或者你可以改为 >=1
        if len(current_group) > 1:
            for item in current_group:
                item['hash'] = str(item['hash'])
                if 'vector' in item: del item['vector'] 
                # 关键：这里生成的路径已经可以直接用于 img src
                item['display_path'] = f"/image_raw?path={urllib.parse.quote(item['path'])}"
            groups.append(current_group)
            
    return groups

# ==================== 8. 路由接口 ====================

@app.route('/')
def index(): 
    # 将默认路径传给前端
    return render_template('index.html', default_path=DEFAULT_WINDOWS_PATH)

@app.route('/scan/init')
def init_scan():
    # 获取前端传来的自定义路径
    target_path = request.args.get('path', DEFAULT_WINDOWS_PATH)
    
    if not os.path.exists(target_path):
        return jsonify({"error": f"路径不存在: {target_path}"}), 400
        
    count = scan_session.load_files(target_path)
    return jsonify({"total": count, "path": target_path})

@app.route('/scan/batch')
def scan_batch():
    start = int(request.args.get('start', 0))
    count = int(request.args.get('count', 20))
    files = scan_session.get_batch(start, count)
    
    groups = process_batch(files)
    has_more = (start + count) < len(scan_session.all_files)
    
    return jsonify({"groups": groups, "has_more": has_more})

@app.route('/image_raw')
def serve_image():
    raw_path = request.args.get('path')
    if not raw_path: return "Err", 400
    
    # 获取文件后缀名
    ext = os.path.splitext(raw_path)[1].lower()
    
    # 如果是 HEIC 格式，进行实时转码
    if ext == '.heic':
        try:
            # 1. 打开 HEIC 图片
            img = Image.open(raw_path)
            # 2. 修正旋转方向 (手机照片常有的问题)
            img = ImageOps.exif_transpose(img)
            # 3. 转为 RGB 模式 (防止部分 HEIC 是 CMYK 或 RGBA 导致保存 JPEG 失败)
            img = img.convert('RGB')
            
            # 4. 保存为 JPEG 字节流到内存中
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG', quality=80) # quality=80 兼顾清晰度和速度
            img_io.seek(0)
            
            # 5. 发送给浏览器，伪装成 jpg
            return send_file(img_io, mimetype='image/jpeg')
        except Exception as e:
            print(f"HEIC 预览失败: {e}")
            return "Error converting HEIC", 500
            
    # 如果是普通图片 (jpg, png, webp)，直接发送文件，效率最高
    else:
        return send_from_directory(os.path.dirname(raw_path), os.path.basename(raw_path))

@app.route('/delete', methods=['POST'])
def delete_files():
    files = request.json.get('files', [])
    deleted = []
    for f in files:
        try: os.remove(f); deleted.append(f)
        except: pass
    return jsonify({"deleted": deleted})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 