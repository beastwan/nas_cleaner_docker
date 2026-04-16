"""
AI 照片选片系统 - 滑动窗口优化版
基于 Flask 的 Web 应用，用于在 NAS 上运行，通过 Docker 部署
集成了滑动窗口算法、持久化存储和缩略图优化
"""
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
import pillow_heif
import sqlite3
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageStat, ExifTags
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN

# 导入自定义模块
from database import PhotoDatabase
from sliding_window import SlidingWindowScanner, PhotoItem, PhotoGroup

# <--- 新增：注册 HEIC 打开器，让 PIL 能读取 HEIC
pillow_heif.register_heif_opener()

app = Flask(__name__)

# ==================== 1. 系统配置区域 ====================
# 默认路径配置 - 智能检测可用路径
def get_default_path():
    """智能检测默认路径，优先使用存在的路径"""
    # 可能的路径列表（按优先级排序）
    possible_paths = [
        '/homes',          # Docker映射路径
        '/data',           # 备用路径
        'D:/photos',       # Windows本地测试路径
        'C:/Users/china/Pictures'  # Windows用户图片目录
    ]
    
    # 检查每个路径是否存在
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 检测到可用路径: {path}")
            return path
    
    # 如果没有找到存在的路径，返回第一个作为默认
    print(f"⚠️  未找到存在的路径，使用默认: {possible_paths[0]}")
    return possible_paths[0]

DEFAULT_WINDOWS_PATH = get_default_path()  # 智能检测默认路径
PHOTO_DIR = os.environ.get('NAS_PHOTO_DIR', DEFAULT_WINDOWS_PATH)
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.bmp', '.tiff', '.tif'}

# 自动设置最大线程数 (CPU核心数 + 2)
MAX_WORKERS = (os.cpu_count() or 4) + 2

# 缩略图配置
THUMBNAIL_SIZE = (400, 400)  # 缩略图大小
THUMBNAIL_QUALITY = 80  # JPEG 质量

# 数据库配置
DB_PATH = "scan_history.db"

# ==================== 文件列表缓存 ====================
# 缓存扫描路径的文件列表，避免重复遍历文件系统
file_cache = {}  # scan_path -> [file_list]
file_cache_lock = threading.Lock()  # 缓存访问锁

def get_cached_file_list(scan_path):
    """
    获取缓存的文件列表，如果未缓存则扫描并缓存
    返回：文件路径列表
    """
    with file_cache_lock:
        if scan_path not in file_cache:
            print(f"🔄 扫描并缓存文件列表: {scan_path}")
            file_list = []
            start_time = time.time()
            
            for root, dirs, files in os.walk(scan_path):
                # 过滤群晖缩略图目录
                dirs[:] = [d for d in dirs if '@eadir' not in d.lower() and not d.startswith('.')]
                
                for file in files:
                    # 👉 【修改点 2】：跳过 macOS 或群晖产生的隐藏文件
                    if file.startswith('.'):
                        continue
                    if os.path.splitext(file)[1].lower() in EXTENSIONS:
                        full_path = os.path.join(root, file)
                        file_list.append(full_path)
            
            file_cache[scan_path] = file_list
            elapsed = time.time() - start_time
            print(f"✅ 文件列表缓存完成: {len(file_list)} 个文件, 耗时 {elapsed:.2f} 秒")
        
        return file_cache[scan_path]

def clear_file_cache(scan_path=None):
    """
    清理文件缓存
    scan_path: 如果为None，清理所有缓存；否则清理指定路径的缓存
    """
    with file_cache_lock:
        if scan_path is None:
            file_cache.clear()
            print("🗑️ 已清理所有文件缓存")
        elif scan_path in file_cache:
            del file_cache[scan_path]
            print(f"🗑️ 已清理路径缓存: {scan_path}")

# ==================== 2. AI 模型初始化 ====================
# 检测计算设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"🔄 正在加载 AI 模型 (运行设备: {device}, 线程数: {MAX_WORKERS})...")

# --- 模型 A: MTCNN (人脸检测 & 关键点) ---
# 用于：人脸位置、5个关键点(眼/鼻/嘴)、人脸置信度
mtcnn = MTCNN(keep_all=True, device=device, thresholds=[0.6, 0.7, 0.7], margin=20)

# --- 模型 B: ResNet18 (语义特征提取) ---
# 用于：计算图片之间的"内容相似度"，判断是否为同一场景
weights = models.ResNet18_Weights.DEFAULT
resnet = models.resnet18(weights=weights)
resnet.fc = nn.Identity()  # 移除最后的分类层，只取特征
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

# ==================== 3. 数据库和扫描器初始化 ====================
# 初始化数据库
db = PhotoDatabase(DB_PATH)

# 全局扫描会话管理
class ScanSessionManager:
    """扫描会话管理器"""
    
    def __init__(self):
        self.active_scanner = None
        self.active_session = None
        self.is_running = False
        self.is_paused = False
        self.lock = threading.Lock()
        
        # 从数据库恢复活动会话
        self._restore_session()
    
    def _restore_session(self):
        """从数据库恢复活动会话"""
        session = db.get_active_session()
        if session:
            print(f"🔄 恢复活动会话: {session['scan_path']} ({session['processed_count']}/{session['total_files']})")
            self.active_session = session
            
            # 创建扫描器
            self.active_scanner = SlidingWindowScanner(
                window_size=session['window_size'],
                similarity_threshold=session['similarity_threshold']
            )
            
            # 加载忽略的照片
            ignored = db.get_ignored_photos()
            self.active_scanner.set_ignored_paths(ignored)
            
            self.is_running = (session['status'] == 'scanning')
            self.is_paused = (session['status'] == 'paused')
    
    def start_session(self, scan_path: str, total_files: int, 
                     window_size: int = 20, similarity_threshold: float = 0.85):
        """开始新的扫描会话"""
        with self.lock:
            # 停止现有会话
            if self.active_scanner:
                self.active_scanner.clear()
            
            # 创建新会话
            session_id = db.create_scan_session(
                scan_path, total_files, window_size, similarity_threshold
            )
            
            # 创建扫描器
            self.active_scanner = SlidingWindowScanner(
                window_size=window_size,
                similarity_threshold=similarity_threshold
            )
            
            # 加载忽略的照片
            ignored = db.get_ignored_photos()
            self.active_scanner.set_ignored_paths(ignored)
            
            self.active_session = {
                'id': session_id,
                'scan_path': scan_path,
                'total_files': total_files,
                'processed_count': 0,
                'window_size': window_size,
                'similarity_threshold': similarity_threshold,
                'status': 'scanning'
            }
            
            self.is_running = True
            self.is_paused = False
    
    def pause_session(self):
        """暂停当前会话"""
        with self.lock:
            if self.active_session and self.is_running:
                self.is_paused = True
                self.is_running = False
                db.update_scan_session(
                    self.active_session['id'],
                    self.active_session['processed_count'],
                    'paused'
                )
    
    def resume_session(self):
        """恢复当前会话"""
        with self.lock:
            if self.active_session and self.is_paused:
                self.is_paused = False
                self.is_running = True
                db.update_scan_session(
                    self.active_session['id'],
                    self.active_session['processed_count'],
                    'scanning'
                )
    
    def stop_session(self):
        """停止当前会话"""
        with self.lock:
            if self.active_session:
                self.is_running = False
                self.is_paused = False
                db.update_scan_session(
                    self.active_session['id'],
                    self.active_session['processed_count'],
                    'completed'
                )
                self.active_session = None
                if self.active_scanner:
                    self.active_scanner.clear()
    
    def update_progress(self, processed_count: int):
        """更新进度"""
        with self.lock:
            if self.active_session:
                self.active_session['processed_count'] = processed_count
                if self.is_running:
                    db.update_scan_session(
                        self.active_session['id'],
                        processed_count,
                        'scanning'
                    )
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        with self.lock:
            if not self.active_session:
                return {
                    'is_running': False,
                    'is_paused': False,
                    'has_session': False
                }
            
            return {
                'is_running': self.is_running,
                'is_paused': self.is_paused,
                'has_session': True,
                'session': self.active_session,
                'window_info': self.active_scanner.get_window_info() if self.active_scanner else None
            }
    
    def get_scanner(self) -> Optional[SlidingWindowScanner]:
        """获取当前扫描器"""
        with self.lock:
            return self.active_scanner

# 创建全局会话管理器
session_manager = ScanSessionManager()

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

# ==================== 5. 改进的评分逻辑 ====================

def score_person_image_improved(cv_img, pil_img, box, landmarks, global_sharpness):
    """
    改进的人物照片评分模型
    参考 Android 应用的评分思路，结合更多因素
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
    # 维度 2: 表情自然度 (20%)
    # ----------------------------------------------------
    # MTCNN 关键点: [左眼, 右眼, 鼻, 左嘴, 右嘴]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    
    # 计算嘴巴对称性
    diff_y = abs(left_mouth[1] - right_mouth[1])
    s_expression = max(0, 100 - (diff_y / face_h * 500))
    scores['expression'] = s_expression

    # ----------------------------------------------------
    # 维度 3: 眼睛状态 (15%) - 模拟睁眼程度
    # ----------------------------------------------------
    eye_center = landmarks[0]  # 左眼
    ex, ey = int(eye_center[0]), int(eye_center[1])
    er = int(face_w * 0.15)  # 眼睛半径
    eye_roi = cv_img[max(0, ey-er):min(h, ey+er), max(0, ex-er):min(w, ex+er)]
    
    if eye_roi.size > 0:
        eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2GRAY)
        eye_contrast = eye_gray.std()  # 亮度标准差
        s_eye = min(eye_contrast * 2, 100)
    else:
        s_eye = 50
    scores['eye'] = s_eye

    # ----------------------------------------------------
    # 维度 4: 构图 (15%) - 三分法
    # ----------------------------------------------------
    face_cx = (x1 + x2) / 2
    face_cy = (y1 + y2) / 2
    s_comp = analyze_rule_of_thirds(w, h, face_cx, face_cy)
    scores['composition'] = s_comp

    # ----------------------------------------------------
    # 维度 5: 光照质量 (10%)
    # ----------------------------------------------------
    hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2HSV)
    brightness = hsv_roi[:,:,2].mean()
    # 距离 150 (中间亮度) 越近越好
    dist_light = abs(brightness - 150)
    s_light = max(0, 100 - (dist_light * 0.8))
    scores['light'] = s_light

    # ----------------------------------------------------
    # 维度 6: 人脸大小和位置 (10%)
    # ----------------------------------------------------
    face_ratio = face_area / (w * h)
    s_face_size = min(face_ratio * 500, 100)  # 占比 > 20% 满分
    scores['face_size'] = s_face_size

    # ----------------------------------------------------
    # 维度 7: 整体色彩和谐度 (5%)
    # ----------------------------------------------------
    sat_mean, contrast, hue_std = analyze_hsv_stats(cv_img)
    s_color = min(sat_mean, 100)
    if hue_std > 80:  # 颜色太杂，扣分
        s_color *= 0.8
    scores['color'] = s_color

    # === 加权汇总 ===
    final_score = (
        0.25 * s_sharpness +
        0.20 * s_expression +
        0.15 * s_eye +
        0.15 * s_comp +
        0.10 * s_light +
        0.10 * s_face_size +
        0.05 * s_color
    )
    
    return final_score, scores

def score_scenery_image_improved(cv_img, global_sharpness):
    """
    改进的景物照片评分模型
    """
    scores = {}
    h, w, _ = cv_img.shape
    
    # 1. 基础统计
    sat_mean, contrast, hue_std = analyze_hsv_stats(cv_img)
    
    # ----------------------------------------------------
    # 维度 1: 构图评分 (30%) - 视觉重心与三分线
    # ----------------------------------------------------
    cx, cy = estimate_saliency_center(cv_img)
    s_comp = analyze_rule_of_thirds(w, h, cx, cy)
    scores['composition'] = s_comp
    
    # ----------------------------------------------------
    # 维度 2: 色彩和谐度 (25%)
    # ----------------------------------------------------
    s_color = min(sat_mean, 100)
    # 如果颜色太杂 (hue_std 高)，扣分
    if hue_std > 80: 
        s_color *= 0.7
    scores['color'] = s_color
    
    # ----------------------------------------------------
    # 维度 3: 光影层次 (20%) - 对比度
    # ----------------------------------------------------
    s_light = min(contrast * 1.5, 100)
    scores['light'] = s_light
    
    # ----------------------------------------------------
    # 维度 4: 清晰度 (15%)
    # ----------------------------------------------------
    s_sharpness = min(global_sharpness / 3.0, 100)
    scores['sharpness'] = s_sharpness
    
    # ----------------------------------------------------
    # 维度 5: 视觉复杂度平衡 (10%)
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
        0.25 * s_color +
        0.20 * s_light +
        0.15 * s_sharpness +
        0.10 * s_balance
    )
    
    return final_score, scores

# ==================== 6. 照片处理核心函数 ====================
def safe_read_image(path):
    """安全读取图片，完美支持带有中文或特殊字符的路径"""
    try:
        # 使用 numpy + imdecode 代替 cv2.imread，解决中文路径读取报错
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取图片失败: {path}, 错误: {e}")
        return None

def analyze_image_task(path):
    """单张图片处理入口 - 改进版"""
    try:
        # 检查是否已处理过
        if db.is_photo_processed(path):
            print(f"⏭️ 跳过已处理照片: {path}")
            return None
        
        # 🛡️ === 关键修改点：增加针对读取失败的精准拦截 === 🛡️
        try:
            pil_img = Image.open(path).convert('RGB')
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception as read_err:
            print(f"⚠️ [跳过] 无法读取图片 {path} (可能是格式不支持或损坏文件): {read_err}")
            return None  # 遇到坏图片，直接返回 None，让外层循环继续扫描下一张
        
        # 获取文件信息
        file_size = os.path.getsize(path)
        modified_time = os.path.getmtime(path)
        
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
            "type": "scenery",  # 默认为风景
            "details": {}       # 详细得分表
        }
        
        final_score = 0
        
        if boxes is not None and len(boxes) > 0:
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
                final_score, score_details = score_person_image_improved(
                    cv_img, pil_img, boxes[best_idx], landmarks[best_idx], global_sharpness
                )
                ai_data["details"] = score_details
        else:
            # --- 判定为风景照 ---
            final_score, score_details = score_scenery_image_improved(cv_img, global_sharpness)
            ai_data["details"] = score_details

        # 创建照片数据字典
        photo_data = {
            "path": path,
            "hash": str(phash),
            "file_size": file_size,
            "modified_time": modified_time,
            "feature_vector": feature_vector.tolist(),
            "score": round(final_score, 1),
            "ai_type": ai_data["type"],
            "ai_details": ai_data["details"]
        }
        
        # 保存到数据库
        photo_id = db.save_photo_result(photo_data)
        
        # 创建 PhotoItem 对象
        photo_item = PhotoItem(
            path=path,
            file_hash=str(phash),
            feature_vector=feature_vector.tolist(),
            score=round(final_score, 1),
            ai_type=ai_data["type"],
            ai_details=ai_data["details"],
            file_size=file_size,
            modified_time=modified_time
        )
        
        return photo_item

    except Exception as e:
        # 这个外层的大 except 用来捕获后面的 AI 计算错误 (比如张量形状不对、内存溢出等)
        print(f"❌ [AI处理异常] 处理 {path} 时出错: {e}")
        return None

def compute_cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: 
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# ==================== 7. 批量处理函数 ====================

def process_batch_with_sliding_window(file_paths):
    """使用滑动窗口算法批量处理照片"""
    scanner = session_manager.get_scanner()
    if not scanner:
        return []
    
    # 过滤已处理照片
    filtered_paths = []
    for path in file_paths:
        if not db.is_photo_processed(path):
            filtered_paths.append(path)
    
    if not filtered_paths:
        return []
    
    results = []
    new_groups = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for photo_item in executor.map(analyze_image_task, filtered_paths):
            if photo_item:
                results.append(photo_item)
                
                # 使用滑动窗口处理照片
                group = scanner.process_photo(photo_item)
                if group:
                    new_groups.append(group)
                    
                    # 保存组到数据库
                    group_hash = group.group_hash
                    best_photo_id = None
                    
                    # 查找最佳照片的数据库ID
                    if group.best_photo:
                        # 这里需要根据路径查找照片ID
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT id FROM processed_photos WHERE file_path = ?",
                            (group.best_photo.path,)
                        )
                        row = cursor.fetchone()
                        if row:
                            best_photo_id = row[0]
                        conn.close()
                    
                    # 创建组
                    group_id = db.create_photo_group(group_hash, best_photo_id)
                    
                    # 添加组成员
                    for photo in group.photos:
                        # 查找照片ID
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT id FROM processed_photos WHERE file_path = ?",
                            (photo.path,)
                        )
                        row = cursor.fetchone()
                        if row:
                            photo_id = row[0]
                            is_best = (photo.path == group.best_photo.path) if group.best_photo else False
                            db.add_photo_to_group(group_id, photo_id, is_best)
                        conn.close()
    
    # 更新进度
    if session_manager.active_session:
        session_manager.update_progress(
            session_manager.active_session['processed_count'] + len(filtered_paths)
        )
    
    # 转换组为前端格式
    frontend_groups = []
    for group in new_groups:
        group_data = []
        for photo in group.photos:
            group_data.append({
                "path": photo.path,
                "hash": photo.file_hash,
                "score": photo.score,
                "ai_data": {
                    "type": photo.ai_type,
                    "details": photo.ai_details
                },
                "display_path": f"/image_thumb?path={urllib.parse.quote(photo.path)}",
                "original_path": f"/image_raw?path={urllib.parse.quote(photo.path)}"
            })
        
        # 按分数排序
        group_data.sort(key=lambda x: x['score'], reverse=True)
        
        frontend_groups.append(group_data)
    
    return frontend_groups

# ==================== 8. 路由接口 ====================

@app.route('/')
def index():
    """主页"""
    # 获取当前状态
    status = session_manager.get_status()
    
    # 获取配置参数
    window_size = request.args.get('window_size', 20, type=int)
    similarity_threshold = request.args.get('similarity_threshold', 0.85, type=float)
    
    return render_template(
        'index_new.html',
        default_path=DEFAULT_WINDOWS_PATH,
        window_size=window_size,
        similarity_threshold=similarity_threshold,
        session_status=status
    )

@app.route('/api/status')
def get_status():
    """获取当前状态"""
    status = session_manager.get_status()
    return jsonify(status)

@app.route('/api/scan/init', methods=['POST'])
def init_scan():
    """初始化扫描"""
    data = request.json
    scan_path = data.get('path', DEFAULT_WINDOWS_PATH)
    window_size = data.get('window_size', 20)
    similarity_threshold = data.get('similarity_threshold', 0.85)
    
    if not os.path.exists(scan_path):
        return jsonify({"error": f"路径不存在: {scan_path}"}), 400
    
    # 清理旧的缓存（如果有）
    clear_file_cache(scan_path)
    
    # 获取文件列表（会自动缓存）
    try:
        file_list = get_cached_file_list(scan_path)
    except Exception as e:
        print(f"❌ 扫描文件列表失败: {e}")
        return jsonify({"error": f"扫描文件列表失败: {str(e)}"}), 500
    
    total_files = len(file_list)
    
    # 开始新会话
    session_manager.start_session(
        scan_path, total_files, window_size, similarity_threshold
    )
    
    return jsonify({
        "success": True,
        "total": total_files,
        "path": scan_path,
        "window_size": window_size,
        "similarity_threshold": similarity_threshold,
        "cache_info": {
            "cached": True,
            "file_count": total_files
        }
    })

@app.route('/api/scan/batch')
def scan_batch():
    """获取下一批处理结果 - 使用缓存文件列表"""
    batch_size = request.args.get('batch_size', 20, type=int)
    
    scanner = session_manager.get_scanner()
    if not scanner or not session_manager.active_session:
        return jsonify({"error": "没有活动的扫描会话"}), 400
    
    # 获取文件列表
    scan_path = session_manager.active_session['scan_path']
    processed_count = session_manager.active_session['processed_count']
    
    # 使用缓存的文件列表
    try:
        all_files = get_cached_file_list(scan_path)
    except Exception as e:
        print(f"❌ 获取缓存文件列表失败: {e}")
        return jsonify({"error": "获取文件列表失败"}), 500
    
    # 计算偏移量
    offset = processed_count
    batch_files = all_files[offset:offset + batch_size]
    
    # 处理批次
    groups = process_batch_with_sliding_window(batch_files)
    
    # 检查是否完成
    has_more = (offset + len(batch_files)) < len(all_files)
    
    return jsonify({
        "groups": groups,
        "has_more": has_more,
        "processed": session_manager.active_session['processed_count'],
        "total": session_manager.active_session['total_files'],
        "batch_info": {
            "offset": offset,
            "batch_size": len(batch_files),
            "total_files": len(all_files)
        }
    })

@app.route('/api/scan/pause', methods=['POST'])
def pause_scan():
    """暂停扫描"""
    session_manager.pause_session()
    return jsonify({"success": True, "status": "paused"})

@app.route('/api/scan/resume', methods=['POST'])
def resume_scan():
    """恢复扫描"""
    session_manager.resume_session()
    return jsonify({"success": True, "status": "resumed"})

@app.route('/api/scan/stop', methods=['POST'])
def stop_scan():
    """停止扫描"""
    session_manager.stop_session()
    return jsonify({"success": True, "status": "stopped"})

@app.route('/api/groups')
def get_groups():
    """获取照片组"""
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    groups = db.get_photo_groups(limit, offset)
    return jsonify({"groups": groups})

@app.route('/api/groups/<int:group_id>')
def get_group_details(group_id):
    """获取组详细信息"""
    group_info = db.get_group_details(group_id)
    if not group_info:
        return jsonify({"error": "组不存在"}), 404
    
    return jsonify(group_info)

@app.route('/api/photos/action', methods=['POST'])
def photo_action():
    """照片操作（保留/删除）"""
    data = request.json
    photo_path = data.get('path')
    action = data.get('action')  # 'keep' 或 'delete'
    
    if not photo_path or action not in ['keep', 'delete']:
        return jsonify({"error": "参数错误"}), 400
    
    # 查找照片ID
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM processed_photos WHERE file_path = ?",
        (photo_path,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return jsonify({"error": "照片不存在"}), 404
    
    photo_id = row[0]
    
    # 记录用户操作
    db.record_user_action(photo_id, action)
    
    # 如果是保留操作，添加到忽略列表
    if action == 'keep':
        scanner = session_manager.get_scanner()
        if scanner:
            scanner.add_ignored_paths([photo_path])
    
    return jsonify({"success": True, "action": action})

@app.route('/api/delete', methods=['POST'])
def delete_files():
    """删除文件"""
    files = request.json.get('files', [])
    deleted = []
    
    for file_path in files:
        try:
            os.remove(file_path)
            deleted.append(file_path)
            
            # 从数据库中删除记录
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM processed_photos WHERE file_path = ?",
                (file_path,)
            )
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"删除文件失败 {file_path}: {e}")
    
    return jsonify({"deleted": deleted})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """清除历史记录"""
    db.clear_history()
    return jsonify({"success": True})

# ==================== 9. 图片服务路由 ====================

@app.route('/image_thumb')
def serve_thumbnail():
    """提供缩略图"""
    raw_path = request.args.get('path')
    if not raw_path:
        return "缺少路径参数", 400
    
    # 获取文件后缀名
    ext = os.path.splitext(raw_path)[1].lower()
    
    try:
        # 替换这里：原本可能是直接用 Image.open 或 cv2
        # 我们这里用 PIL 安全打开，因为 PIL 原生支持中文路径
        img = Image.open(raw_path)
        img.thumbnail(THUMBNAIL_SIZE)
        
        # 转换并返回
        img = img.convert('RGB')
        img_io = io.BytesIO()
        img.save(img_io, 'JPEG', quality=THUMBNAIL_QUALITY)
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        print(f"生成缩略图失败: {e}")
        return "生成缩略图失败", 500

@app.route('/image_raw')
def serve_image():
    """提供原图"""
    raw_path = request.args.get('path')
    if not raw_path:
        return "缺少路径参数", 400
    
    # 获取文件后缀名
    ext = os.path.splitext(raw_path)[1].lower()
    
    # 如果是 HEIC 格式，进行实时转码
    if ext == '.heic':
        try:
            img = Image.open(raw_path)
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')
            
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG', quality=90)
            img_io.seek(0)
            
            return send_file(img_io, mimetype='image/jpeg')
        except Exception as e:
            print(f"HEIC 预览失败: {e}")
            return "HEIC 转换失败", 500
            
    # 如果是普通图片，直接发送文件
    else:
        return send_from_directory(os.path.dirname(raw_path), os.path.basename(raw_path))

# ==================== 10. 主程序入口 ====================

if __name__ == '__main__':
    print("🚀 AI 照片选片系统启动中...")
    print(f"📁 默认扫描路径: {DEFAULT_WINDOWS_PATH}")
    print(f"📊 数据库路径: {DB_PATH}")
    print(f"⚙️  最大工作线程: {MAX_WORKERS}")
    print(f"🖼️  缩略图大小: {THUMBNAIL_SIZE}")
    print("🌐 服务器启动在 http://0.0.0.0:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
