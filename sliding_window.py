"""
滑动窗口算法模块 - 参考 Android 应用的滑动窗口算法实现
用于优化相似照片分组，解决连续相似照片被分到不同组的问题
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json


class PhotoItem:
    """照片项类，封装照片的元数据和特征"""
    
    def __init__(self, path: str, file_hash: str, feature_vector: List[float], 
                 score: float, ai_type: str = "scenery", ai_details: Dict = None,
                 file_size: int = 0, modified_time: float = 0):
        """
        初始化照片项
        
        Args:
            path: 文件路径
            file_hash: 文件哈希值
            feature_vector: 特征向量
            score: 评分
            ai_type: AI类型 ('person' 或 'scenery')
            ai_details: AI详细评分
            file_size: 文件大小
            modified_time: 修改时间
        """
        self.path = path
        self.file_hash = file_hash
        self.feature_vector = np.array(feature_vector, dtype=np.float32)
        self.score = score
        self.ai_type = ai_type
        self.ai_details = ai_details or {}
        self.file_size = file_size
        self.modified_time = modified_time
        self.is_best = False
        self.user_selected = False
        self.group_id = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'path': self.path,
            'hash': self.file_hash,
            'feature_vector': self.feature_vector.tolist(),
            'score': self.score,
            'ai_type': self.ai_type,
            'ai_details': self.ai_details,
            'file_size': self.file_size,
            'modified_time': self.modified_time,
            'is_best': self.is_best,
            'user_selected': self.user_selected,
            'group_id': self.group_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhotoItem':
        """从字典创建 PhotoItem 实例"""
        item = cls(
            path=data['path'],
            file_hash=data['hash'],
            feature_vector=data['feature_vector'],
            score=data['score'],
            ai_type=data.get('ai_type', 'scenery'),
            ai_details=data.get('ai_details', {}),
            file_size=data.get('file_size', 0),
            modified_time=data.get('modified_time', 0)
        )
        item.is_best = data.get('is_best', False)
        item.user_selected = data.get('user_selected', False)
        item.group_id = data.get('group_id')
        return item


class PhotoGroup:
    """照片组类，包含相似的照片"""
    
    def __init__(self, photos: List[PhotoItem] = None):
        """
        初始化照片组
        
        Args:
            photos: 照片项列表
        """
        self.photos = photos or []
        self.group_hash = self._generate_group_hash()
        self.best_photo = None
        self._update_best_photo()
    
    def _generate_group_hash(self) -> str:
        """生成组的唯一哈希标识"""
        if not self.photos:
            return hashlib.md5(b"empty").hexdigest()
        
        # 使用所有照片的哈希值生成组哈希
        hash_str = "".join(sorted([p.file_hash for p in self.photos]))
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def _update_best_photo(self):
        """更新最佳照片"""
        if self.photos:
            self.best_photo = max(self.photos, key=lambda p: p.score)
            for photo in self.photos:
                photo.is_best = (photo.path == self.best_photo.path)
    
    def add_photo(self, photo: PhotoItem):
        """添加照片到组"""
        self.photos.append(photo)
        self._update_best_photo()
        # 重新生成组哈希
        self.group_hash = self._generate_group_hash()
    
    def remove_photo(self, photo_path: str) -> bool:
        """从组中移除照片"""
        for i, photo in enumerate(self.photos):
            if photo.path == photo_path:
                self.photos.pop(i)
                self._update_best_photo()
                self.group_hash = self._generate_group_hash()
                return True
        return False
    
    def get_size(self) -> int:
        """获取组大小"""
        return len(self.photos)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'group_hash': self.group_hash,
            'photos': [p.to_dict() for p in self.photos],
            'best_photo': self.best_photo.to_dict() if self.best_photo else None,
            'size': len(self.photos)
        }


class SlidingWindowScanner:
    """
    滑动窗口扫描器 - 核心算法实现
    
    算法原理：
    1. 维护一个固定大小的窗口（默认20张照片）
    2. 新照片与窗口内所有照片比较相似度
    3. 如果找到相似照片，形成新组或加入现有组
    4. 窗口满时移除最旧的照片
    5. 解决连续相似照片被分到不同组的问题
    """
    
    def __init__(self, window_size: int = 20, similarity_threshold: float = 0.85):
        """
        初始化滑动窗口扫描器
        
        Args:
            window_size: 滑动窗口大小
            similarity_threshold: 相似度阈值
        """
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        
        # 滑动窗口数据结构
        self.window: List[PhotoItem] = []  # 窗口内的照片项
        self.window_map: Dict[str, PhotoItem] = {}  # 快速查找映射
        
        # 全局分组
        self.global_groups: List[PhotoGroup] = []
        self.group_map: Dict[str, PhotoGroup] = {}  # 组哈希到组的映射
        
        # 忽略的照片（用户已保留）
        self.ignored_paths: set = set()
    
    def set_ignored_paths(self, paths: List[str]):
        """设置忽略的照片路径"""
        self.ignored_paths = set(paths)
    
    def add_ignored_paths(self, paths: List[str]):
        """添加忽略的照片路径"""
        self.ignored_paths.update(paths)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            float: 相似度值 (0-1)
        """
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def find_similar_in_window(self, photo: PhotoItem) -> Tuple[Optional[PhotoItem], float]:
        """
        在窗口内查找相似照片
        
        Args:
            photo: 要查找的照片
            
        Returns:
            Tuple[Optional[PhotoItem], float]: (相似照片, 相似度)
        """
        best_match = None
        best_similarity = 0.0
        
        for window_photo in self.window:
            # 跳过同一张照片
            if window_photo.path == photo.path:
                continue
            
            similarity = self.cosine_similarity(photo.feature_vector, window_photo.feature_vector)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = window_photo
        
        return best_match, best_similarity
    
    def find_similar_in_groups(self, photo: PhotoItem) -> Tuple[Optional[PhotoGroup], float]:
        """
        在全局分组中查找相似组
        
        Args:
            photo: 要查找的照片
            
        Returns:
            Tuple[Optional[PhotoGroup], float]: (相似组, 相似度)
        """
        best_group = None
        best_similarity = 0.0
        
        for group in self.global_groups:
            if not group.best_photo:
                continue
            
            similarity = self.cosine_similarity(photo.feature_vector, group.best_photo.feature_vector)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_group = group
        
        return best_group, best_similarity
    
    def add_photo_to_window(self, photo: PhotoItem):
        """
        添加照片到滑动窗口
        
        Args:
            photo: 要添加的照片
        """
        # 添加到窗口
        self.window.append(photo)
        self.window_map[photo.path] = photo
        
        # 维护窗口大小
        if len(self.window) > self.window_size:
            removed_photo = self.window.pop(0)
            del self.window_map[removed_photo.path]
    
    def process_photo(self, photo: PhotoItem) -> Optional[PhotoGroup]:
        """
        处理单张照片，返回新形成的组（如果有）
        
        Args:
            photo: 要处理的照片
            
        Returns:
            Optional[PhotoGroup]: 新形成的组，如果没有则返回 None
        """
        # 检查是否被忽略
        if photo.path in self.ignored_paths:
            return None
        
        new_group = None
        
        # 第一步：与全局分组比较
        similar_group, group_similarity = self.find_similar_in_groups(photo)
        
        if similar_group:
            # 找到相似组，加入该组
            similar_group.add_photo(photo)
            photo.group_id = similar_group.group_hash
            self.group_map[similar_group.group_hash] = similar_group
        else:
            # 第二步：与窗口内照片比较
            similar_photo, window_similarity = self.find_similar_in_window(photo)
            
            if similar_photo:
                # 找到相似照片，形成新组
                new_group = PhotoGroup([similar_photo, photo])
                
                # 设置组ID
                similar_photo.group_id = new_group.group_hash
                photo.group_id = new_group.group_hash
                
                # 添加到全局分组
                self.global_groups.append(new_group)
                self.group_map[new_group.group_hash] = new_group
                
                # 从窗口中移除已分组的照片
                if similar_photo.path in self.window_map:
                    self.window = [p for p in self.window if p.path != similar_photo.path]
                    del self.window_map[similar_photo.path]
        
        # 第三步：将照片加入滑动窗口（如果未分组或分组后仍需保留在窗口）
        if not similar_group and (not similar_photo or new_group is None):
            self.add_photo_to_window(photo)
        
        return new_group
    
    def process_batch(self, photos: List[PhotoItem]) -> List[PhotoGroup]:
        """
        处理一批照片
        
        Args:
            photos: 照片项列表
            
        Returns:
            List[PhotoGroup]: 新形成的组列表
        """
        new_groups = []
        
        for photo in photos:
            new_group = self.process_photo(photo)
            if new_group:
                new_groups.append(new_group)
        
        return new_groups
    
    def get_window_info(self) -> Dict[str, Any]:
        """获取窗口信息"""
        return {
            'window_size': len(self.window),
            'max_window_size': self.window_size,
            'photo_paths': [p.path for p in self.window],
            'similarity_threshold': self.similarity_threshold
        }
    
    def get_groups_info(self) -> Dict[str, Any]:
        """获取分组信息"""
        return {
            'total_groups': len(self.global_groups),
            'total_photos_in_groups': sum(g.get_size() for g in self.global_groups),
            'groups': [g.to_dict() for g in self.global_groups]
        }
    
    def clear(self):
        """清除所有数据"""
        self.window.clear()
        self.window_map.clear()
        self.global_groups.clear()
        self.group_map.clear()
        self.ignored_paths.clear()
    
    def save_state(self, file_path: str):
        """保存状态到文件"""
        state = {
            'window_size': self.window_size,
            'similarity_threshold': self.similarity_threshold,
            'window': [p.to_dict() for p in self.window],
            'global_groups': [g.to_dict() for g in self.global_groups],
            'ignored_paths': list(self.ignored_paths)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_state(self, file_path: str):
        """从文件加载状态"""
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.window_size = state.get('window_size', 20)
        self.similarity_threshold = state.get('similarity_threshold', 0.85)
        
        # 加载窗口
        self.window.clear()
        self.window_map.clear()
        for photo_dict in state.get('window', []):
            photo = PhotoItem.from_dict(photo_dict)
            self.window.append(photo)
            self.window_map[photo.path] = photo
        
        # 加载全局分组
        self.global_groups.clear()
        self.group_map.clear()
        for group_dict in state.get('global_groups', []):
            photos = [PhotoItem.from_dict(p) for p in group_dict.get('photos', [])]
            group = PhotoGroup(photos)
            self.global_groups.append(group)
            self.group_map[group.group_hash] = group
        
        # 加载忽略路径
        self.ignored_paths = set(state.get('ignored_paths', []))