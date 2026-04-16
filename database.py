"""
数据库模块 - 用于持久化存储扫描结果和用户操作
使用 SQLite 数据库存储已处理照片、相似组和用户选择
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class PhotoDatabase:
    """照片扫描数据库管理类"""
    
    def __init__(self, db_path: str = "scan_history.db"):
        """
        初始化数据库连接
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. 已处理照片表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_hash TEXT NOT NULL,
                file_size INTEGER,
                modified_time REAL,
                feature_vector TEXT,  -- JSON 格式的特征向量
                score REAL,
                ai_type TEXT,  -- 'person' 或 'scenery'
                ai_details TEXT,  -- JSON 格式的详细评分
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_ignored BOOLEAN DEFAULT 0  -- 是否被用户忽略（保留）
            )
        ''')
        
        # 2. 相似照片组表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS photo_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_hash TEXT UNIQUE NOT NULL,  -- 组的唯一标识
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                best_photo_id INTEGER,  -- 最佳照片ID
                FOREIGN KEY (best_photo_id) REFERENCES processed_photos(id)
            )
        ''')
        
        # 3. 组成员关系表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS group_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id INTEGER NOT NULL,
                photo_id INTEGER NOT NULL,
                is_best BOOLEAN DEFAULT 0,  -- 是否为组内最佳照片
                user_selected BOOLEAN DEFAULT 0,  -- 用户是否选择保留
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(group_id, photo_id),
                FOREIGN KEY (group_id) REFERENCES photo_groups(id),
                FOREIGN KEY (photo_id) REFERENCES processed_photos(id)
            )
        ''')
        
        # 4. 用户操作记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                action TEXT NOT NULL,  -- 'keep' 或 'delete'
                action_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (photo_id) REFERENCES processed_photos(id)
            )
        ''')
        
        # 5. 扫描会话表（用于恢复进度）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_path TEXT NOT NULL,
                total_files INTEGER,
                processed_count INTEGER DEFAULT 0,
                window_size INTEGER DEFAULT 20,
                similarity_threshold REAL DEFAULT 0.85,
                status TEXT DEFAULT 'idle',  -- 'idle', 'scanning', 'paused', 'completed'
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引以提高查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_path ON processed_photos(file_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_photos_hash ON processed_photos(file_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_groups_hash ON photo_groups(group_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_members_group ON group_members(group_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_members_photo ON group_members(photo_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_actions_photo ON user_actions(photo_id)')
        
        conn.commit()
        conn.close()
    
    def is_photo_processed(self, file_path: str) -> bool:
        """
        检查照片是否已处理过
        
        Args:
            file_path: 照片文件路径
            
        Returns:
            bool: 是否已处理
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM processed_photos WHERE file_path = ?",
            (file_path,)
        )
        count = cursor.fetchone()[0]
        
        conn.close()
        return count > 0
    
    def get_ignored_photos(self) -> List[str]:
        """
        获取所有被用户忽略（保留）的照片路径
        
        Returns:
            List[str]: 忽略的照片路径列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT file_path FROM processed_photos WHERE is_ignored = 1"
        )
        ignored = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return ignored
    
    def save_photo_result(self, photo_data: Dict[str, Any]) -> int:
        """
        保存照片处理结果
        
        Args:
            photo_data: 照片数据字典，包含：
                - path: 文件路径
                - hash: 哈希值
                - file_size: 文件大小
                - modified_time: 修改时间
                - feature_vector: 特征向量（列表）
                - score: 评分
                - ai_type: AI类型
                - ai_details: AI详细评分
        
        Returns:
            int: 插入的照片ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查是否已存在
        cursor.execute(
            "SELECT id FROM processed_photos WHERE file_path = ?",
            (photo_data['path'],)
        )
        existing = cursor.fetchone()
        
        if existing:
            # 更新现有记录
            cursor.execute('''
                UPDATE processed_photos 
                SET file_hash = ?, file_size = ?, modified_time = ?,
                    feature_vector = ?, score = ?, ai_type = ?, ai_details = ?,
                    processed_date = CURRENT_TIMESTAMP
                WHERE file_path = ?
            ''', (
                photo_data['hash'],
                photo_data.get('file_size'),
                photo_data.get('modified_time'),
                json.dumps(photo_data.get('feature_vector', [])),
                photo_data.get('score', 0),
                photo_data.get('ai_type', 'scenery'),
                json.dumps(photo_data.get('ai_details', {})),
                photo_data['path']
            ))
            photo_id = existing[0]
        else:
            # 插入新记录
            cursor.execute('''
                INSERT INTO processed_photos 
                (file_path, file_hash, file_size, modified_time, 
                 feature_vector, score, ai_type, ai_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                photo_data['path'],
                photo_data['hash'],
                photo_data.get('file_size'),
                photo_data.get('modified_time'),
                json.dumps(photo_data.get('feature_vector', [])),
                photo_data.get('score', 0),
                photo_data.get('ai_type', 'scenery'),
                json.dumps(photo_data.get('ai_details', {}))
            ))
            photo_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return photo_id
    
    def create_photo_group(self, group_hash: str, best_photo_id: Optional[int] = None) -> int:
        """
        创建相似照片组
        
        Args:
            group_hash: 组的唯一哈希标识
            best_photo_id: 最佳照片ID
            
        Returns:
            int: 组ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查是否已存在
        cursor.execute(
            "SELECT id FROM photo_groups WHERE group_hash = ?",
            (group_hash,)
        )
        existing = cursor.fetchone()
        
        if existing:
            group_id = existing[0]
            if best_photo_id:
                cursor.execute(
                    "UPDATE photo_groups SET best_photo_id = ? WHERE id = ?",
                    (best_photo_id, group_id)
                )
        else:
            cursor.execute('''
                INSERT INTO photo_groups (group_hash, best_photo_id)
                VALUES (?, ?)
            ''', (group_hash, best_photo_id))
            group_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return group_id
    
    def add_photo_to_group(self, group_id: int, photo_id: int, is_best: bool = False):
        """
        添加照片到组
        
        Args:
            group_id: 组ID
            photo_id: 照片ID
            is_best: 是否为最佳照片
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO group_members (group_id, photo_id, is_best)
            VALUES (?, ?, ?)
        ''', (group_id, photo_id, 1 if is_best else 0))
        
        conn.commit()
        conn.close()
    
    def record_user_action(self, photo_id: int, action: str):
        """
        记录用户操作
        
        Args:
            photo_id: 照片ID
            action: 操作类型 ('keep' 或 'delete')
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_actions (photo_id, action)
            VALUES (?, ?)
        ''', (photo_id, action))
        
        # 如果是保留操作，标记照片为忽略
        if action == 'keep':
            cursor.execute('''
                UPDATE processed_photos 
                SET is_ignored = 1 
                WHERE id = ?
            ''', (photo_id,))
        
        conn.commit()
        conn.close()
    
    def create_scan_session(self, scan_path: str, total_files: int, 
                           window_size: int = 20, similarity_threshold: float = 0.85) -> int:
        """
        创建扫描会话
        
        Args:
            scan_path: 扫描路径
            total_files: 总文件数
            window_size: 窗口大小
            similarity_threshold: 相似度阈值
            
        Returns:
            int: 会话ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO scan_sessions 
            (scan_path, total_files, processed_count, window_size, 
             similarity_threshold, status, start_time)
            VALUES (?, ?, 0, ?, ?, 'scanning', CURRENT_TIMESTAMP)
        ''', (scan_path, total_files, window_size, similarity_threshold))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id
    
    def update_scan_session(self, session_id: int, processed_count: int, 
                           status: str = 'scanning'):
        """
        更新扫描会话进度
        
        Args:
            session_id: 会话ID
            processed_count: 已处理数量
            status: 状态 ('scanning', 'paused', 'completed')
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE scan_sessions 
            SET processed_count = ?, status = ?
            WHERE id = ?
        ''', (processed_count, status, session_id))
        
        # 如果状态是完成，设置结束时间
        if status == 'completed':
            cursor.execute('''
                UPDATE scan_sessions 
                SET end_time = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_active_session(self) -> Optional[Dict[str, Any]]:
        """
        获取活动的扫描会话
        
        Returns:
            Optional[Dict]: 会话信息字典，如果没有活动会话则返回 None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, scan_path, total_files, processed_count, 
                   window_size, similarity_threshold, status
            FROM scan_sessions 
            WHERE status IN ('scanning', 'paused')
            ORDER BY created_date DESC 
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'scan_path': row[1],
                'total_files': row[2],
                'processed_count': row[3],
                'window_size': row[4],
                'similarity_threshold': row[5],
                'status': row[6]
            }
        return None
    
    def get_photo_groups(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        获取照片组列表
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Dict]: 组信息列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT g.id, g.group_hash, g.created_date, g.best_photo_id,
                   COUNT(gm.photo_id) as member_count
            FROM photo_groups g
            LEFT JOIN group_members gm ON g.id = gm.group_id
            GROUP BY g.id
            ORDER BY g.created_date DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        groups = []
        for row in cursor.fetchall():
            groups.append({
                'id': row[0],
                'group_hash': row[1],
                'created_date': row[2],
                'best_photo_id': row[3],
                'member_count': row[4]
            })
        
        conn.close()
        return groups
    
    def get_group_details(self, group_id: int) -> Dict[str, Any]:
        """
        获取组的详细信息
        
        Args:
            group_id: 组ID
            
        Returns:
            Dict: 组详细信息
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取组基本信息
        cursor.execute('''
            SELECT g.group_hash, g.created_date, g.best_photo_id,
                   p.file_path as best_photo_path
            FROM photo_groups g
            LEFT JOIN processed_photos p ON g.best_photo_id = p.id
            WHERE g.id = ?
        ''', (group_id,))
        
        group_info = cursor.fetchone()
        if not group_info:
            conn.close()
            return {}
        
        # 获取组成员
        cursor.execute('''
            SELECT p.id, p.file_path, p.score, p.ai_type, p.ai_details,
                   gm.is_best, gm.user_selected
            FROM group_members gm
            JOIN processed_photos p ON gm.photo_id = p.id
            WHERE gm.group_id = ?
            ORDER BY p.score DESC
        ''', (group_id,))
        
        members = []
        for row in cursor.fetchall():
            members.append({
                'id': row[0],
                'file_path': row[1],
                'score': row[2],
                'ai_type': row[3],
                'ai_details': json.loads(row[4]) if row[4] else {},
                'is_best': bool(row[5]),
                'user_selected': bool(row[6])
            })
        
        conn.close()
        
        return {
            'group_hash': group_info[0],
            'created_date': group_info[1],
            'best_photo_id': group_info[2],
            'best_photo_path': group_info[3],
            'members': members,
            'member_count': len(members)
        }
    
    def clear_history(self):
        """清除所有历史记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tables = ['user_actions', 'group_members', 'photo_groups', 
                  'processed_photos', 'scan_sessions']
        
        for table in tables:
            cursor.execute(f'DELETE FROM {table}')
        
        conn.commit()
        conn.close()
    
    def close(self):
        """关闭数据库连接（SQLite 会自动管理）"""
        pass

    def clear_history_by_path(self, scan_path: str):
        """清除指定路径下的所有处理记录和分组"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 先删除组成员（通过子查询关联）
        cursor.execute('''
            DELETE FROM group_members 
            WHERE photo_id IN (
                SELECT id FROM processed_photos 
                WHERE file_path LIKE ?
            )
        ''', (scan_path + '%',))
        
        # 删除该路径下的照片记录
        cursor.execute('''
            DELETE FROM processed_photos 
            WHERE file_path LIKE ?
        ''', (scan_path + '%',))
        
        # 删除没有任何成员的空组（可选）
        cursor.execute('''
            DELETE FROM photo_groups 
            WHERE id NOT IN (
                SELECT DISTINCT group_id FROM group_members
            )
        ''')
        
        conn.commit()
        conn.close()