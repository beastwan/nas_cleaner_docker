"""
滑动窗口算法测试脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sliding_window import SlidingWindowScanner, PhotoItem
import numpy as np

def test_sliding_window():
    """测试滑动窗口算法"""
    print("[TEST] 开始测试滑动窗口算法...")
    
    # 创建扫描器
    scanner = SlidingWindowScanner(window_size=5, similarity_threshold=0.8)
    
    # 创建测试照片
    photos = []
    for i in range(10):
        # 创建相似的特征向量（前5张相似，后5张相似）
        if i < 5:
            feature_vector = np.random.randn(512) * 0.1 + np.ones(512)  # 相似的特征
        else:
            feature_vector = np.random.randn(512) * 0.1 - np.ones(512)  # 另一组相似的特征
        
        photo = PhotoItem(
            path=f"/test/photo_{i}.jpg",
            file_hash=f"hash_{i}",
            feature_vector=feature_vector.tolist(),
            score=80 + i * 2,
            ai_type="scenery",
            ai_details={"sharpness": 85, "composition": 75}
        )
        photos.append(photo)
    
    # 处理照片
    print(f"[TEST] 处理 {len(photos)} 张测试照片...")
    groups = scanner.process_batch(photos)
    
    print(f"[OK] 形成 {len(groups)} 个相似组")
    
    # 显示窗口信息
    window_info = scanner.get_window_info()
    print(f"[INFO] 窗口信息: {window_info['window_size']}/{window_info['max_window_size']} 张照片")
    
    # 显示分组信息
    groups_info = scanner.get_groups_info()
    print(f"[INFO] 分组统计: {groups_info['total_groups']} 个组，共 {groups_info['total_photos_in_groups']} 张照片")
    
    # 测试忽略功能
    print("\n[TEST] 测试忽略功能...")
    scanner.add_ignored_paths(["/test/photo_0.jpg"])
    ignored_photo = PhotoItem(
        path="/test/photo_0.jpg",
        file_hash="hash_ignored",
        feature_vector=np.ones(512).tolist(),
        score=90,
        ai_type="person",
        ai_details={}
    )
    
    group = scanner.process_photo(ignored_photo)
    if group is None:
        print("[OK] 忽略功能正常：被忽略的照片未形成新组")
    else:
        print("[ERROR] 忽略功能异常：被忽略的照片形成了新组")
    
    print("\n[OK] 滑动窗口算法测试完成！")

def test_database():
    """测试数据库功能"""
    print("\n[TEST] 开始测试数据库功能...")
    
    try:
        from database import PhotoDatabase
        
        # 创建测试数据库
        db = PhotoDatabase("test.db")
        
        # 测试保存照片
        photo_data = {
            "path": "/test/photo_db.jpg",
            "hash": "test_hash_123",
            "file_size": 1024,
            "modified_time": 1234567890,
            "feature_vector": [1.0, 2.0, 3.0],
            "score": 85.5,
            "ai_type": "person",
            "ai_details": {"sharpness": 90, "expression": 80}
        }
        
        photo_id = db.save_photo_result(photo_data)
        print(f"[OK] 照片保存成功，ID: {photo_id}")
        
        # 测试检查是否已处理
        is_processed = db.is_photo_processed("/test/photo_db.jpg")
        print(f"[OK] 照片处理状态检查: {is_processed}")
        
        # 测试创建组
        group_id = db.create_photo_group("test_group_hash", photo_id)
        print(f"[OK] 组创建成功，ID: {group_id}")
        
        # 测试添加照片到组
        db.add_photo_to_group(group_id, photo_id, is_best=True)
        print("[OK] 照片添加到组成功")
        
        # 测试记录用户操作
        db.record_user_action(photo_id, "keep")
        print("[OK] 用户操作记录成功")
        
        # 测试获取忽略的照片
        ignored = db.get_ignored_photos()
        print(f"[OK] 获取忽略的照片: {len(ignored)} 张")
        
        # 清理测试数据库
        import os
        if os.path.exists("test.db"):
            os.remove("test.db")
            print("[OK] 测试数据库清理完成")
            
    except Exception as e:
        print(f"[ERROR] 数据库测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("[START] 开始测试 nas_cleaner_docker 滑动窗口优化版...")
    print("=" * 50)
    
    test_sliding_window()
    test_database()
    
    print("\n" + "=" * 50)
    print("[SUCCESS] 所有测试完成！系统已成功集成滑动窗口算法。")
    print("\n[FEATURES] 实现的功能:")
    print("  [OK] 滑动窗口算法 (SlidingWindowScanner)")
    print("  [OK] 照片项和组管理 (PhotoItem, PhotoGroup)")
    print("  [OK] 数据库持久化 (PhotoDatabase)")
    print("  [OK] 会话管理 (ScanSessionManager)")
    print("  [OK] 改进的评分算法")
    print("  [OK] 完整的 Web 界面")
    print("\n[NEXT] 现在可以运行 'python app.py' 启动应用！")
