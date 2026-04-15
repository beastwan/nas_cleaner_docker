# NAS 照片选片系统 - 滑动窗口优化版

## 概述

本系统是基于滑动窗口算法的 AI 照片选片系统，专门为 Synology NAS Docker 环境优化。系统通过滑动窗口算法解决了连续相似照片被分到不同组的问题，提高了相似照片分组的准确性。

## 主要改进

### 1. 滑动窗口算法 (SlidingWindowScanner)
- **窗口机制**: 维护一个固定大小的窗口（默认20张照片）
- **相似度比较**: 新照片与窗口内所有照片比较相似度
- **智能分组**: 找到相似照片时形成新组或加入现有组
- **窗口维护**: 窗口满时移除最旧的照片
- **解决连续相似问题**: 避免连续相似照片被分到不同组

### 2. 数据库持久化 (PhotoDatabase)
- **SQLite 数据库**: 存储扫描结果和用户操作
- **会话管理**: 支持暂停/恢复扫描进度
- **历史记录**: 记录用户选择和操作历史
- **状态恢复**: 系统重启后可恢复之前的扫描状态

### 3. 改进的评分算法
- **人物照片评分**: 清晰度、表情、眼睛状态、构图、光照、人脸大小、色彩和谐度
- **风景照片评分**: 构图、色彩和谐度、光影层次、清晰度、视觉复杂度平衡
- **AI 类型识别**: 自动识别人物照和风景照

### 4. 会话管理 (ScanSessionManager)
- **会话状态**: 支持 idle、scanning、paused、completed 状态
- **进度跟踪**: 实时显示扫描进度
- **参数配置**: 可配置窗口大小、相似度阈值等参数

## 文件结构

```
nas_cleaner_docker/
├── app.py                    # 主应用文件（已更新）
├── database.py              # 数据库模块（新增）
├── sliding_window.py        # 滑动窗口算法（新增）
├── templates/
│   └── index.html          # 前端界面（已更新）
├── requirements.txt         # 依赖包列表
├── Dockerfile              # Docker 构建文件
├── docker-compose.yml      # Docker Compose 配置
├── start_app.bat           # Windows 启动脚本（新增）
├── test_sliding_window.py  # 测试脚本（新增）
└── README_SLIDING_WINDOW.md # 本文档
```

## 快速开始

### 方法1: 使用启动脚本（Windows）
```bash
cd nas_cleaner_docker
start_app.bat
```

### 方法2: 手动启动
```bash
cd nas_cleaner_docker
pip install -r requirements.txt
python app.py
```

### 方法3: Docker 启动
```bash
cd nas_cleaner_docker
docker-compose up -d
```

## 使用说明

### 1. 配置扫描参数
- **扫描路径**: 设置要扫描的照片目录（默认: /homes）
- **窗口大小**: 滑动窗口的大小（默认: 20张）
- **相似度阈值**: 照片相似度阈值（默认: 0.85）
- **批次大小**: 每次处理的照片数量（默认: 20张）

### 2. 开始扫描
1. 点击"开始扫描"按钮
2. 系统会自动扫描照片并分组
3. 实时显示扫描进度和窗口状态

### 3. 照片管理
- **查看分组**: 相似照片会自动分组显示
- **选择照片**: 点击照片选择要删除的照片
- **保留照片**: 双击照片预览，点击"保留此照片"
- **删除照片**: 选择照片后点击"删除选中"

### 4. 会话控制
- **暂停扫描**: 点击"暂停"按钮暂停当前扫描
- **恢复扫描**: 点击"恢复"按钮继续扫描
- **停止扫描**: 点击"停止"按钮结束扫描会话

## 技术细节

### 滑动窗口算法原理
```
算法流程:
1. 初始化一个固定大小的窗口
2. 对于每张新照片:
   a. 与全局分组中的最佳照片比较相似度
   b. 如果相似，加入该组
   c. 否则，与窗口内照片比较相似度
   d. 如果相似，形成新组，从窗口移除已分组的照片
   e. 否则，将照片加入窗口
3. 如果窗口已满，移除最旧的照片
```

### 数据库设计
```sql
-- 主要表结构:
1. processed_photos      # 已处理照片
2. photo_groups          # 相似照片组
3. group_members         # 组成员关系
4. user_actions          # 用户操作记录
5. scan_sessions         # 扫描会话
```

### API 接口
```
GET  /                    # 主页
GET  /api/status          # 获取状态
POST /api/scan/init       # 初始化扫描
GET  /api/scan/batch      # 获取批次结果
POST /api/scan/pause      # 暂停扫描
POST /api/scan/resume     # 恢复扫描
POST /api/scan/stop       # 停止扫描
GET  /api/groups          # 获取照片组
POST /api/photos/action   # 照片操作
POST /api/delete          # 删除文件
GET  /image_thumb         # 获取缩略图
GET  /image_raw           # 获取原图
```

## 测试验证

运行测试脚本验证功能:
```bash
python test_sliding_window.py
```

预期输出:
```
[START] 开始测试 nas_cleaner_docker 滑动窗口优化版...
[TEST] 开始测试滑动窗口算法...
[OK] 形成 2 个相似组
[INFO] 窗口信息: 0/5 张照片
[INFO] 分组统计: 2 个组，共 10 张照片
[OK] 滑动窗口算法测试完成！
[OK] 数据库功能测试完成！
[SUCCESS] 所有测试完成！系统已成功集成滑动窗口算法。
```

## 部署到 Synology NAS

### 1. 准备 Docker 镜像
```bash
docker build -t nas-photo-cleaner .
```

### 2. 配置 Docker Compose
```yaml
version: '3'
services:
  photo-cleaner:
    image: nas-photo-cleaner
    container_name: nas-photo-cleaner
    ports:
      - "5000:5000"
    volumes:
      - /volume1/homes:/homes:ro  # 挂载 NAS 照片目录
      - ./data:/app/data          # 挂载数据目录
    environment:
      - NAS_PHOTO_DIR=/homes      # 设置扫描路径
    restart: unless-stopped
```

### 3. 启动服务
```bash
docker-compose up -d
```

## 故障排除

### 常见问题
1. **依赖安装失败**: 确保使用 Python 3.8+，尝试使用国内镜像源
2. **模型加载失败**: 检查网络连接，确保能下载 PyTorch 模型
3. **权限问题**: 确保 Docker 容器有读取照片目录的权限
4. **内存不足**: 减少批次大小或窗口大小

### 日志查看
```bash
# 查看应用日志
docker logs nas-photo-cleaner

# 查看实时日志
docker logs -f nas-photo-cleaner
```

## 性能优化建议

1. **调整窗口大小**: 根据照片数量调整窗口大小（建议 10-30）
2. **调整相似度阈值**: 根据需求调整相似度阈值（0.8-0.9）
3. **调整批次大小**: 根据系统性能调整批次大小（10-50）
4. **使用 GPU**: 如果 NAS 支持 GPU，可启用 GPU 加速
5. **定期清理数据库**: 定期清理历史记录释放空间

## 版本历史

### v2.0 (当前版本)
- 集成滑动窗口算法
- 添加数据库持久化
- 改进评分算法
- 添加会话管理
- 优化前端界面

### v1.0 (原始版本)
- 基础照片扫描功能
- 简单的相似度比较
- 基础评分算法

## 联系方式

如有问题或建议，请参考原始项目文档或联系开发者。

---
**注意**: 本系统为优化版本，保留了原始系统的所有功能，并添加了滑动窗口算法优化。