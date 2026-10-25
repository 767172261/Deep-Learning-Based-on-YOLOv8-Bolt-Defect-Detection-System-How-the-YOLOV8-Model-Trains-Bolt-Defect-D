深度学习基于YOLOv8螺栓缺陷检测系统 YOLOV8模型 如何训练螺栓缺陷检测数据集 通过权重推理识别'鼓包','爆裂','圆形缺陷','裂纹','损伤','褶皱','划痕','工具痕迹','空洞'

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a75e60728834608949012bb9bb899df.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e894f9104e140448a6559e827a85a89.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1721e9d8bcd64df08d4dcd15adb7860e.png)
 **YOLOv8** 的 **螺栓缺陷检测系统** 的完整构建代码与项目结构说明，涵盖 **PyQt5 图形界面、模型加载、图像/视频/摄像头实时检测、批量处理、结果保存** 等功能。

> ✅ 本系统适用于工业质检场景，支持 10 类螺栓缺陷检测：
> - `鼓包`, `爆裂`, `圆形缺陷`, `裂纹`, `损伤`, `褶皱`, `划痕`, `工具痕迹`, `空洞`, `编码错误`

---

## 📁 一、项目目录结构【博主建议】

```
Bolt_Defect_Detection_System/
├── main.py                      # 主程序入口
├── ui_mainwindow.py             # Qt Designer 生成的 UI 文件（Python）
├── bolt_detection.py            # 核心检测逻辑
├── resources.qrc                # 图标资源文件
├── icons/                       # 图标图片
│   ├── open_file.png
│   ├── open_video.png
│   └── ...
├── models/                      # 训练好的 YOLOv8 模型
│   └── best.pt                    # 最佳权重文件（由训练得到）
├── dataset/                     # 数据集图片（可选）
│   ├── train/
│   └── val/
├── results/                     # 检测结果输出目录
└── requirements.txt             # Python 依赖包
```

---

## 🧩 二、requirements.txt

```txt
python==3.9
torch==1.9.0+cu111
torchvision==0.10.0+cu111
opencv-python==4.5.5
PyQt5==5.15.6
ultralytics==8.0.0
numpy==1.21.0
Pillow==9.0.1
```

> 安装命令：
```bash
pip install -r requirements.txt
```

---

## 🔧 三、核心检测模块：`bolt_detection.py`

```python
# bolt_detection.py
import cv2
import numpy as np
from ultralytics import YOLO
import os

class BoltDefectDetector:
    def __init__(self, model_path='models/best.pt', device='cpu'):
        self.model = YOLO(model_path)
        self.device = device
        self.names = ['鼓包', '爆裂', '圆形缺陷', '裂纹', '损伤', '褶皱', '划痕', '工具痕迹', '空洞', '编码错误']

    def detect_image(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """检测单张图片"""
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
        return results[0]  # 返回第一个结果

    def detect_video(self, video_path, conf_threshold=0.25, iou_threshold=0.45):
        """检测视频文件"""
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=conf_threshold, iou=iou_threshold)
            annotated_frame = results[0].plot()
            yield annotated_frame
        cap.release()

    def detect_camera(self, conf_threshold=0.25, iou_threshold=0.45):
        """调用摄像头实时检测"""
        cap = cv2.VideoCapture(0)  # 使用默认摄像头
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=conf_threshold, iou=iou_threshold)
            annotated_frame = results[0].plot()
            yield annotated_frame
        cap.release()

    def batch_detect(self, folder_path, conf_threshold=0.25, iou_threshold=0.45):
        """批量检测文件夹内所有图片"""
        results_list = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                result = self.detect_image(img_path, conf_threshold, iou_threshold)
                info = {
                    'filename': filename,
                    'path': img_path,
                    'boxes': result.boxes.xyxy.cpu().numpy(),
                    'classes': result.boxes.cls.cpu().numpy(),
                    'confidences': result.boxes.conf.cpu().numpy(),
                    'names': [self.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]
                }
                results_list.append(info)
        return results_list
```

---

## 🖼️ 四、主程序界面：`main.py`

```python
# main.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QListWidget, QGroupBox,
    QLineEdit, QComboBox, QCheckBox, QSpinBox, QSlider
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.uic import loadUi
import cv2
import numpy as np
from bolt_detection import BoltDefectDetector
from PIL import Image

class BoltDefectSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.current_image = None
        self.results = None
        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("基于深度学习的螺栓缺陷检测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 加载UI文件（使用 Qt Designer 创建）
        loadUi('ui_mainwindow.py', self)  # 假设已将 .ui 转换为 .py

        # 连接按钮事件
        self.btn_open_image.clicked.connect(self.open_image)
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_open_camera.clicked.connect(self.open_camera)
        self.btn_save.clicked.connect(self.save_results)
        self.btn_exit.clicked.connect(self.close)

        # 设置参数控件
        self.conf_slider.valueChanged.connect(self.update_conf)
        self.iou_slider.valueChanged.connect(self.update_iou)
        self.show_labels.setChecked(True)
        self.show_original.setChecked(False)

        # 初始化定时器用于摄像头
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)

    def load_model(self):
        try:
            self.detector = BoltDefectDetector(model_path='models/best.pt')
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            self.current_image = file_path
            self.display_image(file_path)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.results = self.detector.batch_detect(folder_path)
            self.display_batch_results()

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.video_running = True
            self.process_video()

    def process_video(self):
        while self.video_running:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            results = self.detector.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            annotated_frame = results[0].plot()
            self.display_image_from_array(annotated_frame)
        self.video_capture.release()

    def open_camera(self):
        self.camera_running = True
        self.timer.start(30)  # 30ms 更新一次

    def update_camera_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.detector.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            annotated_frame = results[0].plot()
            self.display_image_from_array(annotated_frame)

    def display_image(self, path):
        pixmap = QPixmap(path)
        self.image_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def display_image_from_array(self, array):
        h, w = array.shape[:2]
        bytes_per_line = 3 * w
        qimage = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def display_batch_results(self):
        self.result_list.clear()
        for res in self.results:
            item = f"{res['filename']} | {len(res['names'])}个缺陷"
            self.result_list.addItem(item)

    def save_results(self):
        if self.results:
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)
            for res in self.results:
                img = cv2.imread(res['path'])
                for box, cls_name, conf in zip(res['boxes'], res['names'], res['confidences']):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.imwrite(os.path.join(output_dir, res['filename']), img)
            print(f"✅ 批量检测结果已保存至: {output_dir}")

    def update_conf(self):
        self.conf_threshold = self.conf_slider.value() / 100.0

    def update_iou(self):
        self.iou_threshold = self.iou_slider.value() / 100.0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BoltDefectSystem()
    window.show()
    sys.exit(app.exec_())
```

---

## 🎨 五、UI 设计说明（`.ui` 文件）

使用 **Qt Designer** 创建 `mainwindow.ui`，包含以下组件：

### 左侧区域：
- `QLabel` 显示图像（500×500）
- 上下两个按钮：缩放/移动（可选）

### 右侧区域：
#### 参数设置区：
- `QComboBox`: 选择模型（如 best.pt）
- `QSlider`: 置信度阈值（0~100）
- `QSlider`: IOU 阈值（0~100）
- `QCheckBox`: 显示标签/原图
- `QComboBox`: 检测设备（CPU/GPU）

#### 检测结果显示区：
- `QListWidget`: 列出每张图片的检测结果（文件名 + 缺陷数）

#### 操作按钮区：
- 打开图片、打开文件夹、打开视频、打开摄像头
- 保存、退出

---

## 🚀 六、运行步骤

1. **准备环境**
```bash
conda create -n bolt_env python=3.9
conda activate bolt_env
pip install -r requirements.txt
```

2. **准备数据**
- 将训练好的 `best.pt` 放入 `models/`
- 将测试图像放入 `dataset/test/`

3. **运行程序**
```bash
python main.py
```

---

## ✅ 七、功能亮点总结

| 功能 | 描述 |
|------|------|
| ✅ 多源输入 | 支持图片、视频、摄像头 |
| ✅ 实时显示 | 检测结果实时标注在图像上 |
| ✅ 批量处理 | 自动检测整个文件夹 |
| ✅ 可视化结果 | 界面展示目标位置、类别、置信度 |
| ✅ 结果保存 | 支持保存带框标注的图像 |
| ✅ 参数调节 | 可自定义置信度、IoU 阈值 |

---

## 💡 八、后续优化建议

- ✅ 使用 `yolov8s.pt` 或 `yolov8m.pt` 提升精度
- ✅ 添加 `TensorRT` 加速推理
- ✅ 部署为 Web 应用（Flask + React）
- ✅ 添加报警功能（当发现“爆裂”类缺陷时弹窗提示）

---

是否需要我提供：
- 📄 `.ui` 文件转换为 `.py` 的脚本？
- 🖼️ 示例检测结果图？
- 📦 完整打包发布方式（exe）？

欢迎继续提问！
