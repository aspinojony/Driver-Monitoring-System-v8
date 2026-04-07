<div align="center">

# 🚗 Advanced Driver Monitoring System (DMS) v8

### 基于深度学习与空间结构特征的车载多模态感知监控控制台

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](#)
[![PyQt6](https://img.shields.io/badge/PyQt6-UI%20Framework-green?style=for-the-badge&logo=qt)](#)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Deep%20Learning-orange?style=for-the-badge)](#)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-red?style=for-the-badge&logo=google)](#)
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)](#)

_面向毕业设计答辩与原型演示的驾驶员状态监测系统，结合行为识别与疲劳检测进行实时风险提示。_

</div>

---

## 🌟 功能概览

- **疲劳检测**：基于 MediaPipe Face Mesh 计算 EAR / MAR / PERCLOS，识别闭眼、打哈欠、疲劳状态。
- **行为识别**：基于 YOLOv8 分类模型识别分心驾驶行为，如使用手机、喝水、与乘客交谈等。
- **暗光增强**：支持基于 CLAHE 的低照度增强处理，缓解暗环境下的人脸特征丢失问题。
- **双端展示**：提供 **PyQt6 桌面端** 与 **Flask Web 控制台** 两种运行入口。
- **会话记录**：支持事件记录与报告导出，便于展示与论文材料整理。

---

## ⚙️ 项目结构

```text
core/         核心算法与推理逻辑
ui/           PyQt6 桌面端界面
templates/    Web 版页面模板
scripts/      数据增强 / 训练辅助脚本
runs/         分类训练输出（含 best.pt / last.pt）
data/weights/ 备用/历史模型权重
main.py       桌面端入口
web_app.py    Web 演示入口
```

---

## 🛠️ 本地运行环境

## 推荐版本

- **Python 3.11**
- macOS / Windows 均可，作者本地开发环境为 macOS

> 说明：项目在较新的 Python 版本上可能遇到依赖兼容问题，建议优先使用 Python 3.11。

### 1. 创建虚拟环境并安装依赖

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🤖 模型权重说明

GitHub 仓库默认**不保证包含全部训练权重**。项目运行时主要用到以下文件：

### 行为分类主模型

```text
runs/classify/domain_adapted_cls_final/weights/best.pt
```

### 姿态模型（可选/备用）

```text
data/weights/yolov8n-pose.pt
```

如果 `best.pt` 缺失，当前代码会自动回退到公开基础模型：

```text
yolov8n-cls.pt
```

这样可以保证项目在缺少私有训练权重时也能先启动并演示。

---

## 🚀 启动方式

### 方案 A：启动桌面端（PyQt6）

```bash
python main.py
```

### 方案 B：启动 Web 演示版（推荐答辩展示）

```bash
python web_app.py
```

默认访问地址：

- <http://127.0.0.1:5050>

---

## 📷 摄像头权限说明（macOS）

首次运行时，如果无法打开摄像头，请在：

- **系统设置 → 隐私与安全性 → 摄像头**

中为当前终端（如 Terminal / iTerm）开启摄像头权限。

---

## 🎥 核心交互模块说明

| 模块名称 | 技术实现 | 业务用途 |
| :-- | :-- | :-- |
| 实时推流控制台 | PyQt6 / WebCam / RTSP | 兼容车载摄像头与本地测试摄像头 |
| 自适应感光增强 | OpenCV CLAHE | 提升暗光场景下的人脸可见性 |
| 疲劳检测 | MediaPipe Face Mesh | 计算 EAR / MAR / PERCLOS |
| 行为识别 | YOLOv8 Classification | 识别分心与异常驾驶行为 |
| 日志导出 | Session Logger | 输出会话事件记录与报告 |

---

## 🧪 适合继续完善的方向

1. 增加更清晰的模型版本说明与训练指标表。
2. 将训练输出、运行代码、实验素材进一步拆分，提升工程整洁度。
3. 将 Web 版页面改造成更适合毕业答辩展示的可视化大屏。
4. 增加统一的一键启动脚本，降低复现门槛。

---

## 📌 说明

本项目适合作为：

- 毕业设计原型系统
- 智能座舱安全监测演示项目
- 驾驶员疲劳 / 分心检测方向的工程化展示基础

如果这个项目对你有帮助，欢迎点个 ⭐。
