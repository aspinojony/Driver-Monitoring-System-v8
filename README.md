<div align="center">

# 🚗 Advanced Driver Monitoring System (DMS) v8

### 基于深度学习与空间结构特征的车载多模态感知监控控制台

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](#)
[![PyQt6](https://img.shields.io/badge/PyQt6-UI%20Framework-green?style=for-the-badge&logo=qt)](#)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Deep%20Learning-orange?style=for-the-badge)](#)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-red?style=for-the-badge&logo=google)](#)
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)](#)

_专为智能座舱环境设计的毫秒级预警引擎，能无视极端暗光环境精准逮捕疲劳与分心驾驶行为。_

</div>

---

## 🌟 核心功能亮点 (Key Features)

- **🕸️ 468点 3D 面部网格映射**：利用 MediaPipe 高频提取 `极度疲劳(PERCLOS)`、`闭眼(EAR)` 与 `连续哈欠(MAR)`。
- **🧠 卷积神经网络姿态感知**：搭载经离线数据增强微调的 YOLOv8-cls 骨干网络，可识别 `玩手机`、`抽烟`、`向后转身` 等高达 9 种违章姿态。
- **🌙 CLAHE 极致暗场营救机制**：当车内极度逆光或进入黑暗隧道导致面部追踪丢失时，系统自动在内存中进行环境光矩阵直方图均衡化解码，实现零延迟断点续传。
- **🤖 千人千面自适应引擎**：彻底抛弃死板的预设阈值。前 60 帧动态学习驾驶员专属面部基线张量。
- **🛡️ 滞回滤波防抖矩阵**：独创的时段滑动窗口逻辑 (Smooth Window)，100% 免疫日常唱歌、谈话与短促眨眼造成的误触报警。
- **📟 工业级多线程流式交互大屏**：基于 PyQt6 编写的全并发客户端，音视频解码与流媒体分析彻底与 UI 主线程剥离。

## ⚙️ 系统架构剖析

![System Architecture](https://img.shields.io/badge/Architecture-Overview-lightgrey?style=for-the-badge)

本项目采用**双并发流**进行动作解析，任何一项指标越界均会触发联动：

- `YOLO Worker`: 处理低频大图画幅感知，识别全局躯干违章。
- `MediaPipe Worker`: 处理极致高频的人脸精微结构抓取。
- `Feedback Engine`: 根据危险层级，智能渲染 UI 卡片至 🟢安全态、🟡警告态(警示黄色边框) 或 🔴高危态(刺眼警急红，并带有语音 TTS 播报冷却池)。

## 🛠️ 快速安装运行 (Quick Start)

### 1. 环境准备

项目根目录已提供完整的依赖锁，建议使用虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 补全权重文件 (Weights)

由于模型体积与训练集过大，GitHub 不含原始 `.pt` 权重文件。
请确保在 `./data/weights/` 目录下放置训练好的 `best.pt` 权重文件。（本项目内置了强大的物理图像增强脚本 `scripts/offline_augment.py` 辅助训练）。

### 3. 一键启动终端控制台

```bash
python main.py
```

## 🎥 核心交互模块说明

| 模块名称           | 技术实现                 | 业务用途                                   |
| :----------------- | :----------------------- | :----------------------------------------- |
| **实时推流控制台** | PyQt6 / WebCam 0 / RTSP  | 兼容车内原装摄像头硬件与测试级录像流       |
| **自适应感光矩阵** | OpenCV CLAHE             | 强行提亮背光暗部特征，极小化检测盲区       |
| **自适应阈值回归** | Temporal Logic Buffers   | "千人千面"计算，自动推算不同瞳距的闭眼界限 |
| **多通道异常侦测** | YOLOv8 + Facial Topology | 组合拳侦测：疲劳+违章分心双线并行          |

## 🧪 高级研究特性及拓展 (For Academic Defense)

本项目专门为论文撰写和架构演示预留了以下深度学习特性切入点：

1. **网络改造前瞻**：推荐在骨干网络中预留 `CBAM` 或 `SE` 注意力层空间，解决背景过拟合。
2. **端侧推理量化**：已解耦所有运算逻辑，无缝支持转换为 `ONNX` 将模型以 INT8 精度下发至车载 SoC 行车记录仪。
3. **恶劣物理增强**：内置离线数据强化脚本，一键将数据扩充 3 倍并叠加高斯模糊、矩阵致暗处理。

---

_Developed as a high-fidelity Engineering Prototype / Graduation Project. If this code helps with your academic defense or project landing, consider leaving a ⭐ Star!_
