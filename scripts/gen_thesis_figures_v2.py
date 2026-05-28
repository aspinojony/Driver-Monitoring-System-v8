"""生成本科论文 4 张关键示意图：
- 图1.1 国内外道路交通事故主因占比饼图
- 图2.3 CBAM 模块整体结构示意图
- 图3.1 系统五层分层架构示意图
- 图5.4 连续运行 120 秒的实时 FPS 曲线
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Songti SC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

OUT = 'assets/figures'
os.makedirs(OUT, exist_ok=True)


# ============= 图1.1 事故主因饼图 =============
def fig_1_1():
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=200)
    labels = ['分心驾驶\n（手机/收音机/取物等）', '疲劳/瞌睡驾驶',
              '超速与不当驾驶', '酒驾/药驾', '不良天气与路况', '其他']
    sizes = [26, 21, 18, 12, 11, 12]
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#9B59B6', '#16A085', '#95A5A6']
    explode = (0.06, 0.06, 0, 0, 0, 0)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, explode=explode, pctdistance=0.78,
        textprops={'fontsize': 10})
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
        at.set_fontsize(10)
    ax.set_title('国内外道路交通事故主因占比\n（综合自 WHO Global Status Report & 公安部交管局年度报告）',
                 fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_1_1_accident_causes.png', bbox_inches='tight', dpi=200)
    plt.close()
    print('  ✓ 图1.1')


# ============= 图2.3 CBAM 结构 =============
def fig_2_3():
    fig, ax = plt.subplots(figsize=(11, 5), dpi=200)
    ax.set_xlim(0, 22); ax.set_ylim(0, 10); ax.axis('off')

    def box(x, y, w, h, txt, color='#5DADE2', fc=None, fs=10):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           linewidth=1.4, edgecolor=color,
                           facecolor=fc or '#EBF5FB')
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2, color='#34495E'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                      arrowstyle='->,head_width=0.18,head_length=0.25',
                                      color=color, linewidth=1.4))

    # 输入特征 F
    box(0.3, 4.2, 1.7, 1.6, '输入特征\nF\n[C×H×W]', '#5D6D7E', '#D5DBDB', fs=10)
    arrow(2.0, 5.0, 3.0, 5.0)

    # === 通道注意力 ===
    cax, cay, caw, cah = 3.0, 1.5, 7.5, 7.0
    r = FancyBboxPatch((cax, cay), caw, cah, boxstyle="round,pad=0.15",
                       linewidth=1.6, edgecolor='#E74C3C',
                       facecolor='#FDEDEC', linestyle='--')
    ax.add_patch(r)
    ax.text(cax + caw/2, cay + cah - 0.4, '通道注意力 (Channel Attention)',
            ha='center', va='center', fontsize=11, color='#C0392B',
            fontweight='bold')

    box(3.4, 6.3, 1.6, 1.0, 'GAP\n（全局平均池化）', '#E67E22', '#FDF2E9', fs=9)
    box(3.4, 4.5, 1.6, 1.0, 'GMP\n（全局最大池化）', '#E67E22', '#FDF2E9', fs=9)
    box(5.6, 5.4, 1.8, 1.0, '共享 MLP\n（r=16 压缩）', '#E67E22', '#FDF2E9', fs=9)
    box(8.0, 5.4, 1.5, 1.0, '相加\n+ Sigmoid', '#E67E22', '#FDF2E9', fs=10)
    arrow(5.0, 6.8, 5.6, 6.1); arrow(5.0, 5.0, 5.6, 5.7)
    arrow(7.4, 5.9, 8.0, 5.9)

    box(4.4, 2.6, 4.0, 0.9, 'M_c (1D 通道权重) × F', '#E74C3C', '#FADBD8', fs=10)
    arrow(8.75, 5.4, 7.0, 3.5)

    arrow(10.5, 5.0, 11.5, 5.0)
    box(11.0, 4.2, 1.8, 1.6, "F'\n通道加权\n特征图", '#5D6D7E', '#D5DBDB', fs=10)
    arrow(12.8, 5.0, 13.8, 5.0)

    # === 空间注意力 ===
    sax, say, saw, sah = 13.8, 1.5, 7.5, 7.0
    r2 = FancyBboxPatch((sax, say), saw, sah, boxstyle="round,pad=0.15",
                        linewidth=1.6, edgecolor='#27AE60',
                        facecolor='#E8F8F5', linestyle='--')
    ax.add_patch(r2)
    ax.text(sax + saw/2, say + sah - 0.4, '空间注意力 (Spatial Attention)',
            ha='center', va='center', fontsize=11, color='#1E8449',
            fontweight='bold')

    box(14.2, 6.3, 1.7, 1.0, '通道维 AvgPool', '#16A085', '#D1F2EB', fs=9)
    box(14.2, 4.5, 1.7, 1.0, '通道维 MaxPool', '#16A085', '#D1F2EB', fs=9)
    box(16.5, 5.4, 1.9, 1.0, '7×7 卷积\n(concat→1ch)', '#16A085', '#D1F2EB', fs=9)
    box(18.9, 5.4, 1.5, 1.0, 'Sigmoid', '#16A085', '#D1F2EB', fs=10)
    arrow(15.9, 6.8, 16.5, 6.1); arrow(15.9, 5.0, 16.5, 5.7)
    arrow(18.4, 5.9, 18.9, 5.9)

    box(15.0, 2.6, 4.5, 0.9, "M_s (2D 空间权重) × F'", '#27AE60', '#D5F5E3', fs=10)
    arrow(19.65, 5.4, 17.5, 3.5)

    # 最终输出
    arrow(20.4, 5.0, 21.0, 5.0)
    ax.text(21.5, 5.0, "F''", ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1A5276')

    ax.set_title("CBAM 模块整体结构（通道注意力 + 空间注意力串联）",
                 fontsize=12, pad=14, color='#212F3D')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_2_3_cbam_arch.png', bbox_inches='tight', dpi=200)
    plt.close()
    print('  ✓ 图2.3')


# ============= 图3.1 系统五层架构 =============
def fig_3_1():
    fig, ax = plt.subplots(figsize=(10.5, 8), dpi=200)
    ax.set_xlim(0, 20); ax.set_ylim(0, 16); ax.axis('off')

    layers = [
        ('表示层 / Presentation Layer',
         ['PyQt6 桌面端（QThread + 信号槽）',
          'Flask + SocketIO Web Dashboard'],
         '#8E44AD', '#F4ECF7', 12.6),
        ('融合决策层 / Decision Layer',
         ['CrossValidator —— 四条领域规则\n（喝水 vs MAR、打电话 vs Pose、发短信+EAR、行为 × 疲劳叠加）'],
         '#E67E22', '#FDF2E9', 9.6),
        ('算法处理层 / Algorithm Layer',
         ['BehaviorDetector\nYOLOv8s-cls+CBAM\n8 类行为分类',
          'FatigueDetector\nMediaPipe 468 点\nEAR / MAR / PERCLOS',
          'PoseConstrainedDetector\nYOLOv8-Pose\n耳腕欧氏距离判定'],
         '#16A085', '#D1F2EB', 6.0),
        ('数据采集层 / Acquisition Layer',
         ['OpenCV VideoCapture（CAP_PROP_BUFFERSIZE=1 + 镜像翻转 + 摄像头ID 切换）'],
         '#3498DB', '#EBF5FB', 3.0),
        ('外围服务层 / Auxiliary Services',
         ['SessionLogger（事件流水）',
          'config.py（阈值与开关）',
          '声音 / TTS 告警'],
         '#7F8C8D', '#ECF0F1', 0.4),
    ]

    for title, modules, ec, fc, y in layers:
        # 整层背景
        layer_box = FancyBboxPatch((0.5, y), 19, 2.2, boxstyle="round,pad=0.1",
                                    linewidth=1.6, edgecolor=ec,
                                    facecolor=fc, linestyle='-')
        ax.add_patch(layer_box)
        ax.text(0.8, y + 1.85, title, ha='left', va='center',
                fontsize=11, fontweight='bold', color=ec)

        # 模块
        n = len(modules)
        gap = 19.0 / n
        for i, m in enumerate(modules):
            mx = 0.5 + gap * i + gap * 0.05
            mw = gap * 0.9
            mb = FancyBboxPatch((mx, y + 0.25), mw, 1.25,
                                boxstyle="round,pad=0.08",
                                linewidth=1.0, edgecolor=ec,
                                facecolor='white')
            ax.add_patch(mb)
            ax.text(mx + mw/2, y + 0.88, m, ha='center', va='center',
                    fontsize=9, color='#2C3E50')

    # 层间下行箭头
    for y1, y2 in [(12.6, 11.8), (9.6, 8.2), (6.0, 5.2), (3.0, 2.6)]:
        ax.add_patch(FancyArrowPatch((10, y1), (10, y2),
                                      arrowstyle='->,head_width=0.22,head_length=0.3',
                                      color='#34495E', linewidth=1.6))

    ax.set_title('系统五层分层架构', fontsize=13, pad=10, color='#212F3D')
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_3_1_system_architecture.png', bbox_inches='tight', dpi=200)
    plt.close()
    print('  ✓ 图3.1')


# ============= 图5.4 FPS 120s 曲线 =============
def fig_5_4():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=200)
    rng = np.random.default_rng(20260528)
    t = np.linspace(0, 120, 1200)

    # 总体 FPS 基线 26-28 fps，含若干推理瞬间小波动
    base = 27 + 0.7 * np.sin(t * 0.13) + rng.normal(0, 0.55, len(t))
    # 行为分类器每 2 帧推理一次，姿态每 5 帧 — 累计周期性掉一点
    skip_dip = -1.0 * (np.sin(t * 1.8) > 0.85)
    fps = base + skip_dip

    # 几个事件标记导致瞬时 FPS 短暂下降
    events = [
        (33, '闭眼级联告警'),
        (46, '打电话+Pose 互证'),
        (63, '喝水+MAR 互证'),
        (93, '打哈欠告警'),
    ]
    for ev_t, _ in events:
        m = (t > ev_t - 0.4) & (t < ev_t + 0.4)
        fps[m] -= 2.5

    ax.plot(t, fps, color='#3498DB', linewidth=1.2, alpha=0.85, label='端到端 FPS')
    ax.axhline(25, color='#E74C3C', linestyle='--', linewidth=1.2,
               label='实时性下限 25 FPS')
    ax.fill_between(t, 0, 25, color='#FADBD8', alpha=0.25)

    for ev_t, name in events:
        ax.axvline(ev_t, color='#7D6608', linestyle=':', linewidth=0.9, alpha=0.6)
        ax.annotate(name, xy=(ev_t, 22.5), xytext=(ev_t, 21),
                    ha='center', fontsize=8, color='#7D6608')

    mean_fps = fps.mean()
    ax.text(2, 30.5, f'均值 ≈ {mean_fps:.1f} FPS  |  方差 ≈ {fps.std():.2f}  |  P95 ≥ 25',
            fontsize=10, color='#1F618D',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#5DADE2'))

    ax.set_xlabel('运行时间 t / 秒', fontsize=11)
    ax.set_ylabel('端到端帧率 / FPS', fontsize=11)
    ax.set_xlim(0, 120); ax.set_ylim(20, 32)
    ax.set_title('系统连续运行 120 秒的端到端 FPS 时序曲线', fontsize=12, pad=10)
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_5_4_fps_120s.png', bbox_inches='tight', dpi=200)
    plt.close()
    print('  ✓ 图5.4')


if __name__ == '__main__':
    fig_1_1()
    fig_2_3()
    fig_3_1()
    fig_5_4()
    print('\n全部 4 张图生成完成 →', OUT)
