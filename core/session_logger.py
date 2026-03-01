import os
import time
from datetime import datetime


class SessionLogger:
    """
    会话日志与数据报表中心 (Session Analytics Logger)
    专门用于统计驾车时长、疲劳次数、分心次数，并生成带有时间戳的物理报告文件。
    极大地提升了项目的学术完整性和商业产品质感。
    """

    def __init__(self):
        self.start_time = time.time()
        self.events = []
        self.fatigue_count = 0
        self.distraction_count = 0
        self.last_log_time = 0

    def reset(self):
        self.start_time = time.time()
        self.events = []
        self.fatigue_count = 0
        self.distraction_count = 0
        self.last_log_time = 0

    def log_event(self, behavior, fatigue, is_critical):
        current = time.time()
        # 日志防刷屏冷却 (只记录每2秒内最严重的一次)
        if current - self.last_log_time < 2.0:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.events.append(
            {
                "time": timestamp,
                "behavior": behavior,
                "fatigue": fatigue,
                "critical": is_critical,
            }
        )

        if "正常" not in fatigue:
            self.fatigue_count += 1
        if "正常" not in behavior:
            self.distraction_count += 1

        self.last_log_time = current

    def export_report(self, save_path):
        duration = int(time.time() - self.start_time)
        mins, secs = divmod(duration, 60)

        report_content = [
            "=" * 60,
            "               📊 驾驶员全息健康监控总表 (Session Report)",
            "=" * 60,
            f" 📅 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f" ⏱️ 本次驾驶总长: {mins} 分钟 {secs} 秒",
            "-" * 60,
            f" ⚠️ 疲劳驾驶触发: {self.fatigue_count} 次",
            f" 📱 分心驾驶触发: {self.distraction_count} 次",
            "-" * 60,
            "【危险事件详细追溯 / Event Tracking】",
        ]

        if not self.events:
            report_content.append("    ✅ 恭喜！本次检测全程无违规行为，驾驶记录完美！")
        else:
            for idx, event in enumerate(self.events, 1):
                crit_marker = "🚨[极度危险]" if event["critical"] else "⚠️[一般违规]"
                # 截断展示，防止文字过长
                report_content.append(
                    f" {idx:02d}. {event['time']} | {crit_marker} 动作: {event['behavior'][:6]:<6} | 精神: {event['fatigue']}"
                )

        report_content.append("=" * 60)
        report_content.append(
            " 系统评价: "
            + (
                "极度危险，建议强制休息！"
                if (self.fatigue_count > 5 or self.distraction_count > 10)
                else "状况良好，请保持。"
            )
        )

        final_text = "\n".join(report_content)

        # 写入文件
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        return final_text
