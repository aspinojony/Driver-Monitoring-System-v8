import os
import time
from datetime import datetime


class SessionLogger:
    """会话日志与报告导出。统计驾驶时长、疲劳/分心事件，生成 TXT/HTML 报告。"""

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
        # 2 秒冷却，防止刷屏
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
            "                  驾驶员监控会话报告",
            "=" * 60,
            f" 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f" 本次驾驶总长: {mins} 分 {secs} 秒",
            "-" * 60,
            f" 疲劳事件触发: {self.fatigue_count} 次",
            f" 分心事件触发: {self.distraction_count} 次",
            "-" * 60,
            "【事件追溯 / Event Tracking】",
        ]

        if not self.events:
            report_content.append("    本次会话全程无违规行为。")
        else:
            for idx, event in enumerate(self.events, 1):
                crit_marker = "[严重]" if event["critical"] else "[一般]"
                report_content.append(
                    f" {idx:02d}. {event['time']} | {crit_marker} 行为: {event['behavior'][:8]:<8} | 疲劳: {event['fatigue']}"
                )

        report_content.append("=" * 60)
        report_content.append(
            " 综合评价: "
            + (
                "高风险，建议立即休息"
                if (self.fatigue_count > 5 or self.distraction_count > 10)
                else "低风险"
            )
        )

        final_text = "\n".join(report_content)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        return final_text
