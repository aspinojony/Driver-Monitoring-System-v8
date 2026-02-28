import sys
import os
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    # Set the model path override for MainWindow's BehaviorDetector if needed,
    # but the cleanest way is just to let MainWindow initialize it, or we patch it here.
    # The BehaviorDetector defaults to the old model in its __init__. Let's update that class.

    app = QApplication(sys.argv)
    window = MainWindow()

    # Try to load the new classification model, fallback to the old detection one if missing
    new_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "weights",
        "yolov8n_driver_cls",
        "weights",
        "best.pt",
    )
    if os.path.exists(new_model_path) and window.thread is None:
        # We need to tell the thread to use this new model when started.
        # However, BehaviorDetector is initialized inside VideoThread, which is created on btn_start.
        pass  # We will modify BehaviorDetector's default path instead.

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
