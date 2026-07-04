from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import modulepack as mp


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.freq_edits = []
        self.amp_edits = []
        self.constvolt_edits = []

        self.ref_group = QButtonGroup(self)
        self.tgt_group = QButtonGroup(self)
        self.mode_group = QButtonGroup(self)

        self.setWindowTitle("ART DAQ Sine Wave Controller - PyQt6")
        self.resize(720, 680)

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)

        # =========================
        # 控制按钮
        # =========================
        control_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.close_btn = QPushButton("Close")

        self.start_btn.setFixedWidth(110)
        self.stop_btn.setFixedWidth(110)
        self.close_btn.setFixedWidth(110)

        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }"
        )
        self.close_btn.setStyleSheet(
            "QPushButton { background-color: #9E9E9E; color: white; font-weight: bold; padding: 8px; }"
        )

        self.start_btn.clicked.connect(self.start_task)
        self.stop_btn.clicked.connect(self.stop_task)
        self.close_btn.clicked.connect(self.close)

        control_layout.addStretch()
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.close_btn)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # =========================
        # 波形模式切换
        # =========================
        mode_group = QGroupBox("Waveform Mode")
        mode_layout = QHBoxLayout(mode_group)

        self.sine_rb = QRadioButton("Sine Wave")
        self.const_rb = QRadioButton("Constant Voltage")
        self.mode_group.addButton(self.sine_rb, 0)
        self.mode_group.addButton(self.const_rb, 1)
        self.sine_rb.setChecked(True)

        self.mode_group.idClicked.connect(self.switch_mode)

        mode_layout.addStretch()
        mode_layout.addWidget(self.sine_rb)
        mode_layout.addWidget(self.const_rb)
        mode_layout.addStretch()

        main_layout.addWidget(mode_group)

        # =========================
        # 通道参数
        # =========================
        channel_group = QGroupBox("Channel Parameters")
        channel_layout = QGridLayout(channel_group)

        channel_layout.addWidget(QLabel("Channel"), 0, 0)
        channel_layout.addWidget(QLabel("Frequency (Hz)"), 0, 1)
        channel_layout.addWidget(QLabel("Amplitude (V)"), 0, 2)
        channel_layout.addWidget(QLabel("Action"), 0, 3)

        freq_validator = QDoubleValidator(
            mp.MIN_FREQUENCY,
            mp.MAX_FREQUENCY,
            6,
            self,
        )

        amp_validator = QDoubleValidator(
            0.0,
            mp.MAX_AMPLITUDE,
            6,
            self,
        )

        for ch in range(mp.NUM_CHANNELS):
            ch_label = QLabel(f"Ch{ch}")
            ch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            freq_edit = QLineEdit("1000.0")
            amp_edit = QLineEdit("1.0")

            freq_edit.setValidator(freq_validator)
            amp_edit.setValidator(amp_validator)

            apply_btn = QPushButton("Apply")
            apply_btn.clicked.connect(lambda checked=False, c=ch: self.apply_channel(c))

            self.freq_edits.append(freq_edit)
            self.amp_edits.append(amp_edit)

            row = ch + 1
            channel_layout.addWidget(ch_label, row, 0)
            channel_layout.addWidget(freq_edit, row, 1)
            channel_layout.addWidget(amp_edit, row, 2)
            channel_layout.addWidget(apply_btn, row, 3)

        main_layout.addWidget(channel_group)

        # =========================
        # 常量电压
        # =========================
        const_group = QGroupBox("Constant Voltage (per channel)")
        const_layout = QGridLayout(const_group)

        const_layout.addWidget(QLabel("Channel"), 0, 0)
        const_layout.addWidget(QLabel("Voltage (V)"), 0, 1)
        const_layout.addWidget(QLabel("Action"), 0, 2)

        const_validator = QDoubleValidator(
            -mp.MAX_AMPLITUDE,
            mp.MAX_AMPLITUDE,
            6,
            self,
        )

        for ch in range(mp.NUM_CHANNELS):
            ch_label = QLabel(f"Ch{ch}")
            ch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            const_edit = QLineEdit("0.0")
            const_edit.setValidator(const_validator)

            apply_btn = QPushButton("Apply")
            apply_btn.clicked.connect(lambda checked=False, c=ch: self.apply_constant(c))

            self.constvolt_edits.append(const_edit)

            row = ch + 1
            const_layout.addWidget(ch_label, row, 0)
            const_layout.addWidget(const_edit, row, 1)
            const_layout.addWidget(apply_btn, row, 2)

        main_layout.addWidget(const_group)

        # =========================
        # 相位差设置
        # =========================
        phase_group = QGroupBox("Phase Difference")
        phase_layout = QVBoxLayout(phase_group)

        ref_label = QLabel("Reference Channel:")
        phase_layout.addWidget(ref_label)

        ref_layout = QHBoxLayout()
        for i in range(mp.NUM_CHANNELS):
            rb = QRadioButton(f"Ch{i}")
            self.ref_group.addButton(rb, i)
            ref_layout.addWidget(rb)

            if i == 0:
                rb.setChecked(True)

        ref_layout.addStretch()
        phase_layout.addLayout(ref_layout)

        tgt_label = QLabel("Target Channel:")
        phase_layout.addWidget(tgt_label)

        tgt_layout = QHBoxLayout()
        for i in range(mp.NUM_CHANNELS):
            rb = QRadioButton(f"Ch{i}")
            self.tgt_group.addButton(rb, i)
            tgt_layout.addWidget(rb)

            if i == 1:
                rb.setChecked(True)

        tgt_layout.addStretch()
        phase_layout.addLayout(tgt_layout)

        phase_input_layout = QHBoxLayout()

        self.phase_edit = QLineEdit("0.0")
        phase_validator = QDoubleValidator(-360000.0, 360000.0, 6, self)
        self.phase_edit.setValidator(phase_validator)

        self.apply_phase_btn = QPushButton("Apply Phase")
        self.apply_phase_btn.clicked.connect(self.apply_phase)

        phase_input_layout.addWidget(QLabel("Phase Diff (deg):"))
        phase_input_layout.addWidget(self.phase_edit)
        phase_input_layout.addWidget(self.apply_phase_btn)
        phase_input_layout.addStretch()

        phase_layout.addLayout(phase_input_layout)

        main_layout.addWidget(phase_group)

        # =========================
        # 状态信息
        # =========================
        status_group = QGroupBox("Status / Log")
        status_layout = QVBoxLayout(status_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)

        status_layout.addWidget(self.log_text)

        main_layout.addWidget(status_group)

        # 分隔和底部说明
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)

        info_label = QLabel(
            f"Device: {mp.DEVICE_NAME}    "
            f"Channels: {mp.NUM_CHANNELS}    "
            f"Sample Rate: {mp.SAMPLE_RATE} Hz    "
            f"Buffer Size: {mp.BUFFER_SIZE}"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: gray;")
        main_layout.addWidget(info_label)

        self.log("Ready. Click Start to begin output.")

    def log(self, msg: str):
        self.log_text.append(msg)
        print(msg)

    def show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)
        self.log(f"[Error] {message}")

    def start_task(self):
        try:
            mp.start()
            self.log(">>> Task started")
        except Exception as e:
            self.show_error("Start Error", str(e))

    def stop_task(self):
        try:
            mp.stop()
            self.log(">>> Task stopped")
        except Exception as e:
            self.show_error("Stop Error", str(e))

    def apply_channel(self, ch: int):
        try:
            freq_text = self.freq_edits[ch].text().strip()
            amp_text = self.amp_edits[ch].text().strip()

            if not freq_text or not amp_text:
                raise ValueError("Frequency and amplitude cannot be empty")

            frequency = float(freq_text)
            amplitude = float(amp_text)

            mp.set_sine(ch, amplitude, frequency)

            self.log(f"[Ch{ch}] freq={frequency} Hz, amp={amplitude} V")

        except Exception as e:
            self.show_error("Parameter Error", str(e))

    def apply_constant(self, ch: int):
        try:
            volt_text = self.constvolt_edits[ch].text().strip()

            if not volt_text:
                raise ValueError("Voltage cannot be empty")

            voltage = float(volt_text)

            mp.set_constant(ch, voltage)

            self.log(f"[Ch{ch}] constant voltage = {voltage} V")

        except Exception as e:
            self.show_error("Parameter Error", str(e))

    def switch_mode(self, mode_id: int):
        if mode_id == 0:
            mp.set_mode("sine")
            self.log(">>> Mode switched to: Sine Wave")
        else:
            mp.set_mode("constant")
            self.log(">>> Mode switched to: Constant Voltage")

    def apply_phase(self):
        try:
            ref = self.ref_group.checkedId()
            tgt = self.tgt_group.checkedId()

            if ref < 0 or tgt < 0:
                raise ValueError("Please select reference and target channels")

            if ref == tgt:
                raise ValueError("Reference and target channels must be different")

            phase_text = self.phase_edit.text().strip()
            if not phase_text:
                raise ValueError("Phase difference cannot be empty")

            diff = float(phase_text)

            mp.set_phase_diff(ref, tgt, diff)

            self.log(f"[Phase] Ch{ref} -> Ch{tgt} = {diff} deg")

        except Exception as e:
            self.show_error("Phase Error", str(e))

    def closeEvent(self, event):

        try:
            mp.close()
            self.log(">>> Task closed")
        except Exception as e:
            QMessageBox.warning(self, "Close Warning", str(e))

        event.accept()