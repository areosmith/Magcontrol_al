# 三轴旋转匀强磁场计算模块

本目录集中保存新增的磁场计算程序、标定配置、测试，以及对原项目文件的
集成副本。根目录的 `README.md` 和 `modulepack.py` 仍保留，避免破坏现有
GUI 的导入方式。

计算链路：

```text
RotatingFieldCommand
  -> 目标旋转磁场 B(t)
  -> 实测磁场矩阵求逆得到线圈电流 I(t)
  -> 线圈 R/L 逆模型得到 DAQ 电压 V(t)
  -> modulepack.set_voltage_source(...)
```

内部统一使用 SI 单位：磁场为 T、电流为 A、频率为 Hz、电压为 V。

## 目录内容

```text
magnetic_control/
├── __init__.py
├── calculator.py
├── calibration.py
├── magnetic_control_config.example.json
├── integration/
│   └── modulepack.py
└── tests/
    ├── test_calculator.py
    ├── test_calibration.py
    └── test_modulepack.py
```

`integration/modulepack.py` 是根目录已修改 `modulepack.py` 的归档副本。
现有 GUI 仍然使用根目录版本。

## 最小使用示例

从项目根目录运行：

```python
import time

import modulepack as mp
from magnetic_control import (
    ActuatorCalibration,
    RotatingFieldCalculator,
    RotatingFieldCommand,
)

calibration = ActuatorCalibration.from_json(
    "magnetic_control/magnetic_control_config.json"
)
calculator = RotatingFieldCalculator(
    calibration,
    sample_rate_hz=mp.SAMPLE_RATE,
)

command = RotatingFieldCommand(
    axis_xyz=[0.0, 0.0, 1.0],
    magnitude_T=2e-3,
    frequency_hz=5.0,
    rotation_sense=1,
)

print(calculator.preview(command).to_dict())
calculator.set_command(command)
mp.set_voltage_source(calculator.generate_voltage_block)
mp.start()

# 运行中可平滑改变旋转轴、场强、频率和旋向
calculator.set_command(
    RotatingFieldCommand([1.0, 1.0, 0.0], 1.5e-3, 8.0, -1)
)

# 先让电流完成回零斜坡和 DAQ 缓冲延迟，再停止 DAQ
calculator.stop()
buffer_latency_s = mp.PREFILL_BUFFERS * mp.BUFFER_SIZE / mp.SAMPLE_RATE
time.sleep(calculator.status().transition_remaining_s + buffer_latency_s)
mp.stop()
mp.close()
```

现有 GUI 默认仍使用 `modulepack` 的手动正弦/恒压源。恢复手动源：

```python
mp.stop()
mp.use_manual_voltage_source()
```

## 标定配置

复制 `magnetic_control_config.example.json` 为
`magnetic_control_config.json` 后填入实测值。示例数值仅用于说明格式，
`verified` 默认为 `false`。真实硬件模式要求：

- 三轴磁场矩阵满秩，条件数不超过配置阈值；
- 功率放大器增益、线圈电阻和电感已测量；
- 峰值/RMS 电流、DAQ 电压和有效频率范围已确认；
- 将配置明确标记为 `verified: true`。

可使用 `fit_field_matrix()` 拟合
`B = C_BI @ I + B_ambient`，使用 `fit_rl_channel()` 从复数电压/电流
相量拟合单路线圈的 `R + jωL` 模型。

## 故障恢复

自定义电压源出现异常、非有限值、形状错误或越过 AO 范围时，
`modulepack` 会锁存故障并持续返回零电压块。检查
`mp.get_output_fault()`；确认 DAQ 已停止且故障原因已排除后：

```python
calculator.clear_fault()
mp.clear_output_fault()
```

## 自动化测试

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s magnetic_control\tests -v
```
