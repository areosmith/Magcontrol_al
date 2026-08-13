# Magnetic Control

三通道线圈磁场控制与旋转式微型机器人推进控制模块。

本模块根据目标旋转磁场或经实验标定的目标推进力，计算三路 DAQ 模拟
电压。DAQ 作为信号发生器，电压经功率放大器转换为线圈电流，三组线圈
共同生成磁场。高斯计用于离线标定和独立验证磁场模型。

```text
目标旋转场 / 目标推进力
          │
          ▼
磁场与推进模型 ──► 三路电流指令 ──► 三路 DAQ 电压
                                             │
                                             ▼
                                      信号放大器
                                             │
                                             ▼
                                      三组线圈电流
                                             │
                                             ▼
                                           磁场
                                             ▲
                                             │
                                   高斯计标定与验证
```

三组线圈可以具有不同的尺寸、轴距、电阻、电感和放大器增益。程序不假定
三轴完全正交，也不使用同一个比例系数描述三组线圈。中心磁场采用高斯计
拟合得到的完整矩阵：

```text
B = C_BI @ I_model + B_ambient
```

`C_BI` 的三列按 DAQ 通道 `ao0`、`ao1`、`ao2` 排列，分别表示对应通道
产生的三维磁场。线圈尺寸差异、安装角误差和一定程度的通道耦合均由实测
矩阵体现。

## 当前状态

已经实现：

- 三轴任意旋转轴、场强、频率和旋向的旋转磁场波形；
- 相位连续的命令切换和受控停止；
- 三路不同线圈参数、不同限值和不同驱动方式；
- 电压放大器和电流驱动器，可逐通道混合配置；
- 峰值电流、RMS 电流、DAQ 电压、频率和矩阵条件数检查；
- 高斯计 CSV 导入、Gauss/Tesla 单位转换和中心场矩阵拟合；
- 经验证的机器人推进 LUT、step-out 裕度和目标力反解；
- 不可行目标的结构化拒绝，以及显式允许后的安全边界投影；
- 二维/三维规则网格场图插值，不允许超出标定区域外推；
- XY 平台抽象接口、仿真平台和带迟滞的回中控制；
- 与现有 `modulepack` 的 `(3, samples)` 电压块接口。

尚未实现：

- 特定型号高斯计的 USB、串口或厂商 SDK 驱动；
- 真实 XY 平台的厂商协议；
- 相机跟踪和实时力传感器驱动；
- 位置相关场图在当前 DAQ 波形计算器中的在线切换；
- 未经实验标定即可使用的通用“力—频率”模型。

因此，当前实机输出使用中心位置的磁场标定。空间场图和平台模块已经可以
独立测试，但不能据此声称系统已经具备移动过程中的全工作区磁场补偿。

## 硬件链路

### DAQ

现有输出层使用三路模拟输出：

- 通道：`ao0`、`ao1`、`ao2`；
- 采样率：100 kS/s；
- 波形块：每通道 1000 点，即 10 ms；
- 启动预填：4 个波形块；
- 输出范围：-10 V 至 10 V。

DAQ 只负责生成电压，不测量磁场。根目录的 `modulepack.py` 是当前 GUI 和
真实 DAQ 使用的实现；`integration/modulepack.py` 是归档副本。

### 信号放大器

每个通道从以下两种模式中选择一种：

| 模式 | 配置值 | 换算方式 |
|---|---|---|
| 电压放大器 | `voltage_amplifier` | DAQ V → 线圈端 V，再由线圈 R/L 得到电流 |
| 电流驱动器 | `current_driver` | DAQ V → 按 A/V 增益和闭环带宽得到线圈电流 |

如果尚不能确定放大器类型，应先查看设备说明书或测量静态 A/V 关系，不能
直接使用示例配置。旋转场要求每路功放和接线能够输出双极性电流；接线前还
要确认模拟地或隔离方式、输入范围、硬件使能和断电后的续流路径。只支持
单向电流的驱动器不能直接使用当前波形模型。

### 高斯计

高斯计用于测量各组线圈在已知电流下产生的 `Bx`、`By`、`Bz`。当前程序
使用离线 CSV，不直接连接仪器。如果高斯计只有单轴探头，需要以一致的
坐标定义分别测出三个方向，再整理成三轴数据。

## 文件结构

```text
magnetic_control/
├── calibration.py                  # 驱动器、线圈和高斯计标定
├── calculator.py                   # 旋转场到三路 DAQ 电压
├── force_command.py                # 目标推进量及其物理语义
├── propulsion_model.py             # 机器人推进 LUT 和配置
├── force_allocator.py              # 目标推进力到旋转场
├── supervisor.py                   # 力分配与电气计算器协调
├── spatial_field_map.py            # 中心/规则网格空间场图
├── stage.py                        # XY 平台接口、仿真和回中控制
├── magnetic_control_config.example.json
├── robot_profile.example.json
├── gaussmeter_measurements.example.csv
├── integration/
└── tests/
```

系统设计和研究边界见
[`FORCE_CONTROL_MOBILE_WORKSPACE_DESIGN.md`](FORCE_CONTROL_MOBILE_WORKSPACE_DESIGN.md)。

## 环境

- Python 3.10 或更高版本；
- NumPy；
- 真实 DAQ 输出还需要项目 `LIB` 中的 `artdaq` 驱动及对应硬件环境；
- PyQt6 仅用于项目现有 GUI，不是磁场计算模块的必要依赖。

从项目根目录创建环境并安装计算依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r magnetic_control\requirements.txt
```

## 无硬件试运行

示例配置全部是占位值，默认 `verified: false`。仿真时必须显式使用
`allow_unverified=True`：

```python
from magnetic_control import (
    ActuatorCalibration,
    RotatingFieldCalculator,
    RotatingFieldCommand,
)

calibration = ActuatorCalibration.from_json(
    "magnetic_control/magnetic_control_config.example.json"
)
calculator = RotatingFieldCalculator(
    calibration,
    sample_rate_hz=100_000,
    allow_unverified=True,
)

command = RotatingFieldCommand(
    axis_xyz=[0.0, 0.0, 1.0],
    magnitude_T=0.2e-3,
    frequency_hz=5.0,
    rotation_sense=1,
)

preview = calculator.preview(command)
calculator.set_command(command)
voltage = calculator.generate_voltage_block(1000)

print(preview.to_dict())
print(voltage.shape)  # (3, 1000)
```

`allow_unverified=True` 只用于软件测试，不能作为跳过实机标定的常规选项。

## 硬件配置

复制配置模板：

```powershell
Copy-Item magnetic_control\magnetic_control_config.example.json `
  magnetic_control\magnetic_control_config.json
```

主要字段如下：

| 字段 | 单位 | 含义 |
|---|---|---|
| `channel_axes` | — | 通道标签，必须是 `x/y/z` 的排列；当前不负责矩阵或 DAQ 重排 |
| `channel_drive_modes` | — | 每路选择电压放大器或电流驱动器 |
| `voltage_gain_v_per_v` | V/V | JSON v2 中电压放大器输出电压与 DAQ 电压之比 |
| `current_gain_a_per_v` | A/V | 电流驱动器输出电流与 DAQ 电压之比 |
| `current_driver_bandwidth_hz` | Hz | 电流驱动器一阶闭环带宽，0 表示理想模型 |
| `coil_resistance_ohm` | Ω | 三组线圈各自的直流电阻 |
| `coil_inductance_h` | H | 三组线圈各自的电感 |
| `field_matrix_t_per_a` | T/A | 三路模型电流到三维磁场的完整矩阵 |
| `ambient_field_t` | T | 线圈零电流时的背景磁场 |
| `channel_signs` | — | 电气正电流与场模型正方向的符号关系 |
| `daq_voltage_limit_v` | V | 每路允许的 DAQ 峰值电压，当前不得超过 10 V |
| `peak_current_limit_a` | A | 每路线圈峰值电流限制 |
| `rms_current_limit_a` | A | 每路线圈 RMS 电流限制 |
| `valid_frequency_hz` | Hz | 已验证的工作频率范围 |
| `verified` | — | 是否通过完整标定和独立验证 |

三个通道的数组值必须逐路填写。示例中的线圈参数和磁场矩阵不代表实际
硬件性能。若直接调用 Python 构造器，JSON 字段 `voltage_gain_v_per_v` 对应
兼容参数 `amplifier_gain_v_per_v`；使用 `from_json()` 时会自动处理。

`valid_frequency_hz` 中的 `0 Hz` 只能作为标定区间下界；旋转场命令的频率
必须大于零。上限还必须低于 `sample_rate/2`，并以功放、线圈和目标场强下的
实测带宽为准。

## 高斯计中心场标定

### 1. 准备测量

先测量三路放大器的增益和线圈实际电流。然后在固定的工作区中心，依次
施加零电流、每路正负多档电流，并记录高斯计结果。建议同时保留若干多路
组合电流作为验证点。

CSV 可以使用 Tesla：

```text
current_ch0_a,current_ch1_a,current_ch2_a,field_x_t,field_y_t,field_z_t
```

也可以使用 Gauss：

```text
current_ch0_a,current_ch1_a,current_ch2_a,field_x_gauss,field_y_gauss,field_z_gauss
```

电流列是 `I_model`，即与 `field_matrix_t_per_a` 列定义一致的带符号电流。
如果记录的是实际电气电流，应先用 `channel_signs` 转换。两套磁场单位列
不能同时出现在一个文件中。

### 2. 拟合矩阵

```python
from magnetic_control import GaussmeterDataset

dataset = GaussmeterDataset.from_csv(
    "magnetic_control/gaussmeter_measurements.csv"
)
fit = dataset.fit_field_matrix()

print("C_BI [T/A]:\n", fit.field_matrix_t_per_a)
print("ambient [T]:", fit.ambient_field_t)
print("condition number:", fit.condition_number)
print("maximum residual [T]:", fit.max_vector_error_t)
```

若采集程序只记录静态 DAQ 电压，可以在放大器 DC 增益已验证后转换：

```python
dataset = GaussmeterDataset.from_static_daq_voltages(
    calibration,
    daq_voltages_v,      # shape: (N, 3)
    measured_fields_t,   # shape: (N, 3)
)
```

实际电流测量优先于由 DAQ 电压推算电流，因为放大器可能限流、饱和或随
负载变化。

### 3. 验证与启用

将拟合结果写入 `magnetic_control_config.json` 后，还需要使用未参与拟合的
电流组合验证：

- 三个方向的静态场幅值和方向；
- 多个频率下旋转场的幅值、相位和圆度；
- 峰值及 RMS 电流；
- DAQ 与放大器是否饱和；
- 线圈温升和连续运行时间。

只有完成独立验证并保存实测指标后，才将 `verified` 改为 `true`。
仓库中的高斯计 CSV 只是列名和拟合流程示例，所有点都参与拟合，不能作为
独立验证数据，也不能据此启用真实硬件。

## 输出旋转磁场

真实硬件配置通过检查后，可将计算器注册为 `modulepack` 的电压源：

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
try:
    mp.start()
    time.sleep(5.0)  # 示例运行时间；实际程序应由上层控制循环决定
finally:
    # 受控停止。队列时间只是名义估计，不是硬件生效确认。
    calculator.stop()
    queued_s = mp.PREFILL_BUFFERS * mp.BUFFER_SIZE / mp.SAMPLE_RATE
    time.sleep(calculator.status().transition_remaining_s + queued_s)
    mp.stop()
    mp.close()
```

`axis_xyz` 是旋转平面的法向，不自动等同于所有机器人的推进方向。推进方向
还取决于机器人手性、磁化方向和 `rotation_sense`。

## 目标推进力

旋转磁场主要施加磁力矩。特殊形状机器人通过旋转—平移耦合产生推进，
因此目标推进力必须来自具体机器人、介质、壁距和载荷条件下的实验标定。

机器人配置使用二维单调 LUT：行对应场强，列对应频率，值为测得的推进
分量或堵转力。复制 `robot_profile.example.json` 并替换全部实验数据；模板
默认 `validated: false`。

```python
import modulepack as mp
from magnetic_control import (
    ActuatorCalibration,
    ForceAllocator,
    MagneticControlSupervisor,
    PropulsionCommand,
    PropulsionMode,
    RobotProfile,
    RotatingFieldCalculator,
)

requested_force_N = 1e-12  # 纯接口示例；必须改为实际 LUT 范围内的值
calibration = ActuatorCalibration.from_json(
    "magnetic_control/magnetic_control_config.json"
)
profile = RobotProfile.from_json(
    "magnetic_control/robot_profile.json"
)
calculator = RotatingFieldCalculator(calibration, mp.SAMPLE_RATE)
supervisor = MagneticControlSupervisor(
    ForceAllocator(),
    profile,
    calculator,
)

request = PropulsionCommand(
    target_xyz_si=[requested_force_N, 0.0, 0.0],
    mode=PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
    robot_profile_id=profile.profile_id,
)

result = supervisor.preview_force(request)
if not result.accepted:
    raise RuntimeError(result.message)

result = supervisor.set_force(request)
if not result.accepted:
    raise RuntimeError(result.message)

mp.set_voltage_source(supervisor.generate_voltage_block)
# 此处只完成电压源配置；按上一节的 try/finally 模式启停 DAQ。
```

超出推进 LUT 或 step-out 安全范围的请求默认被拒绝。只有命令显式设置
`project_to_feasible=True` 时，才会投影到 LUT 的安全边界，并在
`result.allocation.projected` 中标记。线圈电流和 DAQ 电压约束随后仍由
计算器检查，不在这一步的投影范围内。

被拒绝的新命令不会修改计算器状态；如果旧命令正在输出，它会继续运行。
需要“拒绝即停机”的应用必须在上层显式调用 `supervisor.stop()`。

介质、温度、壁距、载荷和机器人版本必须与标定时一致。当前文件格式没有
单独校验这些元数据，应把它们编码进唯一的 `profile_id` 并由实验流程核对。

推进模式的含义：

| 模式 | 单位 | 含义 |
|---|---|---|
| `ESTIMATED_PROPULSIVE_COMPONENT` | N | 自由游动时估计的推进分量，不是非零净力 |
| `BLOCKED_THRUST` | N | 接触/堵转工况下的力，要求确认并标识接触对象 |
| `FREE_SWIM_VELOCITY` | m/s | 自由游动速度；当前 `ForceAllocator` 不处理此模式 |

零目标由 `MagneticControlSupervisor` 转换为安全停止，不会构造零频率的
`RotatingFieldCommand`。

## 空间场图与 XY 平台

`CenterFieldMap` 默认只承认标定中心有效，除非用户提供经过测量的有效半径。
`RegularGridFieldMap` 支持二维和三维规则网格多线性插值，网格外查询直接
返回无效，不进行外推。位置单位为 m；二维矩阵网格形状为
`(Nx, Ny, 3, 3)`，三维为 `(Nx, Ny, Nz, 3, 3)`，背景场数组末尾维度为 3。

场图查询的 `result.valid` 只表示位置在已测范围内，且插值节点未被
`valid_mask` 排除；它不表示矩阵已验证、满秩或条件数合格。用于控制前仍须
进行这些质量检查。

`stage.py` 提供：

- `XYStage` 硬件接口约定；
- `SimulatedXYStage` 确定性逻辑仿真，需要显式调用 `advance()` 推进时间；
- 行程、速度和名义加速度参数；
- 归零、时间戳和故障状态；
- `DeadbandRecenterController` 带迟滞的回中控制。

仿真平台的 `stop()` 是瞬时逻辑停止，不能用于验证真实机构的制动距离、换向
动力学或安全等级。`local_to_stage` 的符号和尺度也必须由机构与相机外参标定。

接入真实平台时，需要实现 `XYStage` 协议并建立平台、线圈、相机和样品坐标
之间的实测变换。当前 `RotatingFieldCalculator` 仍缓存一个固定中心矩阵，
移动过程的在线空间补偿尚未接入真实波形链路。

## 安全说明

- 示例配置和示例机器人 LUT 均为占位数据，不能用于真实硬件；
- 未验证配置默认无法创建实机计算器；
- 每次输出前先调用 `preview()` 检查稳态电气需求；
- 软件零电压不等于感性线圈电流立即归零；
- `modulepack` 的故障锁存和零电压块不是硬件急停；
- 实机应配置功放硬件使能、急停、续流/能量钳位、温度和电流监测；
- 当前三路线圈用于合成磁场，不能独立指定任意三维磁场梯度。
- 场图查询有效不代表标定可用，实机还必须检查验证状态、秩和条件数。

恢复锁存故障前，应先停止 DAQ 并排除硬件原因：

```python
calculator.clear_fault()
mp.clear_output_fault()
```

## 测试

从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s magnetic_control\tests -v
```

测试覆盖磁场合成、两类驱动器、高斯计拟合、目标力反解、空间插值、平台
仿真、监督器和 DAQ 电压源契约。故障注入测试会有意打印两条
`[Output Fault]` 日志；测试最终显示 `OK` 即为预期结果。

## 后续工作

近期实机工作的合理顺序是：

1. 确认三路放大器类型并测量每路增益；
2. 测量三组线圈的 R/L、电流限制和温升；
3. 用高斯计完成中心场矩阵与背景场标定；
4. 验证旋转场的幅值、方向、圆度和频率范围；
5. 对具体机器人测量推进 LUT 和 step-out；
6. 接入视觉或微力测量形成闭环；
7. 最后接入 XY 平台和位置相关场图。
