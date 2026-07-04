import sys
import threading
from pathlib import Path

import numpy as np



ARTDAQ_LIB_PATH = r"C:\Users\Hsu ichen\PycharmProjects\磁控花剌子模\LIB"

if Path(ARTDAQ_LIB_PATH).exists() and ARTDAQ_LIB_PATH not in sys.path:
    sys.path.append(ARTDAQ_LIB_PATH)

import artdaq
from artdaq.constants import AcquisitionType, RegenerationMode


DEVICE_NAME = "Dev1"

NUM_CHANNELS = 4
SAMPLE_RATE = 100000
BUFFER_SIZE = 1000


PREFILL_BUFFERS = 4

AO_MIN_VAL = -10.0
AO_MAX_VAL = 10.0

MAX_AMPLITUDE = 10.0
MIN_FREQUENCY = 0.0
MAX_FREQUENCY = SAMPLE_RATE / 2.0


class SharedParams:


    def __init__(self):
        self.lock = threading.Lock()

        self.freq = [1000.0] * NUM_CHANNELS
        self.amp = [1.0] * NUM_CHANNELS


        self.phase_rad = [0.0] * NUM_CHANNELS

    def check_channel(self, ch: int):
        if not isinstance(ch, int):
            raise TypeError("Channel index must be int")

        if ch < 0 or ch >= NUM_CHANNELS:
            raise ValueError(f"Channel must be between 0 and {NUM_CHANNELS - 1}")

    def set_channel(self, ch: int, freq=None, amp=None):
        self.check_channel(ch)

        with self.lock:
            if freq is not None:
                freq = float(freq)
                if freq < MIN_FREQUENCY or freq > MAX_FREQUENCY:
                    raise ValueError(
                        f"Frequency must be between {MIN_FREQUENCY} and {MAX_FREQUENCY} Hz"
                    )
                self.freq[ch] = freq

            if amp is not None:
                amp = float(amp)
                if amp < 0.0 or amp > MAX_AMPLITUDE:
                    raise ValueError(f"Amplitude must be between 0 and {MAX_AMPLITUDE} V")
                self.amp[ch] = amp

    def set_phase_diff(self, ch_ref: int, ch_target: int, diff_deg: float):
        self.check_channel(ch_ref)
        self.check_channel(ch_target)

        if ch_ref == ch_target:
            raise ValueError("Reference and target channels must be different")

        diff_rad = np.deg2rad(float(diff_deg))

        with self.lock:
            self.phase_rad[ch_target] = self.phase_rad[ch_ref] + diff_rad
            self.phase_rad[ch_target] = self.phase_rad[ch_target] % (2.0 * np.pi)

    def generate_waveform(self, samples: int = BUFFER_SIZE):


        with self.lock:
            freq = list(self.freq)
            amp = list(self.amp)
            phase = list(self.phase_rad)

            n = np.arange(samples, dtype=np.float64)
            data = np.zeros((NUM_CHANNELS, samples), dtype=np.float64)

            for ch in range(NUM_CHANNELS):
                omega = 2.0 * np.pi * freq[ch] / SAMPLE_RATE
                data[ch, :] = amp[ch] * np.sin(phase[ch] + omega * n)

                # 更新该通道相位到下一个 buffer 的起点
                self.phase_rad[ch] = (phase[ch] + omega * samples) % (2.0 * np.pi)

        return data


_shared = SharedParams()
_task = None
_running = False
_task_lock = threading.RLock()
_callback_ref = None


def _transfer_callback(task_handle, event_type, num_samples, callback_data):
    """ART DAQ 缓冲区传输回调。"""
    global _task, _running, _task_lock
    try:
        if not _running:
            return 0

        data = _shared.generate_waveform(BUFFER_SIZE)

        with _task_lock:
            if _task is not None and _running:
                _task.write(data, auto_start=False)

    except Exception as e:
        print(f"[Callback Error] {e}")

    return 0


def _close_locked():
    global _task, _running, _callback_ref

    _running = False

    if _task is not None:
        try:
            _task.stop()
        except Exception:
            pass

        try:
            _task.close()
            print(">>> Task closed")
        except Exception as e:
            print(f"[Close Error] {e}")
            raise
        finally:
            _task = None

    _callback_ref = None


def start():
    """启动 DAQ 连续输出。"""
    global _task, _running, _callback_ref

    with _task_lock:
        _close_locked()

        task = artdaq.Task("AOWave")

        try:
            for i in range(NUM_CHANNELS):
                task.ao_channels.add_ao_voltage_chan(
                    f"{DEVICE_NAME}/ao{i}",
                    name_to_assign_to_channel=f"ch{i}",
                    min_val=AO_MIN_VAL,
                    max_val=AO_MAX_VAL,
                )

            task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

            task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=BUFFER_SIZE * PREFILL_BUFFERS,
            )

            _callback_ref = _transfer_callback

            task.register_every_n_samples_transferred_from_buffer_event(
                BUFFER_SIZE,
                _callback_ref,
            )

            prefill_data = np.concatenate(
                [_shared.generate_waveform(BUFFER_SIZE) for _ in range(PREFILL_BUFFERS)],
                axis=1,
            )

            task.write(prefill_data, auto_start=False)

            _task = task
            _running = True

            task.start()

            print(">>> Task started")

        except Exception:
            _running = False

            try:
                task.stop()
            except Exception:
                pass

            try:
                task.close()
            except Exception:
                pass

            _task = None
            raise


def stop():
    """停止 DAQ 输出（不关闭任务）。"""
    global _running

    with _task_lock:
        _running = False

        if _task is not None:
            try:
                _task.stop()
                print(">>> Task stopped")
            except Exception as e:
                print(f"[Stop Error] {e}")
                raise


def close():
    with _task_lock:
        _close_locked()


def set_sine(channel, amplitude, frequency):
    _shared.set_channel(channel, freq=frequency, amp=amplitude)


def set_phase_diff(channel_a, channel_b, phase_diff_deg):
    _shared.set_phase_diff(channel_a, channel_b, phase_diff_deg)


def generate_waveform(samples: int = BUFFER_SIZE):
    return _shared.generate_waveform(samples=samples)
