"""Copy of the modified root ``modulepack.py`` for integration reference.

The existing GUI imports the root copy.  Keep both copies synchronized when
changing DAQ integration behavior.
"""

import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np


ARTDAQ_LIB_PATH = Path(__file__).resolve().parents[2] / "LIB"

if ARTDAQ_LIB_PATH.exists() and str(ARTDAQ_LIB_PATH) not in sys.path:
    sys.path.insert(0, str(ARTDAQ_LIB_PATH))

import artdaq
from artdaq.constants import AcquisitionType, RegenerationMode


DEVICE_NAME = "Dev1"

NUM_CHANNELS = 3
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
        self.amp = [0.0] * NUM_CHANNELS
        self.phase_rad = [0.0] * NUM_CHANNELS
        self.constvolt = [0.0] * NUM_CHANNELS

        self.generator = self._generate_waveform

    def check_channel(self, ch: int):
        if not isinstance(ch, int):
            raise TypeError("Channel index must be int")

        if ch < 0 or ch >= NUM_CHANNELS:
            raise ValueError(f"Channel must be between 0 and {NUM_CHANNELS - 1}")

    def set_channel(self, ch: int, freq=None, amp=None, constvolt=None):
        self.check_channel(ch)

        with self.lock:
            if freq is not None:
                freq = float(freq)
                if freq < MIN_FREQUENCY or freq > MAX_FREQUENCY:
                    raise ValueError(
                        f"Frequency must be between "
                        f"{MIN_FREQUENCY} and {MAX_FREQUENCY} Hz"
                    )
                self.freq[ch] = freq

            if amp is not None:
                amp = float(amp)
                if amp < 0.0 or amp > MAX_AMPLITUDE:
                    raise ValueError(
                        f"MAX_VOLT must be between 0 and {MAX_AMPLITUDE} V"
                    )
                self.amp[ch] = amp

            if constvolt is not None:
                constvolt = float(constvolt)
                if constvolt < AO_MIN_VAL or constvolt > AO_MAX_VAL:
                    raise ValueError(
                        f"Constant voltage must be between "
                        f"{AO_MIN_VAL} and {AO_MAX_VAL} V"
                    )
                self.constvolt[ch] = constvolt

    def set_phase_diff(self, ch_ref: int, ch_target: int, diff_deg: float):
        self.check_channel(ch_ref)
        self.check_channel(ch_target)

        if ch_ref == ch_target:
            raise ValueError("Reference and target channels must be different")

        diff_rad = np.deg2rad(float(diff_deg))

        with self.lock:
            self.phase_rad[ch_target] = self.phase_rad[ch_ref] + diff_rad
            self.phase_rad[ch_target] %= 2.0 * np.pi

    def generate(self, samples: int = BUFFER_SIZE):
        with self.lock:
            return self.generator(samples)

    def set_generator(self, generator):
        with self.lock:
            self.generator = generator

    def _generate_waveform(self, samples: int):
        freq = np.array(self.freq)
        amp = np.array(self.amp)
        phase = np.array(self.phase_rad)

        n = np.arange(samples, dtype=np.float64)
        omega = 2.0 * np.pi * freq / SAMPLE_RATE

        data = amp[:, None] * np.sin(
            phase[:, None] + omega[:, None] * n[None, :]
        )

        self.phase_rad = list((phase + omega * samples) % (2.0 * np.pi))
        return data

    def _generate_constantform(self, samples: int):
        constvolt = np.array(self.constvolt)
        return np.repeat(constvolt[:, None], samples, axis=1)


_shared = SharedParams()
_task = None
_running = False
_task_lock = threading.RLock()
_callback_ref = None
_source_lock = threading.RLock()
_voltage_source: Callable[[int], np.ndarray] = _shared.generate
_voltage_source_name = "manual"
_output_fault = None


def _zero_voltage_block(samples: int):
    return np.zeros((NUM_CHANNELS, samples), dtype=np.float64)


def _source_display_name(source) -> str:
    return getattr(
        source,
        "__qualname__",
        getattr(source, "__name__", repr(source)),
    )


def _latch_output_fault(message: str):
    global _output_fault

    with _source_lock:
        if _output_fault is None:
            _output_fault = str(message)
            print(f"[Output Fault] {_output_fault}")


def _validate_voltage_block(data, samples: int):
    array = np.asarray(data, dtype=np.float64)
    expected_shape = (NUM_CHANNELS, samples)
    if array.shape != expected_shape:
        raise ValueError(
            f"Voltage source returned shape {array.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Voltage source returned NaN or infinite values")

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < AO_MIN_VAL - 1e-12 or maximum > AO_MAX_VAL + 1e-12:
        raise ValueError(
            f"Voltage source exceeded AO range [{AO_MIN_VAL}, {AO_MAX_VAL}] V: "
            f"min={minimum:.6g} V, max={maximum:.6g} V"
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def _generate_voltage_block(samples: int, fail_fast: bool = False):
    with _source_lock:
        source = _voltage_source
        source_name = _voltage_source_name
        existing_fault = _output_fault

    if existing_fault is not None:
        if fail_fast:
            raise RuntimeError(f"Output is faulted: {existing_fault}")
        return _zero_voltage_block(samples)

    try:
        return _validate_voltage_block(source(samples), samples)
    except Exception as exc:
        message = f"source '{source_name}' failed: {exc}"
        _latch_output_fault(message)
        if fail_fast:
            raise RuntimeError(message) from exc
        return _zero_voltage_block(samples)


def _transfer_callback(task_handle, event_type, num_samples, callback_data):
    global _task, _running, _task_lock

    try:
        if not _running:
            return 0

        data = _generate_voltage_block(BUFFER_SIZE)

        with _task_lock:
            if _task is not None and _running:
                _task.write(data, auto_start=False)
    except Exception as exc:
        print(f"[Callback Error] {exc}")

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
        except Exception as exc:
            print(f"[Close Error] {exc}")
            raise
        finally:
            _task = None

    _callback_ref = None


def start():
    global _task, _running, _callback_ref

    with _task_lock:
        _close_locked()
        task = artdaq.Task("AOWave")

        try:
            for index in range(NUM_CHANNELS):
                task.ao_channels.add_ao_voltage_chan(
                    f"{DEVICE_NAME}/ao{index}",
                    name_to_assign_to_channel=f"ch{index}",
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
                [
                    _generate_voltage_block(BUFFER_SIZE, fail_fast=True)
                    for _ in range(PREFILL_BUFFERS)
                ],
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
    global _running

    with _task_lock:
        _running = False
        if _task is not None:
            try:
                _task.stop()
                print(">>> Task stopped")
            except Exception as exc:
                print(f"[Stop Error] {exc}")
                raise


def close():
    with _task_lock:
        _close_locked()


def set_sine(channel, amplitude, frequency):
    _shared.set_channel(channel, freq=frequency, amp=amplitude)


def set_constant(channel, voltage):
    _shared.set_channel(channel, constvolt=voltage)


def set_mode(mode: str):
    if mode == "sine":
        _shared.set_generator(_shared._generate_waveform)
    elif mode == "constant":
        _shared.set_generator(_shared._generate_constantform)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def set_phase_diff(channel_a, channel_b, phase_diff_deg):
    _shared.set_phase_diff(channel_a, channel_b, phase_diff_deg)


def set_voltage_source(source: Callable[[int], np.ndarray]):
    """Select a custom ``source(samples) -> (3, samples)`` provider."""

    global _voltage_source, _voltage_source_name, _output_fault

    if not callable(source):
        raise TypeError("Voltage source must be callable")
    with _task_lock:
        if _running:
            raise RuntimeError("Stop DAQ output before changing voltage source")
        with _source_lock:
            _voltage_source = source
            _voltage_source_name = _source_display_name(source)
            _output_fault = None


def use_manual_voltage_source():
    global _voltage_source_name

    set_voltage_source(_shared.generate)
    with _source_lock:
        _voltage_source_name = "manual"


def get_output_fault():
    with _source_lock:
        return _output_fault


def clear_output_fault():
    global _output_fault

    with _task_lock:
        if _running:
            raise RuntimeError("Stop DAQ output before clearing an output fault")
        with _source_lock:
            _output_fault = None


def generate(samples: int = BUFFER_SIZE):
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
        raise TypeError("samples must be an integer")
    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be positive")
    return _generate_voltage_block(samples)
