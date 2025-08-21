"""
Util functions: get_gpu_temps, run_with_gpu_monitor, check_active_cpu_gpu_processes, run_cmd, run_cmds_parallel
"""
# pip install nvidia-ml-py3 
import time, threading, traceback
from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import os
import psutil
import math
import numpy as np
import pandas as pd
import subprocess, signal, shutil
import sys
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import monotonic
import platform
import socket
import getpass

try:
    import pynvml as nvml
    _NVML = True
except Exception:
    _NVML = False


def get_gpu_temps() -> list[float]:
    """
    Return a list of GPU temperatures (°C) for all visible GPUs.
    """
    nvml.nvmlInit()
    try:
        device_count = nvml.nvmlDeviceGetCount()
        temps = []
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            temp = float(nvml.nvmlDeviceGetTemperature(
                handle, nvml.NVML_TEMPERATURE_GPU
            ))
            temps.append(temp)
        return temps
    finally:
        nvml.nvmlShutdown()

_NVML_THROTTLE_BITS: List[Tuple[int, str]] = [
    (0x00000001, "GPU_IDLE"),
    (0x00000002, "APP_CLOCKS"),
    (0x00000004, "SW_POWER_CAP"),
    (0x00000008, "HW_SLOWDOWN"),
    (0x00000010, "SYNC_BOOST"),
    (0x00000020, "SW_THERMAL"),
    (0x00000040, "HW_THERMAL"),
    (0x00000080, "HW_POWER_BRAKE"),
    (0x00000100, "DISPLAY"),
    # Some NVML builds use a high bit for “UNKNOWN/OTHER”:
    (0x8000000000000000, "UNKNOWN_HIGHBIT"),
]

def _decode_throttle_mask(mask: int) -> List[str]:
    """Return list of human-readable throttle reasons for a NVML bitmask (no NVML required)."""
    reasons = [name for bit, name in _NVML_THROTTLE_BITS if mask & bit]
    # If any unmapped bits remain, expose them as OTHER(0x...)
    covered = 0
    for bit, _ in _NVML_THROTTLE_BITS:
        covered |= bit
    extra = mask & ~covered
    if extra:
        reasons.append(f"OTHER(0x{extra:x})")
    return reasons

def _decode_throttle_mask_flags(mask: int) -> Dict[str, bool]:
    """Return dict[label] = True/False for each known reason; includes OTHER if unknown bits set."""
    flags = {name: bool(mask & bit) for bit, name in _NVML_THROTTLE_BITS}
    covered = 0
    for bit, _ in _NVML_THROTTLE_BITS:
        covered |= bit
    extra = mask & ~covered
    if extra:
        flags[f"OTHER(0x{extra:x})"] = True
    return flags


def _sample_all_gpus(sample_list: List[Dict[str, Any]],
                     stop_evt: threading.Event,
                     interval_s: float = 1.0) -> None:
    """Background sampler that appends per-GPU telemetry snapshots to sample_list."""
    if not _NVML:
        return
    try:
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        while not stop_evt.is_set():
            ts = time.time()
            for idx in range(device_count):
                try:
                    h = nvml.nvmlDeviceGetHandleByIndex(idx)
                    name = nvml.nvmlDeviceGetName(h).decode() if hasattr(nvml, 'nvmlDeviceGetName') else str(idx)
                    temp = nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
                    util = nvml.nvmlDeviceGetUtilizationRates(h).gpu  # %
                    mem_util = nvml.nvmlDeviceGetUtilizationRates(h).memory  # %
                    fan = None
                    try:
                        fan = nvml.nvmlDeviceGetFanSpeed(h)  # %
                    except nvml.NVMLError:
                        pass  # some datacenter GPUs have no fan or report N/A
                    power = None
                    try:
                        power = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # W
                    except nvml.NVMLError:
                        pass
                    sm_clock = None
                    mem_clock = None
                    try:
                        sm_clock = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM)  # MHz
                        mem_clock = nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_MEM)  # MHz
                    except nvml.NVMLError:
                        pass
                    throttle_mask = 0
                    try:
                        throttle_mask = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
                    except nvml.NVMLError:
                        pass
                    sample_list.append({
                        "t": ts,
                        "gpu": idx,
                        "name": name,
                        "temp_C": temp,
                        "util_gpu_pct": util,
                        "util_mem_pct": mem_util,
                        "fan_pct": fan,
                        "power_W": power,
                        "sm_clock_MHz": sm_clock,
                        "mem_clock_MHz": mem_clock,
                        "throttle_mask": throttle_mask,
                        "throttle_flags": ",".join(_decode_throttle_mask(throttle_mask)),
                    })
                except nvml.NVMLError as e:
                    print(f"[ERROR] Failed to sample GPU {idx}: {e}")
            # sleep with responsiveness to stop signal
            stop_evt.wait(interval_s)
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass




def _process_gpu_df(df: pd.DataFrame, sample_interval_s: float) -> Dict[str, str]:
    """
    Summarize GPU telemetry into per-GPU, comma-separated fields.

    Inputs
    ------
    df : DataFrame with columns:
        ['t','gpu','name','temp_C','util_gpu_pct','util_mem_pct','fan_pct',
         'power_W','sm_clock_MHz','mem_clock_MHz','throttle_mask','throttle_flags']
    sample_interval_s : float
        Sampling interval used by the collector (seconds). Used to estimate time spent throttled.

    Outputs
    -------
    Dict[str, str] with keys:
      max_temp, time_to_max, mean_temp, time_to_mean,
      max_util, mean_util, max_fan, mean_fan,
      max_power, mean_power,
      throttled, time_throttled, time_to_throttle,
      throttle_amount   <-- Average % drop in power when throttled (fallback to util if needed)
    Each value is a comma-separated string with one entry per GPU (sorted by GPU index).
    """
    if df is None or len(df) == 0:
        return {k: "" for k in [
            "max_temp","time_to_max","mean_temp","time_to_mean",
            "max_util","mean_util","max_fan","mean_fan",
            "max_power","mean_power","throttled","time_throttled","time_to_throttle",
            "throttle_amount"
        ]}

    # Ensure correct types and sorted order
    d = df.copy()
    d = d.sort_values(["gpu", "t"])
    for col in ["temp_C","util_gpu_pct","fan_pct","power_W"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    # Helper to format numbers consistently
    def fmt(x, nd=1):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return ""
        return f"{x:.{nd}f}"

    # Per-GPU results kept in lists (then joined with commas)
    order = sorted(d["gpu"].dropna().unique().tolist())

    max_temp_vals, time_to_max_vals, mean_temp_vals, time_to_mean_vals = [], [], [], []
    max_util_vals, mean_util_vals = [], []
    max_fan_vals, mean_fan_vals = [], []
    max_power_vals, mean_power_vals = [], []
    throttled_flags, time_throttled_vals, time_to_throttle_vals = [], [], []
    throttle_amount_vals = []

    for g in order:
        gdf = d[d["gpu"] == g].copy()
        if gdf.empty:
            # Append blanks for this GPU slot
            max_temp_vals.append(""); time_to_max_vals.append("")
            mean_temp_vals.append(""); time_to_mean_vals.append("")
            max_util_vals.append(""); mean_util_vals.append("")
            max_fan_vals.append(""); mean_fan_vals.append("")
            max_power_vals.append(""); mean_power_vals.append("")
            throttled_flags.append(""); time_throttled_vals.append(""); time_to_throttle_vals.append("")
            throttle_amount_vals.append("")
            continue

        t0 = float(gdf["t"].iloc[0])

        # --- Temperature
        if "temp_C" in gdf:
            max_temp = float(np.nanmax(gdf["temp_C"].values))
            idx_max = int(np.nanargmax(gdf["temp_C"].values))
            t_max = float(gdf["t"].iloc[idx_max]) - t0

            mean_temp = float(np.nanmean(gdf["temp_C"].values))
            cross_idx = None
            for i, v in enumerate(gdf["temp_C"].values):
                if not math.isnan(v) and v >= mean_temp:
                    cross_idx = i
                    break
            if cross_idx is None:
                cross_idx = len(gdf) - 1
            t_mean = float(gdf["t"].iloc[cross_idx]) - t0
        else:
            max_temp = np.nan; t_max = np.nan; mean_temp = np.nan; t_mean = np.nan

        # --- Utilization (GPU)
        if "util_gpu_pct" in gdf:
            max_util = float(np.nanmax(gdf["util_gpu_pct"].values))
            mean_util = float(np.nanmean(gdf["util_gpu_pct"].values))
        else:
            max_util = np.nan; mean_util = np.nan

        # --- Fan
        if "fan_pct" in gdf:
            max_fan = float(np.nanmax(gdf["fan_pct"].values))
            mean_fan = float(np.nanmean(gdf["fan_pct"].values))
        else:
            max_fan = np.nan; mean_fan = np.nan

        # --- Power
        if "power_W" in gdf:
            max_power = float(np.nanmax(gdf["power_W"].values))
            mean_power = float(np.nanmean(gdf["power_W"].values))
        else:
            max_power = np.nan; mean_power = np.nan

        # --- Throttling masks
        has_throttle_col = "throttle_flags" in gdf
        if has_throttle_col:
            thr_mask = gdf["throttle_flags"].astype(str).str.contains(r"\bSW_THERMAL\b|\bHW_THERMAL\b", regex=True)
            throttled_any = bool(thr_mask.any())
            time_thr = float(thr_mask.sum() * sample_interval_s)
            if throttled_any:
                first_idx = int(np.argmax(thr_mask.values))  # first True
                t_to_thr = float(gdf["t"].iloc[first_idx] - t0)
            else:
                t_to_thr = float("nan")
        else:
            thr_mask = pd.Series(False, index=gdf.index)
            throttled_any = False
            time_thr = float("nan")
            t_to_thr = float("nan")

        # --- Throttle amount (% drop) — prefer POWER; fallback to UTIL
        # Consider only "busy" samples (util >= 30%) to avoid idle bias.
        busy_mask = (gdf.get("util_gpu_pct", pd.Series(np.nan, index=gdf.index)) >= 30.0)

        throttle_amount = None
        if throttled_any:
            # POWER path
            if "power_W" in gdf and gdf["power_W"].notna().any():
                base_samples = gdf.loc[busy_mask & ~thr_mask, "power_W"]
                thr_samples  = gdf.loc[busy_mask & thr_mask,  "power_W"]
                if len(base_samples) >= 3 and len(thr_samples) >= 1 and np.nanmean(base_samples) > 0:
                    baseline_p = float(np.nanmean(base_samples))
                    throttled_p = float(np.nanmean(thr_samples))
                    throttle_amount = max(0.0, (baseline_p - throttled_p) / baseline_p * 100.0)
            # UTIL fallback
            if throttle_amount is None and "util_gpu_pct" in gdf and gdf["util_gpu_pct"].notna().any():
                base_u = gdf.loc[busy_mask & ~thr_mask, "util_gpu_pct"]
                thr_u  = gdf.loc[busy_mask & thr_mask,  "util_gpu_pct"]
                if len(base_u) >= 3 and len(thr_u) >= 1 and np.nanmean(base_u) > 0:
                    baseline_u = float(np.nanmean(base_u))
                    throttled_u = float(np.nanmean(thr_u))
                    throttle_amount = max(0.0, (baseline_u - throttled_u) / baseline_u * 100.0)

        # Append formatted values
        def add_or_blank(val, nd=1, cond=True):
            return fmt(val, nd) if (cond and val is not None and not (isinstance(val, float) and math.isnan(val))) else ""

        max_temp_vals.append(add_or_blank(max_temp, 1))
        time_to_max_vals.append(add_or_blank(t_max, 2))
        mean_temp_vals.append(add_or_blank(mean_temp, 1))
        time_to_mean_vals.append(add_or_blank(t_mean, 2))

        max_util_vals.append(add_or_blank(max_util, 1))
        mean_util_vals.append(add_or_blank(mean_util, 1))

        max_fan_vals.append(add_or_blank(max_fan, 1))
        mean_fan_vals.append(add_or_blank(mean_fan, 1))

        max_power_vals.append(add_or_blank(max_power, 1))
        mean_power_vals.append(add_or_blank(mean_power, 1))

        throttled_flags.append("true" if throttled_any else "false")
        time_throttled_vals.append(add_or_blank(time_thr, 2, throttled_any))
        time_to_throttle_vals.append(add_or_blank(t_to_thr, 2, throttled_any))

        throttle_amount_vals.append(add_or_blank(throttle_amount, 1, throttled_any))

    # Build output dict with comma-separated strings
    out = {
        "max_temp": ",".join(max_temp_vals),
        "time_to_max": ",".join(time_to_max_vals),
        "mean_temp": ",".join(mean_temp_vals),
        "time_to_mean": ",".join(time_to_mean_vals),
        "max_util": ",".join(max_util_vals),
        "mean_util": ",".join(mean_util_vals),
        "max_fan": ",".join(mean_fan_vals) if False else ",".join(max_fan_vals),  # keep both lines if you prefer swap
        "mean_fan": ",".join(mean_fan_vals),
        "max_power": ",".join(max_power_vals),
        "mean_power": ",".join(mean_power_vals),
        "throttled": ",".join(throttled_flags),
        "time_throttled": ",".join(time_throttled_vals),
        "time_to_throttle": ",".join(time_to_throttle_vals),
        "throttle_amount": ",".join(throttle_amount_vals),
    }
    return out


# NEW: aggregate CPU sampler (averages across all CPUs + RAM)
def _sample_cpu_agg(sample_list: List[Dict[str, Any]],
                    stop_evt: threading.Event,
                    interval_s: float = 1.0,
                    active_core_threshold_pct: float = 5.0) -> None:
    """
    Background sampler that appends system CPU/RAM telemetry snapshots to sample_list.
    Each record has:
      {
        "t": ts,
        "cpu_avg_util_pct": float,           # average across all logical CPUs
        "cpu_active_count": int,             # #logical CPUs with util >= threshold
        "cpu_count": int,                    # total logical CPUs
        "cpu_temp_avg_C": float|nan,         # average of all reported CPU temp sensors
        "chipset_temp_avg_C": float|nan,     # average of all reported chipset temp sensors
        "ram_used_GB": float,
      }
    """
    try:
        import psutil
    except Exception:
        # psutil not available; do nothing
        return

    # Prime cpu_percent so subsequent calls report interval deltas
    try:
        psutil.cpu_percent(interval=None, percpu=True)
    except Exception:
        pass

    while not stop_evt.is_set():
        ts = time.time()
        try:
            # CPU utilization
            cpu_per = psutil.cpu_percent(interval=None, percpu=True) or []
            cpu_count = len(cpu_per) if isinstance(cpu_per, list) else 0
            cpu_avg = float(np.nanmean(cpu_per)) if cpu_count else np.nan
            cpu_active = int(sum(1 for v in cpu_per if v is not None and v >= active_core_threshold_pct))

            cpu_temp = np.nan
            chipset_temp = np.nan
            try:
                temps = psutil.sensors_temperatures(fahrenheit=False) or {}
                vals = []
                vals_chipset = []
                for _, entries in temps.items():
                    for e in entries:
                        if e.current is not None and e.label == 'cpu':
                            vals.append(float(e.current))
                        if e.current is not None and e.label == 'chipset':
                            vals_chipset.append(float(e.current))
                if vals:
                    cpu_temp = float(np.nanmean(vals))
                if vals_chipset:
                    chipset_temp = float(np.nanmean(vals_chipset))
            except Exception:
                pass

            # RAM
            vm = None
            try:
                vm = psutil.virtual_memory()
            except Exception:
                pass
            ram_total = float(vm.total) / (1024**3) if vm else np.nan
            ram_used = float(vm.used) / (1024**3) if vm else np.nan
            ram_pct = float(vm.percent) if vm else np.nan

            sample_list.append({
                "t": ts,
                "cpu_avg_util_pct": cpu_avg,
                "cpu_active_count": cpu_active,
                "cpu_count": cpu_count,
                "cpu_temp_avg_C": cpu_temp,
                "chipset_temp_avg_C": chipset_temp,
                "ram_used_GB": ram_used,
            })
        except Exception:
            pass
        stop_evt.wait(interval_s)



def _process_cpu_df(df: pd.DataFrame, sample_interval_s: float) -> Dict[str, str]:
    """
    Inputs: DataFrame with columns:
      ['t','cpu_avg_util_pct','cpu_active_count','cpu_count','cpu_temp_avg_C',
       'ram_used_GB']
    Outputs (single series, not per-CPU):
      cpu_max_temp, cpu_time_to_max, cpu_mean_temp,
      cpu_max_util, cpu_mean_util,
      cpu_max_active, cpu_mean_active,
      ram_max_gb, ram_mean_gb
    Values are strings to match GPU summary style.
    """
    def fmt(x, nd=1):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return ""
        return f"{x:.{nd}f}"

    if df is None or df.empty:
        return {
            "cpu_max_temp": "", "cpu_time_to_max": "", "cpu_mean_temp": "",
            "cpu_max_util": "", "cpu_mean_util": "",
            "cpu_max_active": "", "cpu_mean_active": "",
            "ram_max_gb": "", "ram_mean_gb": ""
        }

    d = df.copy().sort_values("t")

    # Temp
    temp_arr = pd.to_numeric(d.get("cpu_temp_avg_C"), errors="coerce").values
    if temp_arr.size and not np.all(np.isnan(temp_arr)):
        cpu_max_temp = float(np.nanmax(temp_arr))
        idx_max = int(np.nanargmax(temp_arr))
        t0 = float(d["t"].iloc[0])
        cpu_time_to_max = float(d["t"].iloc[idx_max] - t0)
        cpu_mean_temp = float(np.nanmean(temp_arr))
    else:
        cpu_max_temp = np.nan; cpu_time_to_max = np.nan; cpu_mean_temp = np.nan

    # Util
    util_arr = pd.to_numeric(d.get("cpu_avg_util_pct"), errors="coerce").values
    cpu_max_util = float(np.nanmax(util_arr)) if util_arr.size else np.nan
    cpu_mean_util = float(np.nanmean(util_arr)) if util_arr.size else np.nan

    # Active CPU count
    act_arr = pd.to_numeric(d.get("cpu_active_count"), errors="coerce").values
    cpu_max_active = float(np.nanmax(act_arr)) if act_arr.size else np.nan
    cpu_mean_active = float(np.nanmean(act_arr)) if act_arr.size else np.nan

    # RAM
    ram_used = pd.to_numeric(d.get("ram_used_GB"), errors="coerce").values
    ram_max_gb = float(np.nanmax(ram_used)) if ram_used.size else np.nan
    ram_mean_gb = float(np.nanmean(ram_used)) if ram_used.size else np.nan

    return {
        "cpu_max_temp": fmt(cpu_max_temp, 1),
        "cpu_time_to_max": fmt(cpu_time_to_max, 2),
        "cpu_mean_temp": fmt(cpu_mean_temp, 1),
        "cpu_max_util": fmt(cpu_max_util, 1),
        "cpu_mean_util": fmt(cpu_mean_util, 1),
        "cpu_max_active": fmt(cpu_max_active, 0),
        "cpu_mean_active": fmt(cpu_mean_active, 1),
        "ram_max_gb": fmt(ram_max_gb, 2),
        "ram_mean_gb": fmt(ram_mean_gb, 2),
    }


def run_with_gpu_monitor(func: Callable, *args,
                         poll_interval_s: float = 1.0,
                         as_dataframe: bool = True,
                         **kwargs) -> Tuple[Any, Any]:
    """
    Run `func(*args, **kwargs)` while sampling NVIDIA GPUs and system CPU/RAM.
    Returns: (func_result, metrics_dict) where metrics_dict merges GPU and CPU summaries.
    """
    gpu_samples: List[Dict[str, Any]] = []
    cpu_samples: List[Dict[str, Any]] = []

    stop_evt = threading.Event()
    gpu_thread = cpu_thread = None
    exc_info = None
    result = None

    if not _NVML:
        print("[WARN] pynvml not available; install with `pip install nvidia-ml-py3` for GPU telemetry.")
    try:
        # start samplers
        if _NVML:
            gpu_thread = threading.Thread(
                target=_sample_all_gpus,
                args=(gpu_samples, stop_evt, poll_interval_s),
                daemon=True,
            )
            gpu_thread.start()

        cpu_thread = threading.Thread(
            target=_sample_cpu_agg,
            args=(cpu_samples, stop_evt, poll_interval_s),
            daemon=True,
        )
        cpu_thread.start()

        # run target
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exc_info = (e, traceback.format_exc())
        t1 = time.time()

        time.sleep(0.01)  # allow final tick
    finally:
        stop_evt.set()
        if gpu_thread is not None:
            gpu_thread.join(timeout=2.0)
        if cpu_thread is not None:
            cpu_thread.join(timeout=2.0)

    # Build summaries
    summary: Dict[str, str] = {}

    # GPU summary (existing flow)
    if gpu_samples and pd is not None and as_dataframe:
        gpu_df = pd.DataFrame(gpu_samples)
        summary.update(_process_gpu_df(gpu_df, poll_interval_s))
    elif gpu_samples:
        # if no pandas, still attempt using existing helper (it expects a DataFrame)
        gpu_df = pd.DataFrame(gpu_samples) if pd is not None else None
        summary.update(_process_gpu_df(gpu_df if gpu_df is not None else pd.DataFrame([]), poll_interval_s))
    else:
        summary.update(_process_gpu_df(pd.DataFrame([]), poll_interval_s))

    # CPU summary
    if cpu_samples and pd is not None and as_dataframe:
        cpu_df = pd.DataFrame(cpu_samples)
        summary.update(_process_cpu_df(cpu_df, poll_interval_s))
    elif cpu_samples:
        # minimal fallback without pandas
        # convert to DataFrame-like with numpy ops
        try:
            t = [rec.get("t") for rec in cpu_samples]
            temp = np.array([rec.get("cpu_temp_avg_C", np.nan) for rec in cpu_samples], dtype=float)
            util = np.array([rec.get("cpu_avg_util_pct", np.nan) for rec in cpu_samples], dtype=float)
            # active = np.array([rec.get("cpu_active_count", np.nan) for rec in cpu_samples], dtype=float)
            ram = np.array([rec.get("ram_used_GB", np.nan) for rec in cpu_samples], dtype=float)
            def fmt(x, nd=1):
                if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                    return ""
                return f"{x:.{nd}f}"
            if len(t) and not np.all(np.isnan(temp)):
                idx = int(np.nanargmax(temp))
                t0 = float(t[0])
                cpu_time_to_max = float(t[idx] - t0)
            else:
                cpu_time_to_max = np.nan
            summary.update({
                "cpu_max_temp": fmt(np.nanmax(temp), 1),
                "cpu_time_to_max": fmt(cpu_time_to_max, 2),
                "cpu_mean_temp": fmt(np.nanmean(temp), 1),
                "cpu_max_util": fmt(np.nanmax(util), 1),
                "cpu_mean_util": fmt(np.nanmean(util), 1),
                "ram_max_gb": fmt(np.nanmax(ram), 1),
                "ram_mean_gb": fmt(np.nanmean(ram), 1),
            })
        except Exception:
            summary.update({
                "cpu_max_temp": "", "cpu_time_to_max": "", "cpu_mean_temp": "",
                "cpu_max_util": "", "cpu_mean_util": "",
                "ram_max_gb": "", "ram_mean_gb": "", 
            })
    else:
        summary.update({
            "cpu_max_temp": "", "cpu_time_to_max": "", "cpu_mean_temp": "",
            "cpu_max_util": "", "cpu_mean_util": "",
            "ram_max_gb": "", "ram_mean_gb": ""
        })

    # Attach run meta (consistent with earlier behavior)
    summary["run_duration_s"] = f"{(t1 - t0):.3f}"

    if exc_info is not None:
        e, tb = exc_info
        print("[ERROR] Target function raised an exception; returning metrics alongside.\n" + tb)
        raise e

    return result, summary



def _username_for_pid(pid: int) -> Optional[str]:
    try:
        return psutil.Process(pid).username()
    except Exception:
        return None


def _cmd_for_pid(pid: int) -> str:
    try:
        p = psutil.Process(pid)
        cmd = " ".join(p.cmdline())
        return cmd if cmd else p.name()
    except Exception:
        return ""


def check_active_cpu_gpu_processes(
    cpu_threshold_pct: float = 20.0,
    gpu_mem_threshold_mb: float = 50.0,
    include_self: bool = True,
    include_system_users: bool = False,
    cpu_sample_seconds: float = 0.25,
) -> Dict[str, Any]:
    """
    Return a dict with active CPU and GPU processes.

    cpu_threshold_pct: minimum per-process CPU% to report
    gpu_mem_threshold_mb: minimum per-process GPU memory (MiB) to report
    include_self: include the current Python process in the results
    include_system_users: if False, filter out obvious system users (root, gdm, etc.)
    cpu_sample_seconds: sampling window for per-process CPU% (approx)
    """
    me = os.getpid()
    # system_users = {"root", "gdm", "systemd-timesync", "systemd-network", "systemd-resolve"}
    non_system_users = [i for i in os.listdir('/home')]
    # print(non_system_users)

    # ----- CPU: two-pass sampling so cpu_percent is meaningful
    procs = [p for p in psutil.process_iter(attrs=['pid','username']) if p.is_running()]
    for p in procs:
        try:
            p.cpu_percent(None)  # prime
        except Exception:
            pass
    time.sleep(cpu_sample_seconds)

    cpu_entries: List[Dict[str, Any]] = []
    for p in procs:
        try:
            if not p.is_running():
                continue
            pid = p.pid
            if not include_self and pid == me:
                continue
            user = p.info.get('username') or _username_for_pid(pid) or "unknown"
            if not include_system_users and user not in non_system_users:
                continue
            cpu_pct = p.cpu_percent(None)  # % since last call
            if cpu_pct < cpu_threshold_pct:
                continue
            mem = p.memory_info().rss if p.is_running() else 0
            cpu_entries.append({
                "pid": pid,
                "user": user,
                "cpu_percent": cpu_pct,
                "rss_bytes": mem,
                "cmd": _cmd_for_pid(pid),
            })
        except Exception:
            continue

    cpu_entries.sort(key=lambda x: x["cpu_percent"], reverse=True)
    # get rid of any entries by system users
    cpu_entries = [e for e in cpu_entries if e["user"] in non_system_users]
    cpu_busy = len(cpu_entries) > 0

    # ----- GPU: NVML enumerate running processes per GPU
    gpu_entries: List[Dict[str, Any]] = []
    if _NVML:
        try:
            nvml.nvmlInit()
            n = nvml.nvmlDeviceGetCount()
            for i in range(n):
                h = nvml.nvmlDeviceGetHandleByIndex(i)
                name = nvml.nvmlDeviceGetName(h).decode() if hasattr(nvml, 'nvmlDeviceGetName') else f"GPU{i}"

                # Compute contexts
                try:
                    procs_compute = nvml.nvmlDeviceGetComputeRunningProcesses(h)
                except Exception:
                    procs_compute = []

                # Graphics contexts (rare on servers but include just in case)
                try:
                    procs_graphics = nvml.nvmlDeviceGetGraphicsRunningProcesses(h)
                except Exception:
                    procs_graphics = []
                for rec in list(procs_compute) + list(procs_graphics):
                    pid = int(rec.pid)
                    # print(f"GPU {i} process: {pid} {rec.usedGpuMemory}")
                    if not include_self and pid == me:
                        # print(f"\tSkipping self process: {pid}")
                        continue
                    used_mb = float(getattr(rec, "usedGpuMemory", 0) or 0) / (1024 * 1024)
                    if used_mb < gpu_mem_threshold_mb:
                        # print(f"\tSkipping low GPU memory process: {pid} {used_mb}MB")
                        continue
                    user = _username_for_pid(pid) or "unknown"
                    # skip if command has xorg in it
                    if "xorg" in _cmd_for_pid(pid).lower():
                        # print(f"\tSkipping xorg process: {pid}")
                        continue
                    if not include_system_users and user not in non_system_users:
                        # print(f"\tSkipping system user process: {pid} {user}")
                        continue
                    gpu_entries.append({
                        "gpu_index": i,
                        "gpu_name": name,
                        "pid": pid,
                        "user": user,
                        "gpu_mem_used_mb": round(used_mb, 1),
                        "cmd": _cmd_for_pid(pid),
                        "type": "compute" if rec in procs_compute else "graphics",
                    })
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
    else:
        raise ImportError("NVML not available – pip install nvidia-ml-py3")

    # Aggregate flags
    gpu_busy = len(gpu_entries) > 0

    return {
        "cpu_busy": cpu_busy,
        "gpu_busy": gpu_busy,
        "cpu_processes": cpu_entries,
        "gpu_processes": gpu_entries,
        "notes": (
            "Install NVML with `pip install nvidia-ml-py3` to enable GPU process listing."
            if not _NVML else ""
        ),
    }



def _reader(pipe, sink, buf):
    try:
        for line in iter(pipe.readline, ''):
            if sink:
                sink.write(line)
                sink.flush()
            buf.write(line)
    except (ValueError, OSError):
        pass  # pipe closed while reading

def run_cmd(
    cmd: str,
    timeout: Optional[float] = None,
    stream: bool = False,
    logfile: Optional[str] = None,
):
    """
    Run a shell command string under bash, optionally stream live,
    capture full stdout/stderr, and kill the process group on timeout or Ctrl-C.

    Parameters
    ----------
    cmd : str
        Command string to run.
    timeout : float, optional
        Kill the process if it runs longer than this many seconds.
    stream : bool, default False
        If True, stream output live to sys.stdout/sys.stderr.
    logfile : str, optional
        Path to a log file. If given, the command string and combined
        outputs will be appended at the end of execution, separated by
        two newlines.

    Returns
    -------
    dict
        {stdout, stderr, returncode, timed_out}
    """
    if stream and shutil.which("stdbuf"):
        cmd_to_run = f"stdbuf -oL -eL {cmd}"
    else:
        cmd_to_run = cmd

    proc = subprocess.Popen(
        cmd_to_run,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1 if stream else -1,
        preexec_fn=os.setsid,
    )

    out_buf, err_buf = StringIO(), StringIO()
    t_out = t_err = None
    if stream:
        t_out = threading.Thread(target=_reader, args=(proc.stdout, sys.stdout, out_buf), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, sys.stderr, err_buf), daemon=True)
        t_out.start(); t_err.start()

    timed_out = False
    start = monotonic()
    try:
        if timeout is None:
            proc.wait()
        else:
            remaining = timeout
            while proc.poll() is None and remaining > 0:
                try:
                    proc.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
                remaining = timeout - (monotonic() - start)
            if proc.poll() is None:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        proc.wait()
        raise
    finally:
        if stream:
            if t_out: t_out.join(timeout=1)
            if t_err: t_err.join(timeout=1)
            try:
                if proc.stdout: proc.stdout.close()
                if proc.stderr: proc.stderr.close()
            except Exception:
                pass

    if not stream:
        out, err = proc.communicate()
        out_buf.write(out or "")
        err_buf.write(err or "")

    result = {
        "stdout": out_buf.getvalue(),
        "stderr": err_buf.getvalue() + ("\n[terminated due to timeout]" if timed_out else ""),
        "returncode": proc.returncode if proc.returncode is not None else -9,
        "timed_out": timed_out,
    }

    # --- Logging ---
    if logfile:
        try:
            os.makedirs(os.path.dirname(logfile), exist_ok=True)
        except Exception:
            pass  # dirname may be empty (current dir)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write("\n\n")
            f.write(cmd + "\n")
            # Append stdout + stderr together
            f.write(result["stdout"])
            if result["stderr"]:
                f.write('STDERR:')
                f.write(result["stderr"])
        # ensures the log file always has command + output in one block

    return result




def run_cmds_parallel(
    cmds: list[str],
    timeout: Optional[float] = None,
    stream: bool = False,
    logfile: Optional[str] = None
):
    """
    Run one or more shell commands in parallel using run_cmd.

    Behavior
    --------
    - If only one command is given:
        Just calls run_cmd(cmd, timeout=..., stream=...) and returns its result dict.
    - If multiple commands:
        Runs all concurrently, forces stream=False to avoid interleaved output,
        and returns a list of result dicts in the same order as `cmds`.

    Parameters
    ----------
    cmds : list[str]
        Command strings to execute.
    timeout : float or None, optional
        Max time (in seconds) allowed for each command. Passed to run_cmd.
        None means no timeout.
    stream : bool, default False
        Whether to stream output to stdout/stderr in real time.
        Only effective if len(cmds) == 1. For multiple commands, stream is
        automatically disabled to avoid interleaved output.

    Returns
    -------
    dict or list[dict]
        - If cmds has length 1: returns the single run_cmd result dict.
        - If cmds has length >1: returns a list of result dicts, in input order.
    """
    if not isinstance(cmds, (list, tuple)):
        raise TypeError("cmds must be a list/tuple of command strings")
    if len(cmds) == 0:
        return []

    # Single command -> just call run_cmd
    if len(cmds) == 1:
        return run_cmd(cmds[0], timeout=timeout, stream=stream, logfile=logfile)

    # Multiple commands -> parallel run, force stream=False
    results = [None] * len(cmds)
    with ThreadPoolExecutor(max_workers=len(cmds)) as ex:
        futs = {
            ex.submit(run_cmd, cmd, timeout=timeout, stream=False, logfile=logfile): i
            for i, cmd in enumerate(cmds)
        }
        try:
            for fut in as_completed(futs):
                idx = futs[fut]
                results[idx] = fut.result()
        except KeyboardInterrupt:
            for f in futs:
                f.cancel()
            raise

    return results

def get_num_gpus() -> int:
    """
    Detect the number of NVIDIA GPUs available on the system.

    Returns
    -------
    int
        Number of GPUs detected. Returns 0 if none are found or if NVML is unavailable.
    """
    try:
        nvml.nvmlInit()
        count = nvml.nvmlDeviceGetCount()
        nvml.nvmlShutdown()
        return count
    except Exception:
        return 0
    

def collect_machine_info():
    import os, platform, socket, getpass, psutil, subprocess

    info = {}

    # Optional libs (scoped to this function)
    _CPUINFO = False
    _NVML = False
    cpuinfo = None
    nvml = None
    try:
        import cpuinfo as _cpuinfo
        cpuinfo = _cpuinfo
        _CPUINFO = True
    except Exception:
        pass
    try:
        import pynvml as _nvml
        nvml = _nvml
        _NVML = True
    except Exception:
        pass

    # Host/user
    info["hostname"] = socket.gethostname()
    info["user"] = getpass.getuser()

    # CPUs
    # CPU name (best-effort cross-platform)
    cpu_name = None
    if _CPUINFO:
        try:
            cpu_name = cpuinfo.get_cpu_info().get("brand_raw")
        except Exception:
            cpu_name = None
    if not cpu_name:
        sysname = platform.system()
        if sysname == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass
            if not cpu_name:
                try:
                    out = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
                    for ln in out.splitlines():
                        if ln.lower().startswith("model name:"):
                            cpu_name = ln.split(":", 1)[1].strip()
                            break
                except Exception:
                    pass
        elif sysname == "Darwin":
            try:
                out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True, check=False)
                cpu_name = out.stdout.strip() or None
            except Exception:
                pass
        elif sysname == "Windows":
            try:
                out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, check=False)
                lines = [l.strip() for l in (out.stdout or "").splitlines() if l.strip()]
                if len(lines) >= 2:
                    cpu_name = lines[1]
            except Exception:
                pass
            if not cpu_name:
                try:
                    out = subprocess.run(
                        ["reg", "query", r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "/v", "ProcessorNameString"],
                        capture_output=True, text=True, check=False
                    )
                    for ln in (out.stdout or "").splitlines():
                        if "ProcessorNameString" in ln:
                            cpu_name = ln.split("REG_SZ", 1)[1].strip()
                            break
                except Exception:
                    pass
    if not cpu_name:
        cpu_name = platform.processor() or platform.uname().processor or "Unknown"

    info["num_cpus"] = os.cpu_count()
    info["cpu_name"] = cpu_name
    info["architecture"] = platform.machine() or ""

    # RAM (GB)
    info["ram_size"] = round(psutil.virtual_memory().total / (1024**3), 2)

    # OS (pretty) + kernel
    pretty_os = None
    sysname = platform.system()
    if sysname == "Linux":
        # Try /etc/os-release first
        try:
            data = {}
            with open("/etc/os-release") as f:
                for ln in f:
                    if "=" in ln:
                        k, v = ln.strip().split("=", 1)
                        data[k] = v.strip().strip('"')
            pretty_os = data.get("PRETTY_NAME") or " ".join(
                [data.get("NAME", ""), data.get("VERSION", "") or data.get("VERSION_ID", "")]
            ).strip()
        except Exception:
            pass
        if not pretty_os:
            # Optional: distro module, if installed
            try:
                import distro
                pretty_os = f"{distro.name(pretty=True)} {distro.version(best=True)}".strip()
            except Exception:
                pass
    elif sysname == "Darwin":
        try:
            out = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True, check=False)
            ver = out.stdout.strip() or platform.mac_ver()[0]
            pretty_os = f"macOS {ver}".strip()
        except Exception:
            pretty_os = "macOS"
    elif sysname == "Windows":
        # Keep it simple and reliable
        pretty_os = f"Windows {platform.version()}"

    info["os"] = pretty_os or f"{platform.system()} {platform.release()}"
    info["kernel"] = f"{platform.system()} {platform.release()}"

    # Python
    info["python_version"] = platform.python_version()

    # GPUs (NVIDIA via NVML)
    info["num_gpus"] = 0
    info["gpu_names"] = ""
    info["nvidia_driver_version"] = ""
    info["cuda_driver_version"] = ""   # e.g., "12.4"
    info["cuda_runtime_version"] = ""  # from nvcc if present

    if _NVML:
        try:
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            info["num_gpus"] = device_count
            gpu_names = []
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                name = nvml.nvmlDeviceGetName(handle)
                name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
                gpu_names.append(name)
            info["gpu_names"] = ", ".join(gpu_names)

            # NVIDIA driver version (e.g., "550.54.14")
            try:
                drv = nvml.nvmlSystemGetDriverVersion()
                info["nvidia_driver_version"] = drv.decode("utf-8") if isinstance(drv, bytes) else str(drv)
            except Exception:
                pass

            # CUDA driver version as int (e.g., 12040 => 12.4)
            try:
                if hasattr(nvml, "nvmlSystemGetCudaDriverVersion_v2"):
                    v = nvml.nvmlSystemGetCudaDriverVersion_v2()
                else:
                    v = nvml.nvmlSystemGetCudaDriverVersion()
                if isinstance(v, int) and v > 0:
                    major, minor = divmod(v, 1000)
                    minor = minor // 10
                    info["cuda_driver_version"] = f"{major}.{minor}"
            except Exception:
                pass

        except Exception:
            info["gpu_names"] = "Error retrieving GPU info"
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass

    # CUDA runtime version via nvcc (if installed)
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=False)
        txt = (out.stdout or "") + "\n" + (out.stderr or "")
        # Heuristic: find 'release X.Y'
        cand = ""
        for ln in txt.splitlines():
            if "release" in ln.lower():
                parts = ln.replace(",", " ").split()
                # find first token that looks like X.Y
                for tok in parts:
                    if tok.count(".") == 1:
                        a, b = tok.split(".")
                        if a.isdigit() and b.isdigit():
                            cand = tok
                            break
                if cand:
                    break
        if cand:
            info["cuda_runtime_version"] = cand
    except Exception:
        pass

    return info
