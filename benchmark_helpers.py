"""
All functions have signature: run_<fn>(gpus : list) -> dict[str]->value. If multiple are passed, it will default to 
running the same benchmark in parallel on both gpus (note, NOT using multiprocessing), 
and will report the mean across all gpus. For benchmarks that support
multiprocessing (currently only the dataloader IO ones), pass e.g. ['0, 1']
"""


from utils import run_cmd, run_cmds_parallel, get_num_gpus
import shutil, shlex
import pandas as pd
import os
import json
import numpy as np
import re
from typing import Tuple
import uuid

def _parse_gpu_spec(gpu_spec: str):
    """Parse a CUDA_VISIBLE_DEVICES-like string into a de-duplicated, ordered list of GPU ID strings."""
    if not isinstance(gpu_spec, str):
        raise TypeError("gpu_spec must be a string like '0' or '0,1'")
    parts = [p.strip() for p in gpu_spec.split(",")]
    parts = [p for p in parts if p != ""]
    if not parts:
        raise ValueError("No GPUs specified (gpu_spec was empty after parsing).")
    seen = set()
    gpus = []
    for p in parts:
        if not p.isdigit() and p not in ['cpu']:
            raise ValueError(f"GPU '{p}' is not a valid numeric GPU ID")
        if p not in seen:
            seen.add(p)
            gpus.append(p)
    return gpus

######## gpu-benchmark #############
def run_gpu_benchmark(gpu_spec: str, model='stable-diffusion-1-5', stream=True, logfile=None, return_empty=False):
    """
    Run GPU benchmark for the specified model on the GPUs given by a CUDA_VISIBLE_DEVICES-like string.

    Parameters
    ----------
    gpu_spec : str
        CUDA-style GPU selector, e.g. '0', '1', '0,1', '0, 1'.
    model : str
        'stable-diffusion-1-5' or 'qwen3-0-6b'.
    stream : bool, default True
        If a single GPU, stream live output (passed to run_cmd).
        For multiple GPUs, streaming is disabled to avoid interleaved logs.
    return_empty : bool, default False
        If true, return a dict of proper structure but values are "".

    Returns
    -------
    dict
        {'num_imgs': <mean images produced across GPUs>}
    """
    if return_empty:
        return {'num_imgs': ""}

    gpus = _parse_gpu_spec(gpu_spec)

    cmd_path = 'gpu-benchmark'
    if shutil.which(cmd_path) is None:
        raise FileNotFoundError(f"{cmd_path} command not found, make sure you installed github.com/jgranley/gpu-benchmark")

    # Build per-GPU output paths + commands
    outfiles = {}
    cmds = []
    for g in gpus:
        outfile = f'./gpu_benchmark_out_{g}.json'
        try:
            if os.path.exists(outfile):
                os.remove(outfile)  # avoid stale data
        except OSError:
            pass
        outfiles[g] = outfile
        cmds.append(f"gpu-benchmark --gpu {g} --output {outfile} --model {model}")

    # Execute (single -> run_cmd, multi -> run_cmds_parallel with stream=False)
    if len(cmds) == 1:
        results = [run_cmd(cmds[0], stream=stream, logfile=logfile)]
    else:
        results = run_cmds_parallel(cmds, stream=False, logfile=logfile)

    # Parse outputs
    num_imgs = []
    try:
        for idx, g in enumerate(gpus):
            outfile = outfiles[g]
            if not os.path.exists(outfile):
                r = results[idx]
                raise FileNotFoundError(
                    f"Expected output not found for GPU {g}: {outfile}\n"
                    f"stderr:\n{r.get('stderr','')}"
                )
            with open(outfile, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list) or len(data) == 0:
                raise ValueError(f"Unexpected result format for GPU {g}: {data} in file {outfile}")

            entry = data[-1]  # use last if multiple
            if str(entry.get('gpu')) != g:
                raise ValueError(f"GPU ID mismatch in result: expected {g}, got {entry.get('gpu')}")

            val = entry.get('primary_metric_value')
            if val is None:
                raise ValueError(f"primary_metric_value missing for GPU {g}: {entry}")

            num_imgs.append(float(val))
    finally:
        # Clean up temp files
        for outfile in outfiles.values():
            try:
                if os.path.exists(outfile):
                    os.remove(outfile)
            except OSError:
                pass

    return {'num_imgs': ','.join(str(v) for v in num_imgs) if num_imgs else float('nan')}


############ gpu-burn ###################

def _parse_gpu_burn(stdout: str):
    """
    Parse gpu-burn stdout and return (gflops, temps) as arrays with shape (n_entries, n_gpus).

    We look for progress lines that contain both 'Gflop/s' and 'temps:'.
    Works for 1+ GPUs; when multiple GPUs, values appear separated by ' - '.
    """
    # print("PARSING\n", stdout)
    progress = []
    max_gpus = 0
    last_perc = -1 # record 1 per percent line
    for line in stdout.splitlines():
        if "Gflop/s" not in line or "temps:" not in line:
            continue

        # example line: 100.0%  proc'd: 405 (26255 Gflop/s) - 405 (26545 Gflop/s)   errors: 0 - 0   temps: 87 C - 74 C
        # extract %
        progress_percent = float(re.findall(r"(\d+\.\d+)%", line)[0]) if re.search(r"(\d+\.\d+)%", line) else None

        # Extract all GFLOP/s numbers inside parentheses: "(12345 Gflop/s)"
        flops = [float(x) for x in re.findall(r"\(([\d\.]+)\s*Gflop/s\)", line)]

        # Extract temps after 'temps:' (captures numbers before 'C'), e.g. "temps: 71 C - 68 C"
        temps_part = line.split("temps:", 1)[-1]
        temps = [float(x) for x in re.findall(r"(\d+)\s*C", temps_part)]

        if not flops or not temps:
            continue
        if progress_percent is None or progress_percent <= last_perc:
            continue
        last_perc = progress_percent

        n = max(len(flops), len(temps))
        max_gpus = max(max_gpus, n)
        progress.append((flops, temps))

    if not progress:
        raise ValueError("No gpu-burn progress lines found with Gflop/s and temps.")

    # Build arrays, padding missing values with NaN if any line has fewer GPUs
    gflops_rows, temps_rows = [], []
    for flops, temps in progress:
        flops_row = [np.nan] * max_gpus
        temps_row = [np.nan] * max_gpus
        for i, v in enumerate(flops[:max_gpus]):
            flops_row[i] = v
        for i, v in enumerate(temps[:max_gpus]):
            temps_row[i] = v
        gflops_rows.append(flops_row)
        temps_rows.append(temps_row)

    gflops = np.array(gflops_rows, dtype=float)  # shape (n_entries, n_gpus)
    temps  = np.array(temps_rows, dtype=float)   # shape (n_entries, n_gpus)
    return gflops, temps


def run_gpu_burn(
    gpu_spec: str,
    seconds: int = 180,
    stream: bool = True,
    logfile: str | None = None,
    return_empty: bool = False
):
    """
    Run gpu-burn for `seconds` on either a single GPU (-i) or all GPUs.

    Behavior
    --------
    - If `gpu_spec` resolves to exactly one GPU: run `./gpu-burn -i <gpu> <seconds>`.
    - If it resolves to 2+ GPUs: gpu-burn will burn *all GPUs* (no -i).
      If the list isn't equal to all detected GPUs, a warning is printed.

    Parameters
    ----------
    gpu_spec : str
        CUDA-like selector, e.g. '0', '1', '0,1', '0, 1, 2'.
    seconds : int
        Duration to burn.
    stream : bool
        Whether to stream gpu-burn output live.
    logfile : str or None
        Optional path to append logs (command + outputs) via run_cmd.

    Returns
    -------
    dict
        {
            'max_gflops': str,
            'mean_gflops': str,
            'min_gflops': str,
        }
    """
    if return_empty:
        return {
            'max_gflops': "",
            'mean_gflops': "",
            'min_gflops': "",
            # 'max_temps': "", # get temps from MONITOR INFO
            # 'mean_temps': "",
            # 'min_temps': ""
        }
    if stream:
        print("GPU Burn streaming not supported")
        stream = False  # gpu-burn output is too verbose for streaming
    burn_dir = "gpu-burn"
    exe_path = os.path.join(burn_dir, "gpu_burn")
    if not (os.path.exists(exe_path) and os.access(exe_path, os.X_OK)):
        raise FileNotFoundError(f"gpu-burn executable not found or not executable: {exe_path}")

    gpus = _parse_gpu_spec(gpu_spec)
    n = len(gpus)

    # Build command
    if n == 1:
        cmd = f"cd {burn_dir} && ./gpu_burn -i {gpus[0]} {int(seconds)}"
    else:
        # When multiple gpus are specified, gpu-burn runs on all available GPUs
        total_gpus = get_num_gpus()

        if total_gpus and n != total_gpus:
            print(f"Warning: gpu-burn will ignore selection {gpus} and burn ALL {total_gpus} GPUs")

        cmd = f"cd {burn_dir} && ./gpu_burn {int(seconds)}"

    # Execute
    res = run_cmd(cmd, stream=stream, logfile=logfile)

    # Parse stdout into arrays
    gflops, temps = _parse_gpu_burn(res["stdout"])
    print(f"Gflops: ", gflops)

    ret = {
        'max_gflops': ','.join(f"{v:.1f}" for v in gflops.max(axis=0)),
        'mean_gflops': ','.join(f"{v:.1f}" for v in gflops.mean(axis=0)),
        'min_gflops': ','.join(f"{v:.1f}" for v in gflops.min(axis=0)),
        # 'max_temps': ','.join(f"{v:.1f}" for v in temps.max(axis=0)),
        # 'mean_temps': ','.join(f"{v:.1f}" for v in temps.mean(axis=0)),
        # 'min_temps': ','.join(f"{v:.1f}" for v in temps.min(axis=0)),
    }

    return ret


######### pytorch-benchmark #################
_TORCH_BENCH_TESTS_CUDA = [
    "test_train[resnet50-cuda]",
    "test_train[vision_maskrcnn-cuda]",
    "test_eval[hf_GPT2-cuda]",
    "test_train[timm_vision_transformer-cuda]",
    "test_eval[torch_multimodal_clip-cuda]",
]

def _tests_for_mode(cpu):
    return [t.replace("-cuda", "-cpu") for t in _TORCH_BENCH_TESTS_CUDA] if cpu else _TORCH_BENCH_TESTS_CUDA

def _k_expr(names):
    return " or ".join(names)

def _parse_pytest_bench_json(path, names_set):
    with open(path, "r") as f:
        data = json.load(f)
    out = {}
    for b in data.get("benchmarks", []):
        name = b.get("name", "")
        if name not in names_set:
            continue
        s = b.get("stats", {}) or {}
        out[name] = {
            "mean": float(s.get("mean", float("nan")))*1000,
            "max": float(s.get("max", float("nan")))*1000,
            "stddev": float(s.get("stddev", float("nan")))*1000,
        }
    return out

def _short_name(full_name):
    # full_name looks like: test_train[resnet50-cuda], test_eval[torch_multimodal_clip-cuda], etc.
    inside = full_name.split("[", 1)[-1].split("]", 1)[0]  # resnet50-cuda, torch_multimodal_clip-cuda, ...
    base = inside.replace("-cuda", "").replace("-cpu", "")
    if "vision_maskrcnn" in base:
        return "mask_rcnn"
    if "timm_vision_transformer" in base:
        return "vit"
    if "torch_multimodal_clip" in base:
        return "clip_eval"
    if "hf_GPT2_large" in base:
        return "gpt2_eval"
    if "resnet50" in base:
        return "resnet50"
    # fallback: compress any other name
    return base.replace("vision_", "").replace("torch_", "").replace("multimodal_", "").replace("hf_", "").replace('test', '').replace('train', '')


def run_torch_benchmark(gpu_spec, stream=True, logfile=None, return_empty=False):
    """
    Run PyTorch benchmarks.

    Parameters
    ----------
    gpu_spec : str
        The GPU specification string.
    stream : bool, optional
        Whether to stream the output (default is True).
    logfile : str, optional
        The path to the log file (default is None).
    return_empty : bool, optional
        If true, return a dict of proper structure but values are "" (default is False).

    Returns
    -------
    dict
        A dictionary containing the benchmark results.
    """
    cpu_mode = (gpu_spec.strip() == "cpu")
    tests = _tests_for_mode(cpu_mode)
    names_set = set(tests)
    kexpr = _k_expr(tests)

    short_names = [_short_name(name) for name in tests]
    if return_empty:
        return {f'{name}_{suffix}': "" for name in short_names for suffix in ["mean", "max", "std"]}

    def _json_path(label):
        return os.path.join(".", f"pytest_bench_{label}_{uuid.uuid4().hex[:8]}.json")

    cmds = []
    labels = []
    json_paths = {}

    if cpu_mode:
        lbl = "cpu"
        jp = _json_path(lbl)
        cmd = (
            f'pytest ./benchmark/test_bench.py '
            f'-k "{kexpr}" '
            f'--ignore_machine_config '
            f'--benchmark-json={shlex.quote(jp)} '
            f'--benchmark-max-time=5'
        )
        cmds.append(cmd); labels.append(lbl); json_paths[lbl] = jp
    else:
        gpus = _parse_gpu_spec(gpu_spec)  # e.g., ["0"] or ["0","1","2"]
        if len(gpus) == 1:
            g = gpus[0]
            lbl = f"gpu{g}"
            jp = _json_path(lbl)
            cmd = (
                f'CUDA_VISIBLE_DEVICES="{g}" '
                f'pytest ./benchmark/test_bench.py '
                f'-k "{kexpr}" '
                f'--ignore_machine_config '
                f'--benchmark-json={shlex.quote(jp)} '
                f'--benchmark-max-time=5'
            )
            cmds.append(cmd); labels.append(lbl); json_paths[lbl] = jp
        else:
            for g in gpus:
                lbl = f"gpu{g}"
                jp = _json_path(lbl)
                cmd = (
                    f'CUDA_VISIBLE_DEVICES="{g}" '
                    f'pytest ./benchmark/test_bench.py '
                    f'-k "{kexpr}" '
                    f'--ignore_machine_config '
                    f'--benchmark-json={shlex.quote(jp)} '
                    f'--benchmark-max-time=5'
                )
                cmds.append(cmd); labels.append(lbl); json_paths[lbl] = jp

    # Execute
    if len(cmds) == 1:
        run_res = [run_cmd(cmds[0], stream=stream, logfile=logfile)]
    else:
        run_res = run_cmds_parallel(cmds, stream=False, logfile=logfile)

    # Parse all JSONs into per-run dicts
    per_run = {}
    for lbl in labels:
        jp = json_paths[lbl]
        if not os.path.exists(jp):
            idx = labels.index(lbl)
            raise FileNotFoundError(
                f"Benchmark JSON not found for {lbl}: {jp}\n"
                f"stderr:\n{run_res[idx].get('stderr','')}"
            )
        per_run[lbl] = _parse_pytest_bench_json(jp, names_set)

    # Build flat output: for each short bench name, collect mean/max/std across runs (order = labels)
    # If a test is missing in a run, use NaN.
    flat = {}
    short_order = []
    for name in tests:
        s = _short_name(name)
        short_order.append(s)

    metrics = ("mean", "max", "stddev")
    suffix_map = {"mean": "mean", "max": "max", "stddev": "std"}
    for name, short in zip(tests, short_order):
        vals_by_metric = {m: [] for m in metrics}
        for lbl in labels:
            rec = per_run[lbl].get(name)
            for m in metrics:
                v = rec.get(m) if rec else float("nan")
                vals_by_metric[m].append(v)
        for m in metrics:
            flat[f"{short}_{suffix_map[m]}"] = ",".join(f"{x:.2f}" for x in vals_by_metric[m])

    # Cleanup JSON files
    for jp in json_paths.values():
        try:
            os.remove(jp)
        except OSError:
            pass

    return flat



######## io dataloader imagenet training ###################

def _hms_to_seconds(s):
    parts = s.strip().split(":")
    try:
        if len(parts) == 3:
            h, m, sec = map(int, parts)
        elif len(parts) == 2:
            h, m = map(int, parts)
            sec = 0
        else:
            return float(s)
        return h*3600 + m*60 + sec
    except Exception:
        return float('nan')

def _parse_training_log_with_summary(log_text):
    """
    Robustly parse torchvision training output, tolerant to truncated logs.

    Returns:
      img_s (np.ndarray)
      iter_times (np.ndarray)
      data_times (np.ndarray)
      epoch_total_time_m (float, NaN if missing, minutes)
      test_acc5 (float, NaN if missing)
    """
    # Find boundaries to avoid parsing into Test section or beyond epoch total
    m_total = re.search(r"Epoch:\s*\[\s*0\s*\]\s*Total time:\s*([0-9:]+)", log_text)
    m_test_start = re.search(r"(?m)^\s*Test:\s", log_text)  # start of eval logs

    # stop at the earliest of total-time or test-start; else parse whole text
    stops = [m.start() for m in (m_total, m_test_start) if m]
    stop_idx = min(stops) if stops else len(log_text)
    train_chunk = log_text[:stop_idx]

    # numbers like 1234, 12.34, 1e-3, 1.2E+03
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    # Capture img/s, time, data from training lines
    pat = re.compile(
        rf"img/s:\s*({num}).*?time:\s*({num})\s*data:\s*({num})",
        re.DOTALL
    )

    img_s, iter_times, data_times = [], [], []
    for m in pat.finditer(train_chunk):
        try:
            img_s.append(float(m.group(1)))
            iter_times.append(float(m.group(2)))
            data_times.append(float(m.group(3)))
        except Exception:
            # tolerate malformed/truncated line
            continue

    img_s = np.array(img_s, dtype=float)
    iter_times = np.array(iter_times, dtype=float)
    data_times = np.array(data_times, dtype=float)

    # Epoch total time (seconds), if present
    epoch_total_time_s = float('nan')
    if m_total:
        epoch_total_time_s = _hms_to_seconds(m_total.group(1))

    # Final Test Acc@5, if present
    m_acc = re.search(r"Test:\s*Acc@1\s*([0-9.]+)\s*Acc@5\s*([0-9.]+)", log_text)
    test_acc5 = float('nan')
    if m_acc:
        try:
            test_acc5 = float(m_acc.group(2))
        except Exception:
            pass

    return img_s, iter_times, data_times, epoch_total_time_s / 60, test_acc5


def run_imagenet_reference(gpu_spec, stream=True, logfile=None, return_empty=False):
    """
    Run the torchvision classification reference (mobilenet_v3_small) in two phases:
      (1) Full-run benchmark: -j 16, --batch-size 128, epochs=1
          -> logs mean_img_s, total_runtime_min, mean_data_time, mean_iter_time
      (2) I/O sweeps: for j in {1,2,4,8,16,32}, timeout=30s each
          -> logs IO_<j>_mean_data_time, IO_<j>_mean_iter_time
          (means computed after dropping the first iteration)

    GPU modes:
      - Parallel per-GPU: "0", "0,1,2" -> one process per GPU with CUDA_VISIBLE_DEVICES set
      - Multiprocessing (DDP): "0+1(+...)" -> single torchrun with nproc_per_node=len(gpus) and
        CUDA_VISIBLE_DEVICES="g0,g1,..."
      - Mixing '+' and ',' is not allowed.

    Returns:
      dict of flat key->string values. For parallel per-GPU, values are comma-separated (GPU order).
      For multiprocessing mode, single values are returned (rank0 prints).
    """
    io_workers = [1, 2, 4, 8, 16]
    if return_empty:
        ret = {"img/s" : "",
                "tot_time_m" : "",
                "data_s" : "",
                "iter_s" : "",
                "acc5" : "",
               }
        ret.update({f"IO_{jw}_{suffix}": "" for jw in io_workers for suffix in ["data_time", "iter_time"]})
        return ret

    # make sure IMAGENET env variable is set
    if "IMAGENET" not in os.environ:
        raise ValueError("Please set IMAGENET environment variable to data location")
    imagenet = os.environ["IMAGENET"]

    has_plus = "+" in gpu_spec
    has_comma = "," in gpu_spec
    if has_plus and has_comma:
        raise ValueError("gpu_spec cannot mix '+' (multiprocessing) with ',' (parallel per-GPU).")

    if has_plus:
        gpus = [g.strip() for g in gpu_spec.split("+") if g.strip() != ""]
        mode = "mp"   # single torchrun with nproc=len(gpus)
    else:
        gpus = _parse_gpu_spec(gpu_spec) if gpu_spec.strip() != "" else []
        mode = "pergpu"  # one torchrun per GPU, nproc=1 each

    def build_cmd(nproc, workers, add_batch=False):
        # add_batch controls whether we include --batch-size 128 (only for full-run)
        base = f'torchrun --nproc_per_node={int(nproc)} ' if nproc > 1 else 'python3 '
        base += (
            f'vision/references/classification/train.py --model mobilenet_v3_small '
            f'--data-path {shlex.quote(imagenet)} '
            f'-j {int(workers)} --epochs 1'
        )
        if add_batch:
            base += ' --batch-size 128'
        return base

    def summarize(stdout_text, drop_first=False):
        img_s, iter_t, data_t, total_min, _acc5 = _parse_training_log_with_summary(stdout_text)
        if drop_first and len(iter_t) >= 2:
            if len(img_s) >= 2:
                img_s = img_s[1:]
            iter_t = iter_t[1:]
            data_t = data_t[1:]
        def _mean(x):
            return float(np.nanmean(x)) if len(x) else float('nan')
        return _mean(img_s), float(total_min), _mean(data_t), _mean(iter_t), _acc5

    out = {}

    # =========================
    # Phase 1: Full-run bench
    # =========================
    if mode == "mp":
        nproc = len(gpus)
        env_prefix = f'CUDA_VISIBLE_DEVICES="{",".join(gpus)}" '
        cmd = env_prefix + build_cmd(nproc=nproc, workers=16, add_batch=True)
        res = run_cmd(cmd, stream=stream, logfile=logfile)
        if res['returncode'] != 0:
            raise RuntimeError(f"Imagenet classification for spec {gpu_spec} failed")
        m_img, t_min, m_data, m_it, acc5 = summarize(res["stdout"], drop_first=False)
        out["img/s"] = f"{m_img:.1f}"
        out["tot_time_m"] = f"{t_min:.1f}"
        out["data_s"] = f"{m_data:.3f}"
        out["iter_s"] = f"{m_it:.3f}"
        out["acc5"] = f"{acc5:.2f}"
    else:
        cmds = []
        for g in gpus:
            env_prefix = f'CUDA_VISIBLE_DEVICES="{g}" '
            if len(gpus) == 1:
                nworkers = 16
            else: 
                nworkers = 8
            cmds.append(env_prefix + build_cmd(nproc=1, workers=nworkers, add_batch=True))
        if len(cmds) == 1:
            res_list = [run_cmd(cmds[0], stream=stream, logfile=logfile)]
        else:
            res_list = run_cmds_parallel(cmds, stream=stream)  # your wrapper auto-disables stream when >1
        vals_img, vals_tot, vals_data, vals_it, vals_acc5 = [], [], [], [], []
        for r in res_list:
            m_img, t_min, m_data, m_it, acc5 = summarize(r["stdout"], drop_first=False)
            vals_img.append(m_img); vals_tot.append(t_min); vals_data.append(m_data); vals_it.append(m_it); vals_acc5.append(acc5)
        out["img/s"] = ",".join(f"{v:.1f}" for v in vals_img)
        out["tot_time_m"] = ",".join(f"{v:.1f}" for v in vals_tot)
        out["data_s"] = ",".join(f"{v:.3f}" for v in vals_data)
        out["iter_s"] = ",".join(f"{v:.3f}" for v in vals_it)
        out["acc5"] = ",".join(f"{v:.2f}" for v in vals_acc5)

    # =========================
    # Phase 2: I/O sweeps
    # =========================
    
    for jw in io_workers:
        if mode == "mp":
            nproc = len(gpus)
            env_prefix = f'CUDA_VISIBLE_DEVICES="{",".join(gpus)}" '
            cmd = env_prefix + build_cmd(nproc=nproc, workers=jw, add_batch=False)  # default batch size
            r = run_cmd(cmd, stream=stream, logfile=logfile, timeout=95)
            _m_img, _t_min, m_data, m_it, _acc5 = summarize(r["stdout"], drop_first=True)
            out[f"IO_{jw}_data_time"] = f"{m_data:.3f}"
            out[f"IO_{jw}_iter_time"] = f"{m_it:.3f}"
        else:
            cmds = []
            for g in gpus:
                env_prefix = f'CUDA_VISIBLE_DEVICES="{g}" '
                cmds.append(env_prefix + build_cmd(nproc=1, workers=jw, add_batch=False))
            if len(cmds) == 1:
                res_list = [run_cmd(cmds[0], stream=stream, logfile=logfile, timeout=30)]
            else:
                res_list = run_cmds_parallel(cmds, stream=stream, timeout=30)
            vals_data, vals_it = [], []
            for r in res_list:
                _m_img, _t_min, m_data, m_it, _acc5 = summarize(r["stdout"], drop_first=True)
                vals_data.append(m_data); vals_it.append(m_it)
            out[f"IO_{jw}_data_time"] = ",".join(f"{v:.3f}" for v in vals_data)
            out[f"IO_{jw}_iter_time"] = ",".join(f"{v:.3f}" for v in vals_it)

    return out
    

    