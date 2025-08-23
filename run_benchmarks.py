"""
Main python script to run benchmarks

"""

import argparse
import os
import datetime
import pandas as pd
import requests
from utils import collect_machine_info, check_active_cpu_gpu_processes, run_with_gpu_monitor, get_num_gpus, get_gpu_temps
from benchmark_helpers import run_gpu_benchmark, run_gpu_burn, run_torch_benchmark, run_imagenet_reference
import time

def _get_applicable_gpu_specs(args, benchmarks):
    """
    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments.
    metric : str
        The benchmark metric to be ran. options are gb, sd, tb, and im, or all for all of them
    Returns
    -------
    list[str]
        A list of applicable GPU specs.
    """
    total_applicable_gpus = [str(i) for i in range(get_num_gpus())] if args.gpus == 'all' else args.gpus.split(',')
    if benchmarks == "sd" or benchmarks == 'burn':
        # all single gpus, and all 
        return total_applicable_gpus + [','.join(total_applicable_gpus)]
    elif benchmarks == "tb":
        # also supports cpu mode
        return total_applicable_gpus + [','.join(total_applicable_gpus)] + ['cpu']
    elif benchmarks == "im":
        # no cpu mode, but supports multiprocessing of all e.g. '0+1'
        return total_applicable_gpus + [','.join(total_applicable_gpus)] + ['+'.join(total_applicable_gpus)]
    elif benchmarks == "all":
        return total_applicable_gpus + [','.join(total_applicable_gpus)] + ['+'.join(total_applicable_gpus)] + ['cpu']
    else:
        raise ValueError(f"Unknown benchmarks: {benchmarks}. Supported metrics are 'sd', 'burn', 'tb', and 'im'.")


def _result_dict_to_df(result_dict):
    """
    Convert the result dictionary to a pandas DataFrame.

    Column order is deterministic regardless of input order:
      metrics first, then monitors, with benchmarks ordered as:
      gpu-burn (_burn), imagenet (_im), torchbench (_tb), stable diffusion (_sd).
    """
    rows = []

    # Keep per-benchmark insertion order for columns we see
    metric_by_bm = {"burn": [], "im": [], "tb": [], "sd": []}
    monitor_by_bm = {"burn": [], "im": [], "tb": [], "sd": []}

    for gpu_spec, bm_map in result_dict.items():
        row = {"gpu_spec": gpu_spec}
        for bm in bm_map.keys():  # bm in {"burn","im","tb","sd"}
            results, monitor = bm_map[bm]
            # Metrics
            for k, v in results.items():
                col = f"{k}_{bm}"
                row[col] = v
                if col not in metric_by_bm[bm]:
                    metric_by_bm[bm].append(col)
            # Monitor info
            for k, v in monitor.items():
                col = f"{k}_{bm}"
                row[col] = v
                if col not in monitor_by_bm[bm]:
                    monitor_by_bm[bm].append(col)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Deterministic benchmark order
    bm_order = ["burn", "im", "tb", "sd"]

    # Flatten the per-benchmark lists in the desired order
    metric_cols = [col for bm in bm_order for col in metric_by_bm[bm]]
    monitor_cols = [col for bm in bm_order for col in monitor_by_bm[bm]]

    # Final column order: gpu_spec, metrics..., monitors...
    ordered_cols = ["gpu_spec"] + metric_cols + monitor_cols
    df = df.reindex(columns=ordered_cols)

    return df


def _wait_gpu_temp(idle_temps, margin=10, verbose=False):
    """
    Halts execution until GPU temps return to within margin % of idle temps
    or are below 60°C.

    Returns seconds waited
    """
    start = time.time()
    current_temps = get_gpu_temps()
    while True:
        if all(abs(c - i) <= margin / 100 * i for c, i in zip(current_temps, idle_temps)) or all(c <= 60 for c in current_temps):
            break
        if verbose:
            target_temps = [max(i + margin / 100 * i, 60) for i in idle_temps]
            print(f"\tCurrent temps: {current_temps}, Target temps: {target_temps}, Idle Temps: {idle_temps}")
        time.sleep(1)
        current_temps = get_gpu_temps()
    return time.time() - start

def run_benchmarks(args):
    benchmarks = []
    if not args.skip_gpu_burn:
        benchmarks.append("burn")
    if not args.skip_torchbench:
        benchmarks.append("tb")
    if not args.skip_imagenet_classification:
        benchmarks.append("im")
    if not args.skip_stable_diffusion:
        benchmarks.append("sd")
    # if len(benchmarks) == 0:
    #     raise ValueError("No benchmarks specified to run")
    
    bm_fns = {
        'im'  : run_imagenet_reference,
        'tb'  : run_torch_benchmark,
        'burn': run_gpu_burn,
        'sd' : run_gpu_benchmark,
    }

    
    all_gpu_specs = _get_applicable_gpu_specs(args, 'all')
    ret_dict = {g : {bm : None for bm in bm_fns.keys()} for g in all_gpu_specs} # each entry will be (results, monitor_info)


    print('Checking to make sure GPU and CPU are clear')
    status = check_active_cpu_gpu_processes(all_gpu_specs)
    if (status['cpu_busy'] or status['gpu_busy']) and not args.force:
        print("Please make sure there are no processes running during the benchmarks.")
        print("If you really want to run with external processes, please use the --force flag.")
        print(status)
        raise RuntimeError("CPU or GPU is busy. Please clear the processes before running benchmarks.")
    print("All clear")

    idle_temps = get_gpu_temps()
    print(f"Idle GPU temps: {idle_temps}")

    for benchmark in bm_fns.keys():
        bm_gpu_specs = _get_applicable_gpu_specs(args, benchmark)
        for gpu_spec in all_gpu_specs:
            return_empty=False
            if gpu_spec not in bm_gpu_specs or benchmark not in benchmarks:
                return_empty = True
            if args.verbose:
                if not return_empty:
                    print(f"\n\nRunning {benchmark} on {gpu_spec}")
                else:
                    print(f"Skipping {benchmark} on {gpu_spec}")
            try:
                bm_results, bm_monitor_info = run_with_gpu_monitor(
                    bm_fns[benchmark],
                    gpu_spec,
                    stream=args.verbose,
                    logfile=args.logfile,
                    return_empty=return_empty)
            # except ctrl-c and delete log file before exiting 
            except KeyboardInterrupt as e:
                if args.logfile and os.path.exists(args.logfile):
                    os.remove(args.logfile)
                raise e
            
            print(f"Finished {benchmark} on {gpu_spec}, waiting for gpus to cool down")
            secs_waited = _wait_gpu_temp(idle_temps, verbose=args.verbose, margin=10)
            if bm_monitor_info is not None:
                bm_monitor_info['cooldown_wait_s'] = secs_waited
            ret_dict[gpu_spec][benchmark] = (bm_results, bm_monitor_info)

    ret_df = _result_dict_to_df(ret_dict)
    return ret_df


def _upload_to_gsheets(args, df):
    if args.verbose:
        print("Uploading results to google sheets: https://docs.google.com/spreadsheets/d/1WVKt2bdIEx8Mu3fXxjMYucGKR6blENCLQnFM7-0EuwQ/edit?usp=sharing")
    URL = getattr(args, "gsheets_url", "") or ""
    if not URL:
        print("[gsheets] Skipped: no --gsheets-url provided.")
        return

    for _, row in df.iterrows():
        payload = {"values": [str(v) if v is not None else "" for v in row.values]}
        try:
            resp = requests.post(URL, json=payload, timeout=15)
            if resp.status_code != 200:
                print(f"[gsheets] Upload failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"[gsheets] Upload error: {e}")


def record_results(args, res_df, timestamp, machine_info):
    ### add in metadata ###
    n = len(res_df)
    front_df = pd.DataFrame({
        "timestamp": [timestamp] * n,
        "hostname": [machine_info.get("hostname", "")] * n,
        "runname": [getattr(args, "runname", "")] * n,
    })

    machine_cols_tail = [k for k in machine_info.keys() if k != "hostname"]
    machine_df = pd.DataFrame([machine_info] * n)
    machine_df = machine_df[machine_cols_tail] if machine_cols_tail else pd.DataFrame(index=range(n))

    final_df = pd.concat(
        [front_df.reset_index(drop=True),
         res_df.reset_index(drop=True),
         machine_df.reset_index(drop=True)],
        axis=1
    )
    if not args.no_upload:
        _upload_to_gsheets(args, final_df)
    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        write_header = not os.path.exists(args.out_csv)
        # check if header matches current
        if os.path.exists(args.out_csv):
            with open(args.out_csv, 'r') as f:
                reader = pd.read_csv(f, nrows=0)
                if not reader.columns.equals(final_df.columns):
                    print(f"[csv] Header mismatch in {args.out_csv}:")
                    print(f"  Expected: {final_df.columns.tolist()}")
                    print(f"  Found:    {reader.columns.tolist()}")
                    print(f"ERROR: could not write results csv, header mismatch")
                    return
        final_df.to_csv(args.out_csv, mode='a', header=write_header, index=False)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('--gpus', type=str, default='all', help='Options: all (default), or a specific gpu id (e.g. 0, 1, 2)')
    parser.add_argument('--skip-stable-diffusion', action='store_true', help='Skip Stable Diffusion benchmark', default=False)
    parser.add_argument('--skip-gpu-burn', action='store_true', help='Skip GPU Burn benchmark', default=False)
    parser.add_argument('--skip-torchbench', action='store_true', help='Skip TorchBench benchmark', default=False)
    parser.add_argument('--skip-imagenet-classification', action='store_true', help='Skip ImageNet Classification benchmark', default=False)
    parser.add_argument('--logfile', type=str, default=None, help='Path to the log file. Use no_log to disable logging')
    parser.add_argument('--out-csv', type=str, default=None, help='Path to output CSV file, if desired')
    parser.add_argument('--no-upload', action='store_true', help='Dont upload results to the google sheet', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output', default=False)
    parser.add_argument('--force', action='store_true', help='Force run benchmarks even if CPU/GPU is busy', default=False)
    parser.add_argument('--runname', type=str, default='', help='Name of the run, for logging')
    parser.add_argument('--gsheets-url', type=str, help='Google Sheets API URL', default='https://script.google.com/macros/s/AKfycbzM_i4Xx8PcKRR8H4AW7ENqNjGNHezLdQTGMkdFgoL_o_X-ZU6DcLSGDpWo50sBVLvoMg/exec')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    machine_info = collect_machine_info()

    if args.logfile is None:
        args.logfile = f".logs/benchmark_{timestamp}.log"
    elif args.logfile == "no_log":
        args.logfile = None
    if args.logfile and os.path.dirname(args.logfile):
        os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
    # also prompt to confirm the selected options
    print("Selected options:")
    print(f"  GPUs: {args.gpus}")
    print(f"  Skip Stable Diffusion: {args.skip_stable_diffusion}")
    print(f"  Skip GPU Burn: {args.skip_gpu_burn}")
    print(f"  Skip TorchBench: {args.skip_torchbench}")
    print(f"  Skip ImageNet Classification: {args.skip_imagenet_classification}")
    print(f"  Log file: {args.logfile}")
    print(f"  Output CSV: {args.out_csv}")
    print(f"  Upload: {not args.no_upload}")
    print(f"  Verbose: {args.verbose}")

    res_df = run_benchmarks(args)

    record_results(args, res_df, timestamp, machine_info)

    print("Done.")
