"""
Main python script to run benchmarks

"""

import argparse
import os
import datetime
import pandas as pd
import requests
from utils import collect_machine_info, check_active_cpu_gpu_processes, run_with_gpu_monitor, get_num_gpus
from benchmark_helpers import run_gpu_benchmark, run_gpu_burn, run_torch_benchmark, run_imagenet_reference


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

    Parameters
    ----------
    result_dict : dict
        The result dictionary containing GPU specs and their corresponding benchmark results.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the benchmark results.
    """
    rows = []
    metric_cols, monitor_cols = set(), set()

    for gpu_spec, bm_map in result_dict.items():
        row = {"gpu_spec": gpu_spec}
        for bm in bm_map.keys():
            results, monitor = bm_map[bm]
            for k, v in results.items():
                col = f"{k}_{bm}"
                row[col] = v
                metric_cols.add(col)
            for k, v in monitor.items():
                col = f"{k}_{bm}"
                row[col] = v
                monitor_cols.add(col)
        rows.append(row)

    df = pd.DataFrame(rows)

    # order: gpu_spec, metrics..., monitors...
    ordered_cols = ["gpu_spec"] + sorted(metric_cols) + sorted(monitor_cols)
    df = df.reindex(columns=ordered_cols)

    return df


def run_benchmarks(args):
    benchmarks = []
    if not args.skip_stable_diffusion:
        benchmarks.append("sd")
    if not args.skip_gpu_burn:
        benchmarks.append("burn")
    if not args.skip_torchbench:
        benchmarks.append("tb")
    if not args.skip_imagenet_classification:
        benchmarks.append("im")
    if len(benchmarks) == 0:
        raise ValueError("No benchmarks specified to run")
    
    bm_fns = {
        'sd' : run_gpu_benchmark,
        'burn': run_gpu_burn,
        'tb'  : run_torch_benchmark,
        'im'  : run_imagenet_reference,
    }

    
    all_gpu_specs = _get_applicable_gpu_specs(args, 'all')
    ret_dict = {g : {bm : None for bm in benchmarks} for g in all_gpu_specs} # each entry will be (results, monitor_info)


    print('Checking to make sure GPU and CPU are clear')
    status = check_active_cpu_gpu_processes(all_gpu_specs)
    if (status['cpu_busy'] or status['gpu_busy']) and not args.force:
        print("Please make sure there are no processes running during the benchmarks.")
        print("If you really want to run with external processes, please use the --force flag.")
        print(status)
        raise RuntimeError("CPU or GPU is busy. Please clear the processes before running benchmarks.")
    print("All clear")


    for benchmark in benchmarks:
        bm_gpu_specs = _get_applicable_gpu_specs(args, benchmark)
        for gpu_spec in all_gpu_specs:
            return_empty=False
            if gpu_spec not in bm_gpu_specs:
                return_empty = True
            if args.verbose:
                print(f"\n\nRunning {benchmark} on {gpu_spec}")
            bm_results, bm_monitor_info = run_with_gpu_monitor(
                bm_fns[benchmark],
                gpu_spec,
                stream=args.verbose,
                logfile=args.logfile,
                return_empty=return_empty)
            ret_dict[gpu_spec][benchmark] = (bm_results, bm_monitor_info)

    ret_df = _result_dict_to_df(ret_dict)
    return ret_df


def _upload_to_gsheets(args, df):
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
    parser.add_argument('--out_csv', type=str, default=None, help='Path to output CSV file, if desired')
    parser.add_argument('--no_upload', action='store_true', help='Dont upload results to the google sheet', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output', default=False)
    parser.add_argument('--force', action='store_true', help='Force run benchmarks even if CPU/GPU is busy', default=False)
    parser.add_argument('--runname', type=str, default='', help='Name of the run, for logging')
    parser.add_argument('--gsheets-url', type=str, help='Google Sheets API URL', default='https://script.google.com/macros/s/AKfycbyWE17cMlYp9pgF8Y5gMTvBCZ3ppcSKz9bLxjPM2ArCTPSfJ-tvF80O2VEyUwiTLXEcEA/exec')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    machine_info = collect_machine_info()

    if args.logfile is None:
        args.logfile = f"benchmark_{timestamp}.log"
    elif args.logfile == "no_log":
        args.logfile = None

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
