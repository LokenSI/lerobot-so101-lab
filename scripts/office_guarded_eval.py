"""Bounded launch guard for one explicitly owned evaluation process and its children."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-monitor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1200)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.output.exists():
        raise ValueError('Require a command and a fresh evaluation output path')
    started = time.monotonic()
    report = {'scope': 'owned evaluation process only; includes cold model loading',
              'command': command, 'complete': False, 'admission': 'pending',
              'gpu_limit_mib': 14336, 'ram_limit_mib': 49152,
              'peak_gpu_mib': None, 'peak_host_mib': None, 'samples_observed': 0}

    cached_sample = None

    def admission():
        nonlocal cached_sample
        try:
            sample = json.loads(args.host_monitor.read_text())
            sample['samples'][-1]['utc']
            cached_sample = sample
        except (OSError, ValueError, KeyError, IndexError) as exc:
            if cached_sample is None:
                raise RuntimeError('No readable whole-host admission sample') from exc
            # A sharing collision must not abort a load when the last valid sample
            # is still fresh. A failed monitor becomes stale and then stops us.
            sample = cached_sample
        row = sample['samples'][-1]
        stamp = dt.datetime.fromisoformat(row['utc'].replace('Z', '+00:00'))
        age = (dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()
        report['samples_observed'] += 1
        report['peak_gpu_mib'] = max(report['peak_gpu_mib'] or 0, sample['gpu_peak_mib'])
        report['peak_host_mib'] = max(report['peak_host_mib'] or 0, sample['host_peak_mib'])
        if sample.get('threshold_14gib_exceeded') or sample.get('threshold_48gib_ram_exceeded'):
            raise MemoryError('Whole-host GPU/RAM admission threshold exceeded')
        if age > 10 or age < -5:
            raise RuntimeError(f'Whole-host admission sample is stale ({age:.1f}s)')

    child = None
    error = None
    try:
        # Wait only for the initial monitor publication, before allocating the model.
        for attempt in range(50):
            try:
                admission()
                break
            except RuntimeError:
                if attempt == 49:
                    raise
                time.sleep(.1)
        child = subprocess.Popen(command, start_new_session=(os.name == 'posix'))
        report['owned_child_pid'] = child.pid
        print(json.dumps({'guard': 'active', 'owned_child_pid': child.pid}), flush=True)
        while child.poll() is None:
            admission()
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError('Owned evaluation exceeded its bounded wall-clock budget')
            time.sleep(.1)
        report['returncode'] = child.returncode
        report['admission'] = 'PASS'
        report['complete'] = child.returncode == 0
        if child.returncode:
            error = f'Owned evaluation exited with status {child.returncode}'
    except Exception as exc:
        error = str(exc)
        report['admission'] = 'FAIL'
        if child is not None and child.poll() is None:
            if os.name == 'posix':
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
                child.wait(timeout=3)
            report['returncode'] = child.returncode
            report['owned_child_terminated'] = True
    finally:
        report['wall_seconds'] = time.monotonic() - started
        if error:
            report['error'] = error
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'guard-report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    if error:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
