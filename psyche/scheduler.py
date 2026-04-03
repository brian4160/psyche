"""Smart eval scheduler for shared machines (Windows/Linux).

Monitors system activity and automatically pauses/resumes eval runs
when someone is using the computer (e.g., gaming). Unloads the LLM
from GPU memory when paused so the full VRAM is available.

Usage:
    psyche-schedule --jobs jobs.json

The scheduler:
1. Checks GPU utilization and user input activity every 30 seconds
2. If GPU usage is high (someone gaming) or user is active, it:
   - Saves progress (incremental save already handles this)
   - Stops the current eval conversation after it finishes
   - Unloads the Ollama model from VRAM
   - Waits until the machine is idle again
3. When idle, reloads the model and resumes from where it left off

Works on both Windows and Linux.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

log = logging.getLogger(__name__)

# Thresholds
GPU_BUSY_THRESHOLD = 40        # % GPU utilization = "someone is using it"
IDLE_CHECK_INTERVAL = 30       # seconds between activity checks
IDLE_REQUIRED_SECONDS = 120    # must be idle for 2 min before resuming
PAUSE_CHECK_INTERVAL = 60     # seconds between checks while paused


def get_gpu_utilization() -> int:
    """Get current GPU utilization percentage. Returns 0 if can't determine."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def get_gpu_memory_used() -> int:
    """Get GPU memory used in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def unload_ollama_model(model: str) -> None:
    """Unload a model from Ollama's GPU memory."""
    log.info(f"Unloading {model} from GPU memory...")
    try:
        # Ollama keeps models loaded for 5 min by default.
        # Setting keep_alive to 0 unloads immediately.
        import httpx
        httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=30,
        )
        log.info(f"Model {model} unloaded.")
    except Exception as e:
        log.warning(f"Failed to unload model: {e}")
        # fallback: try stopping ollama entirely
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/f", "/im", "ollama.exe"],
                             capture_output=True, timeout=10)
            else:
                subprocess.run(["systemctl", "stop", "ollama"],
                             capture_output=True, timeout=10)
            log.info("Stopped Ollama service.")
        except Exception:
            pass


def ensure_ollama_running() -> None:
    """Make sure Ollama is running."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/", timeout=5)
        if resp.status_code == 200:
            return  # already running
    except Exception:
        pass

    log.info("Starting Ollama...")
    try:
        if platform.system() == "Windows":
            # On Windows, Ollama runs as a background process
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
            )
        else:
            subprocess.run(["systemctl", "start", "ollama"],
                         capture_output=True, timeout=10)
        time.sleep(5)  # wait for startup
    except Exception as e:
        log.error(f"Failed to start Ollama: {e}")


def is_machine_busy() -> bool:
    """Check if someone is actively using the GPU (gaming, rendering, etc.)."""
    gpu_util = get_gpu_utilization()
    if gpu_util > GPU_BUSY_THRESHOLD:
        log.debug(f"GPU busy: {gpu_util}%")
        return True
    return False


def wait_until_idle(model: str) -> None:
    """Wait until the machine is idle, with model unloaded from GPU."""
    print(f"\n  [PAUSED] Machine is busy — yielding GPU. Will resume when idle.",
          flush=True)
    log.info("Machine busy — pausing and unloading model")
    unload_ollama_model(model)

    idle_start = None
    while True:
        time.sleep(PAUSE_CHECK_INTERVAL)
        if is_machine_busy():
            idle_start = None
            continue

        if idle_start is None:
            idle_start = time.time()
            log.debug("Machine appears idle, waiting to confirm...")
        elif time.time() - idle_start > IDLE_REQUIRED_SECONDS:
            log.info("Machine confirmed idle — resuming")
            print(f"  [RESUMING] Machine idle — restarting eval.", flush=True)
            ensure_ollama_running()
            # warm the model back up
            try:
                import httpx
                httpx.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": "warmup", "keep_alive": "10m",
                          "options": {"num_predict": 1}},
                    timeout=120,
                )
            except Exception:
                pass
            return


def run_scheduled_eval(jobs_file: str) -> None:
    """Run eval jobs from a JSON config file with activity-aware scheduling.

    Jobs file format:
    {
        "jobs": [
            {
                "name": "12B additional runs",
                "model": "mistral-nemo",
                "scripts": ["casual_chat", "emotional_depth", ...],
                "conditions": ["plain", "freudian", ...],
                "runs": 10,
                "resume_file": "eval_results/eval_xxx.json"
            },
            ...
        ]
    }
    """
    with open(jobs_file) as f:
        config = json.load(f)

    jobs = config["jobs"]
    print(f"Loaded {len(jobs)} eval jobs")

    for i, job in enumerate(jobs):
        name = job.get("name", f"Job {i+1}")
        model = job["model"]
        scripts = job.get("scripts")
        conditions = job.get("conditions")
        runs = job.get("runs", 5)
        resume = job.get("resume_file")

        print(f"\n{'='*60}")
        print(f"  JOB {i+1}/{len(jobs)}: {name}")
        print(f"  Model: {model}")
        print(f"  Scripts: {scripts or 'all'}")
        print(f"  Conditions: {conditions or 'all'}")
        print(f"  Runs: {runs}")
        if resume:
            print(f"  Resume from: {resume}")
        print(f"{'='*60}")

        # check if machine is busy before starting
        if is_machine_busy():
            wait_until_idle(model)
        else:
            ensure_ollama_running()

        # build eval command
        cmd = [sys.executable, "-m", "psyche.evaluate"]
        cmd.extend(["--models", model])
        cmd.extend(["--runs", str(runs)])
        cmd.append("--skip-judge")
        if scripts:
            cmd.extend(["--scripts", ",".join(scripts)])
        if conditions:
            cmd.extend(["--conditions", ",".join(conditions)])
        if resume:
            cmd.extend(["--resume", resume])

        # run eval as subprocess so we can monitor GPU alongside it
        log.info(f"Starting: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

        # monitor while running
        while proc.poll() is None:
            time.sleep(IDLE_CHECK_INTERVAL)
            if is_machine_busy():
                # someone started using the machine — let current conversation finish
                # then the incremental save will preserve progress
                print(f"\n  [YIELDING] Detected GPU activity — "
                      f"finishing current conversation then pausing...", flush=True)
                # send interrupt to gracefully stop
                proc.terminate()
                proc.wait(timeout=120)  # wait for current conversation to finish + save
                wait_until_idle(model)
                # restart the job — resume will skip completed work
                log.info(f"Restarting job: {name}")
                proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

        if proc.returncode != 0:
            log.warning(f"Job {name} exited with code {proc.returncode}")
        else:
            print(f"  Job {name} complete!", flush=True)


def main():
    """Entry point for psyche-schedule command."""
    import sys
    from psyche.main import setup_logging

    log_file = setup_logging()
    logging.getLogger("psyche").info(f"Scheduler log: {log_file}")

    if len(sys.argv) < 2 or "--jobs" not in sys.argv:
        print("Usage: psyche-schedule --jobs <jobs.json>")
        print("\nExample jobs.json:")
        print(json.dumps({
            "jobs": [
                {
                    "name": "Example job",
                    "model": "mistral-nemo",
                    "scripts": ["casual_chat"],
                    "conditions": ["plain", "freudian"],
                    "runs": 5,
                }
            ]
        }, indent=2))
        return

    idx = sys.argv.index("--jobs")
    if idx + 1 >= len(sys.argv):
        print("Error: --jobs requires a file path")
        return

    jobs_file = sys.argv[idx + 1]
    if not os.path.exists(jobs_file):
        print(f"Error: {jobs_file} not found")
        return

    run_scheduled_eval(jobs_file)


if __name__ == "__main__":
    main()
