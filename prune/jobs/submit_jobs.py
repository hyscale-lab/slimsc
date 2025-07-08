# slimsc/prune/jobs/submit_jobs.py
import subprocess
import os
import sys
import argparse
import textwrap
import time
import yaml # Using YAML for configuration
import shlex # For quoting paths safely
from typing import Optional

# --- Configuration ---
PROJECT_ROOT_REL_PATH = "../../.."
DEFAULT_JOBID_FILE = ".last_jobid"
LOGS_DIR_NAME = "logs"
CONDA_INIT_PATH = "$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME = "vllm"
PBS_PROJECT_PREFIX = "personal"
LD_LIBRARY_EXPORT_COMMAND_TEMPLATE = 'export LD_LIBRARY_PATH="$HOME/miniconda3/envs/' + CONDA_ENV_NAME + '/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"'

# --- Helper Functions ---

def check_existing_files(job_name_prefix: str, workdir: str, logs_subdir: str):
    """Check if potential output files exist for a combined job and return False if they do."""
    log_path_base = os.path.join(workdir, logs_subdir)
    filenames = [
        os.path.join(log_path_base, f"{job_name_prefix}.pbs"),
        os.path.join(log_path_base, f"{job_name_prefix}.log"),
        os.path.join(log_path_base, f"{job_name_prefix}_vllm_serve.log"),
        os.path.join(log_path_base, f"{job_name_prefix}_server_ip.txt"),
    ]

    existing = [fname for fname in filenames if os.path.exists(fname)]
    if existing:
        print("Error: The following file(s) already exist in the target logs directory:")
        for fname in existing: print(f"  {fname}")
        print(f"Please remove or rename them, or use a different name_prefix in the YAML.")
        return False
    return True

def read_previous_jobid(filename: str) -> str | None:
    """Reads the last job ID from a file."""
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                job_id = f.read().strip()
                if job_id:
                    print(f"Found previous job ID in {filename}: {job_id}")
                    return job_id
        except IOError as e:
            print(f"Warning: Could not read job ID file {filename}: {e}")
    return None

def write_jobid(filename: str, job_id: str):
    """Writes a job ID to a file, overwriting it."""
    try:
        with open(filename, 'w') as f:
            f.write(job_id + '\n')
        print(f"Saved current job ID {job_id} to {filename} for chaining.")
    except IOError as e:
        print(f"Warning: Could not write job ID to {filename}: {e}")

def create_combined_pbs_script(
    # Core Job Params
    job_name_prefix: str, dependency_job_id: str | None, logs_subdir: str,
    # Server Params
    model_path: str, tensor_parallel_size: int, server_hours: int, gpu_memory_utilization: float | None,
    enable_reasoning: bool, reasoning_parser: str | None,
    # Client Params
    client_gpus: int, eval_type: str, eval_script_args: dict,
    # Naming/Path Params
    base_output_dir: str, model_name: str, dataset_name: str, n_start: int,
    pruning_strategy: Optional[str], threshold: Optional[float],
    threshold_schedule: Optional[str], num_steps_to_delay_pruning_for_naming: Optional[int],
) -> tuple[str, str, str, str]:
    """
    Creates a single, combined PBS script for both server and client processes.
    - For 'sc_control', client runs on CPU within the server's job.
    - For 'similarity', GPUs are partitioned between server and client in one job.
    - Memory is not explicitly requested and relies on the cluster's default allocation per GPU.
    """
    # --- File Naming and Path Setup ---
    job_name = job_name_prefix
    relative_pbs_log_file = os.path.join(logs_subdir, f"{job_name}.log")
    relative_vllm_serve_log_file = os.path.join(logs_subdir, f"{job_name}_vllm_serve.log")
    relative_server_ip_file = os.path.join(logs_subdir, f"{job_name}_server_ip.txt")

    # --- Resource Calculation and GPU setup ---
    if eval_type == "similarity":
        total_gpus = tensor_parallel_size + client_gpus
        server_gpu_indices = ",".join(map(str, range(tensor_parallel_size)))
        client_gpu_indices = ",".join(map(str, range(tensor_parallel_size, total_gpus)))
        server_cuda_export = f'export CUDA_VISIBLE_DEVICES="{server_gpu_indices}"'
        client_cuda_export = f'export CUDA_VISIBLE_DEVICES="{client_gpu_indices}"'
        resource_select = f"select=1:ngpus={total_gpus}"
        gpu_request_line = f"#PBS -l {resource_select}"
    elif eval_type == "sc_control":
        total_gpus = tensor_parallel_size
        server_gpu_indices = ",".join(map(str, range(tensor_parallel_size)))
        server_cuda_export = f'export CUDA_VISIBLE_DEVICES="{server_gpu_indices}"'
        client_cuda_export = "# No GPU needed for sc_control client; it will use available CPUs."
        resource_select = f"select=1:ngpus={total_gpus}"
        gpu_request_line = f"#PBS -l {resource_select}"
    else:
        raise ValueError(f"Unsupported eval_type '{eval_type}'")

    # --- Run Name and Path Construction ---
    run_name: Optional[str] = None
    if eval_type == "similarity":
        schedule_suffix = f"_{threshold_schedule}" if threshold_schedule == 'annealing' else ""
        threshold_for_naming = 0.9 if threshold_schedule == 'annealing' else threshold
        run_name = f"{pruning_strategy}{schedule_suffix}_n{n_start}_thresh{threshold_for_naming:.2f}_delay{num_steps_to_delay_pruning_for_naming}"
    elif eval_type == "sc_control":
        run_name = f"sc_{n_start}_control"
    model_dataset_dir = os.path.join(base_output_dir, model_name, dataset_name, run_name or "unknown_run")
    target_kvc_file_path = os.path.join(model_dataset_dir, "kvcache_usages.csv")

    # --- vLLM Server Command ---
    vllm_parts = ["vllm", "serve", shlex.quote(model_path), f"--tensor-parallel-size {tensor_parallel_size}", "--port $PORT", "--seed 42"]
    if gpu_memory_utilization: vllm_parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")
    if enable_reasoning: vllm_parts.append("--enable-reasoning")
    if reasoning_parser: vllm_parts.append(f'--reasoning-parser {shlex.quote(reasoning_parser)}')
    vllm_base = " ".join(vllm_parts)
    vllm_exec = f"{vllm_base} > >(tee -a {shlex.quote(relative_vllm_serve_log_file)}) 2>&1 &"

    # --- Client Evaluation Command ---
    q_args = {k: shlex.quote(v) if isinstance(v, str) else v for k, v in eval_script_args.items()}
    eval_parts = []
    if eval_type == "similarity":
        eval_parts = ["python -m slimsc.prune.evaluation.similarity_prune_eval", f"--n_start {q_args['n_start']}", f"--threshold {q_args['threshold']}", f"--pruning_strategy {q_args['pruning_strategy']}", f"--tokenizer_path {q_args['tokenizer_path']}", f"--threshold_schedule {q_args.get('threshold_schedule', 'fixed')}"]
        if q_args.get('seed') is not None: eval_parts.append(f"--seed {q_args['seed']}")
        if q_args.get('num_steps_to_delay_pruning') is not None: eval_parts.append(f"--num_steps_to_delay_pruning {q_args['num_steps_to_delay_pruning']}")
    elif eval_type == "sc_control":
        eval_parts = ["python -m slimsc.prune.evaluation.sc_control_eval", f"--n_start {q_args['n_start']}"]
        if q_args.get('tokenizer_path'): eval_parts.append(f"--tokenizer_path {q_args['tokenizer_path']}")
    
    # Common args
    eval_parts.extend([f"--model_name {q_args['model_name']}", f"--model_identifier {q_args['model_identifier']}", "--vllm_url $VLLM_URL", f"--dataset_name {q_args['dataset_name']}"])
    if q_args.get("output_dir"): eval_parts.append(f"--output_dir {q_args['output_dir']}")
    if q_args.get("num_qns"): eval_parts.append(f"--num_qns {q_args['num_qns']}")
    eval_command = " ".join(filter(None, eval_parts))

    # --- Assemble the Combined PBS Script ---
    pbs_project = f"{PBS_PROJECT_PREFIX}-{os.environ.get('USER', 'default')}"
    dependency_directive = f"#PBS -W depend=afterok:{dependency_job_id}" if dependency_job_id else ""
    SERVER_READY_STRING = "Application startup complete."

    pbs_script_content = textwrap.dedent(f"""\
        #!/bin/bash
        {gpu_request_line}
        #PBS -l walltime={server_hours}:00:00
        #PBS -P {pbs_project}
        #PBS -q normal
        #PBS -N {job_name}
        #PBS -j oe
        #PBS -o {relative_pbs_log_file}
        {dependency_directive}

        cleanup() {{
            echo "[$(date)] Caught signal, attempting cleanup..."
            if [[ -n "$VLLM_PID" ]] && kill -0 $VLLM_PID > /dev/null 2>&1; then
                echo "Terminating vLLM server process group (PID: -$VLLM_PID)..."
                kill -TERM -$VLLM_PID && sleep 5
                kill -0 -$VLLM_PID > /dev/null 2>&1 && kill -KILL -$VLLM_PID
            fi
            echo "Cleanup function finished."
        }}
        trap cleanup SIGINT SIGTERM SIGHUP EXIT

        echo "--- PBS Combined Job Start ({job_name}) ---"
        cd $PBS_O_WORKDIR || exit 1
        source "{CONDA_INIT_PATH}" && conda activate {CONDA_ENV_NAME}

        # ======= 1. START SERVER ========
        echo "[$(date)] Setting up and starting vLLM server..."
        {server_cuda_export}
        export KVC_USAGE_FILE={shlex.quote(target_kvc_file_path)}
        {LD_LIBRARY_EXPORT_COMMAND_TEMPLATE}
        mkdir -p {shlex.quote(os.path.dirname(target_kvc_file_path))}
        
        HOST_IP=$(getent hosts "$(hostname -s)" | awk '{{print $1}}' || hostname -i)
        PORT_START=$((8000 + ($(echo $PBS_JOBID | cut -d. -f1) % 1000)))
        PORT=$(comm -23 <(seq $PORT_START $((PORT_START+100))) <(ss -tan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | head -n 1)
        if [ -z "$PORT" ]; then echo "No free port found!"; exit 1; fi
        export PORT
        echo "$HOST_IP:$PORT" > "{relative_server_ip_file}"
        export VLLM_URL="http://${{HOST_IP}}:${{PORT}}"
        
        echo "Server using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "Starting vLLM server on $VLLM_URL in background..."
        setsid {vllm_exec}
        VLLM_PID=$!
        sleep 5
        if ! kill -0 $VLLM_PID > /dev/null 2>&1; then echo "vLLM server failed to start!"; exit 1; fi

        # ===== 2. WAIT FOR SERVER READY ====
        echo "[$(date)] Waiting for server to become ready..."
        MAX_WAIT_SEC=900; INTERVAL=30; elapsed=0
        while ! grep -qF "{SERVER_READY_STRING}" "{relative_vllm_serve_log_file}"; do
            if ! kill -0 $VLLM_PID > /dev/null 2>&1; then echo "Server process died. Aborting."; exit 1; fi
            if [ $elapsed -ge $MAX_WAIT_SEC ]; then echo "Timeout waiting for server. Aborting."; exit 1; fi
            sleep $INTERVAL; elapsed=$((elapsed + INTERVAL))
            echo "Waited $elapsed/$MAX_WAIT_SEC seconds..."
        done
        echo "[$(date)] Server is ready."

        # ======= 3. RUN CLIENT ==========
        echo "[$(date)] Setting up and running evaluation client..."
        cd "{PROJECT_ROOT_REL_PATH}" || exit 1
        
        {client_cuda_export}
        echo "Client using CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-'N/A (CPU job)'}}"
        echo "Running Evaluation Command: {eval_command}"

        {eval_command}
        EVAL_EXIT_CODE=$?
        echo "[$(date)] Evaluation script exited with code: $EVAL_EXIT_CODE"

        echo "--- PBS Combined Job Finished ---"
        exit $EVAL_EXIT_CODE
    """)
    return pbs_script_content, relative_server_ip_file, relative_pbs_log_file, relative_vllm_serve_log_file

def write_pbs_script(job_name_prefix: str, pbs_script_content: str, workdir: str, logs_subdir: str) -> str:
    """Writes the PBS script content to a file."""
    log_path_base = os.path.join(workdir, logs_subdir)
    os.makedirs(log_path_base, exist_ok=True)
    pbs_script_path = os.path.join(log_path_base, f"{job_name_prefix}.pbs")
    try:
        with open(pbs_script_path, "w") as f: f.write(pbs_script_content)
        print(f"PBS script written to: {pbs_script_path}")
        return pbs_script_path
    except IOError as e:
        print(f"Error writing PBS script {pbs_script_path}: {e}")
        sys.exit(1)

def submit_pbs_job(pbs_script_path: str) -> str | None:
    """Submits the PBS job."""
    submit_command = ["qsub", pbs_script_path]
    print(f"Submitting command: {' '.join(submit_command)}")
    try:
        submission_dir = os.path.dirname(os.path.dirname(pbs_script_path))
        process = subprocess.run(submit_command, capture_output=True, text=True, check=True, cwd=submission_dir)
        job_id = process.stdout.strip()
        print(f"PBS Job submitted: {pbs_script_path} -> Job ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job {pbs_script_path}:\n{e.stderr}")
        return None
    except Exception as e:
        print(f"Exception during job submission for {pbs_script_path}: {e}")
        return None

def get_config_value(cfg, keys, default=None):
    """Safely get a nested value from a dictionary."""
    val = cfg
    try:
        for key in keys: val = val[key]
        return val if val is not None else default
    except (KeyError, TypeError): return default

def validate_job_config(job_config, job_name_prefix, eval_type):
     """Checks for essential keys in the job config dictionary."""
     if not get_config_value(job_config, ['model_path']):
         print(f"Error: 'model_path' missing for job '{job_name_prefix}'. Skipping."); return False
     eval_cfg = job_config.get('eval', {})
     required = ['n_start', 'model_name', 'model_identifier', 'dataset_name']
     if eval_type == 'similarity': required.extend(['threshold', 'pruning_strategy', 'tokenizer_path'])
     missing = [arg for arg in required if arg not in eval_cfg]
     if missing:
         print(f"Error: Missing eval args for '{job_name_prefix}': {missing}. Skipping."); return False
     return True

def main_yaml():
    """Main function to orchestrate job submission from YAML using an optimized single-job approach."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Submit optimized single-job PBS tasks from a YAML file.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", default="experiments.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--jobid_file", default=DEFAULT_JOBID_FILE, help="File for chaining jobs between script runs.")
    parser.add_argument("--start_job_index", type=int, default=0, help="0-based index of the job to start from.")
    parser.add_argument("--max_jobs", type=int, default=None, help="Max number of jobs to process.")
    cli_args = parser.parse_args()

    config_path = os.path.join(script_dir, cli_args.config)
    jobid_file_path = os.path.join(script_dir, cli_args.jobid_file) if cli_args.jobid_file else None
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
        if 'jobs' not in config: raise ValueError("YAML must contain a top-level 'jobs' list.")
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"Error loading YAML file {config_path}: {e}"); sys.exit(1)

    all_jobs = config.get('jobs', [])
    start_index = max(0, cli_args.start_job_index)
    end_index = min(start_index + (cli_args.max_jobs or len(all_jobs)), len(all_jobs))
    jobs_to_process = all_jobs[start_index:end_index]
    if not jobs_to_process:
        print("No jobs selected to process."); sys.exit(0)
    print(f"Processing {len(jobs_to_process)} jobs (index {start_index} to {end_index - 1}).")

    previous_job_id = read_previous_jobid(jobid_file_path)
    last_successful_job_id = None
    workdir = script_dir

    for i, job_config in enumerate(jobs_to_process):
        current_job_index = start_index + i
        print(f"\n===== Processing Job {current_job_index + 1}/{len(all_jobs)} =====")

        job_name_prefix = get_config_value(job_config, ['name_prefix'], f"yaml_job_{current_job_index+1}")
        eval_cfg = job_config.get('eval', {})
        eval_type = get_config_value(eval_cfg, ['type'])

        if eval_type not in ['similarity', 'sc_control']:
            print(f"Error: Invalid 'eval.type' for '{job_name_prefix}'. Skipping."); continue
        if not validate_job_config(job_config, job_name_prefix, eval_type): continue
        if not check_existing_files(job_name_prefix, workdir, LOGS_DIR_NAME):
             print(f"Skipping job '{job_name_prefix}' due to existing files."); continue

        # --- Extract All Parameters ---
        server_cfg = job_config.get('server', {})
        client_cfg = job_config.get('client', {})
        # Similarity-specific params (set to None if not similarity)
        eval_pruning_strategy, eval_threshold, eval_threshold_schedule, eval_num_steps_to_delay_pruning_for_naming = (None,) * 4
        if eval_type == 'similarity':
            eval_pruning_strategy = get_config_value(eval_cfg, ['pruning_strategy'])
            eval_threshold = get_config_value(eval_cfg, ['threshold'])
            eval_threshold_schedule = get_config_value(eval_cfg, ['threshold_schedule'], 'fixed')
            eval_num_steps_to_delay_pruning_for_naming = get_config_value(eval_cfg, ['num_steps_to_delay_pruning'], 20)

        # --- Create and Submit the Single Combined Job ---
        combined_pbs_content, _, _, _ = create_combined_pbs_script(
            job_name_prefix=job_name_prefix, dependency_job_id=previous_job_id, logs_subdir=LOGS_DIR_NAME,
            model_path=get_config_value(job_config, ['model_path']),
            tensor_parallel_size=get_config_value(server_cfg, ['tensor_parallel_size'], 2),
            server_hours=get_config_value(server_cfg, ['hours'], 8),
            gpu_memory_utilization=get_config_value(server_cfg, ['gpu_memory_utilization']),
            enable_reasoning=get_config_value(server_cfg, ['enable_reasoning'], False),
            reasoning_parser=get_config_value(server_cfg, ['reasoning_parser']),
            client_gpus=get_config_value(client_cfg, ['gpus'], 1),
            eval_type=eval_type, eval_script_args=eval_cfg,
            base_output_dir=get_config_value(eval_cfg, ['output_dir'], os.path.join(os.path.expanduser("~"), "slimsc/prune/results")),
            model_name=get_config_value(eval_cfg, ['model_name']),
            dataset_name=get_config_value(eval_cfg, ['dataset_name']),
            n_start=get_config_value(eval_cfg, ['n_start']),
            pruning_strategy=eval_pruning_strategy, threshold=eval_threshold,
            threshold_schedule=eval_threshold_schedule,
            num_steps_to_delay_pruning_for_naming=eval_num_steps_to_delay_pruning_for_naming
        )

        pbs_script_path = write_pbs_script(job_name_prefix, combined_pbs_content, workdir, LOGS_DIR_NAME)
        new_job_id = submit_pbs_job(pbs_script_path)

        if not new_job_id:
            print(f"Error submitting job '{job_name_prefix}'. Stopping further processing."); break

        print(f"Successfully submitted combined job for '{job_name_prefix}': Job ID = {new_job_id}")
        previous_job_id = new_job_id
        last_successful_job_id = new_job_id
        time.sleep(1)

    if jobid_file_path and last_successful_job_id:
        write_jobid(jobid_file_path, last_successful_job_id)
    elif jobid_file_path:
         print(f"\nNo jobs were successfully submitted. Job ID file not updated.")
    print("\n===== YAML Job Submission Finished =====")

if __name__ == "__main__":
    main_yaml()