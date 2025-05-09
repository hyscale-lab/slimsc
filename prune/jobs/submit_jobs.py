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
# These paths are relative to where the PBS job starts ($PBS_O_WORKDIR),
# which will be the directory where this script is run (e.g., slimsc/prune/jobs/).
# Path to the project root directory (containing the 'slimsc' package)
PROJECT_ROOT_REL_PATH = "../../.."
# Default location for the job ID chaining file, relative to this script's location
DEFAULT_JOBID_FILE = ".last_client_jobid"
# Subdirectory name for logs, scripts, etc., relative to this script's location
LOGS_DIR_NAME = "logs"

# Conda configuration (adjust if necessary)
CONDA_INIT_PATH = "/home/users/ntu/{user}/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME = "vllm"
# PBS Project configuration (adjust if necessary)
PBS_PROJECT_PREFIX = "personal"

LD_LIBRARY_EXPORT_COMMAND_TEMPLATE = 'export LD_LIBRARY_PATH="/home/users/ntu/{user}/miniconda3/envs/' + CONDA_ENV_NAME + '/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"'

# --- Helper Functions ---

def check_existing_files(job_name_prefix: str, eval_type: str, workdir: str, logs_subdir: str):
    """
    Check if potential output files exist in the logs subdirectory
    within the working directory and exit if they do.
    """
    log_path_base = os.path.join(workdir, logs_subdir)
    # Files expected in the logs subdirectory
    filenames = [
        os.path.join(log_path_base, f"{job_name_prefix}_server.pbs"),
        os.path.join(log_path_base, f"{job_name_prefix}_client.pbs"),
        os.path.join(log_path_base, f"{job_name_prefix}_server.log"),
        os.path.join(log_path_base, f"{job_name_prefix}_client.log"),
        os.path.join(log_path_base, f"{job_name_prefix}_vllm_serve.log"), # Check potential vllm log too
        os.path.join(log_path_base, f"{job_name_prefix}_server_ip.txt"),
    ]
    # Note: Actual evaluation output files depend on the --output_dir argument
    # passed to the eval script and how that script constructs paths.

    existing = [fname for fname in filenames if os.path.exists(fname)]
    if existing:
        print("Error: The following file(s) already exist in the target logs directory:")
        for fname in existing:
            print(f"  {fname}")
        print(f"Job Name Prefix: {job_name_prefix}")
        print(f"Target Directory: {log_path_base}")
        print("Please remove or rename them, or use a different name_prefix in the YAML.")
        # Returning False instead of exiting to allow loop continuation in main_yaml
        return False
    return True


def read_previous_jobid(filename: str) -> str | None:
    """Reads the last job ID from a file."""
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                job_id = f.read().strip()
                if job_id:
                    print(f"Found previous client job ID in {filename}: {job_id}")
                    return job_id
                else:
                    print(f"Warning: Job ID file {filename} is empty.")
                    return None
        except IOError as e:
            print(f"Warning: Could not read job ID file {filename}: {e}")
            return None
    return None


def write_jobid(filename: str, job_id: str):
    """Writes a job ID to a file, overwriting it."""
    if not filename:
        filename = DEFAULT_JOBID_FILE
    try:
        with open(filename, 'w') as f:
            f.write(job_id + '\n')
        print(f"Saved current client job ID {job_id} to {filename} for chaining.")
    except IOError as e:
        print(f"Warning: Could not write job ID to {filename}: {e}")


def create_server_pbs_script(
    job_name_prefix: str,
    model_path: str,
    tensor_parallel_size: int,
    server_hours: int,
    gpu_memory_utilization: float | None, # Allow None
    enable_reasoning: bool,
    reasoning_parser: str | None,
    vllm_use_v1: bool,
    dependency_job_id: str | None,
    logs_subdir: str,
    eval_type: str,
    base_output_dir: str,
    model_name: str,
    dataset_name: str,
    n_start: int,
    # --- Similarity specific (can be None if eval_type is sc_control) ---
    pruning_strategy: Optional[str],
    threshold: Optional[float],
) -> tuple[str, str, str, str]: # Return: pbs_script_content, relative_ip_file, relative_pbs_log_file, relative_vllm_serve_log_file
    """
    Creates the PBS script content for the vLLM server job.
    Paths returned are relative to the script's execution directory ($PBS_O_WORKDIR),
    including the logs_subdir.
    """
    # File paths relative to $PBS_O_WORKDIR, inside the logs_subdir
    server_job_name = f"{job_name_prefix}_server"
    relative_pbs_log_file = os.path.join(logs_subdir, f"{server_job_name}.log") # Main log for PBS script steps
    relative_vllm_serve_log_file = os.path.join(logs_subdir, f"{job_name_prefix}_vllm_serve.log") # Specific log for vllm serve output
    relative_server_ip_file = os.path.join(logs_subdir, f"{job_name_prefix}_server_ip.txt") # File for server IP

    run_name: Optional[str] = None
    if eval_type == "similarity":
        run_name = f"{pruning_strategy}_n{n_start}_thresh{threshold:.2f}"
    elif eval_type == "sc_control":
        # Assuming sc_control run_name is based on n_start
        run_name = f"sc_{n_start}_control"

    model_dataset_dir = os.path.join(base_output_dir, model_name, dataset_name, run_name)
    target_kvc_file_path = os.path.join(model_dataset_dir, "kvcache_usages.csv")
    quoted_target_kvc_file_path = shlex.quote(target_kvc_file_path)

    # Quote paths that might contain spaces or special characters for shell safety
    quoted_model_path = shlex.quote(model_path)
    quoted_reasoning_parser = shlex.quote(reasoning_parser) if reasoning_parser else None
    quoted_vllm_serve_log = shlex.quote(relative_vllm_serve_log_file)
    quoted_server_ip_file = shlex.quote(relative_server_ip_file)

    user = os.environ.get("USER", "default_user")
    conda_init_script = CONDA_INIT_PATH.format(user=user)
    pbs_project = f"{PBS_PROJECT_PREFIX}-{user}"
    formatted_ld_export_command = LD_LIBRARY_EXPORT_COMMAND_TEMPLATE.format(user=user)

    dependency_directive = ""
    if dependency_job_id:
        dependency_directive = f"#PBS -W depend=afterok:{dependency_job_id}"

    # Construct vLLM Command
    vllm_command_parts = [
        "vllm", "serve", f'{quoted_model_path}',
        f"--tensor-parallel-size {tensor_parallel_size}",
        "--port 8000", # Explicitly set port
        "--seed 42"
    ]
    if gpu_memory_utilization is not None and 0 < gpu_memory_utilization <= 1:
         vllm_command_parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")
    if enable_reasoning:
        vllm_command_parts.append("--enable-reasoning")
    if quoted_reasoning_parser:
        vllm_command_parts.append(f'--reasoning-parser {quoted_reasoning_parser}')

    # The base command without redirection
    vllm_base_command = " ".join(vllm_command_parts)

    # Redirection using tee to capture stdout/stderr to the specific log file,
    # and run in the background (&)
    # Ensure the log directory exists before redirecting (although PBS should handle the main log dir)
    vllm_redirection = f"> >(tee -a {quoted_vllm_serve_log}) 2>&1 &"

    # Full command line to execute in the script
    vllm_exec_command = f"{vllm_base_command} {vllm_redirection}"
    # -----------------------------

    exports = [
        f"export CUDA_VISIBLE_DEVICES=$(seq -s , 0 {tensor_parallel_size - 1})",
        f"export KVC_USAGE_FILE={quoted_target_kvc_file_path}",
        formatted_ld_export_command,
    ]
    if not vllm_use_v1:
        exports.append("export VLLM_USE_V1=0")
    
    create_run_dir_command = f'mkdir -p {shlex.quote(os.path.dirname(target_kvc_file_path))}'

    # --- PBS Script Content ---
    # NOTE: All paths referenced inside the script now need to be relative to $PBS_O_WORKDIR
    # and include the logs_subdir prefix where appropriate.
    pbs_script_content = f"""#!/bin/bash
#PBS -l select=1:ngpus={tensor_parallel_size}
#PBS -l walltime={server_hours}:00:00
#PBS -P {pbs_project}
#PBS -q normal
#PBS -N {server_job_name}
#PBS -j oe
#PBS -o {relative_pbs_log_file}
{dependency_directive}

# Define relative paths used within the script
VLLM_SERVE_LOG_RELPATH="{relative_vllm_serve_log_file}"
SERVER_IP_FILE_RELPATH="{relative_server_ip_file}"
MAIN_PBS_LOG_RELPATH="{relative_pbs_log_file}"

# Function to clean up vLLM process on script exit/termination
cleanup() {{
    echo "[$(date)] Caught signal, attempting cleanup..."
    if [[ -n "$VLLM_PID" ]] && kill -0 $VLLM_PID > /dev/null 2>&1; then
        echo "Attempting to terminate vLLM server process group (PID: -$VLLM_PID)..."
        if kill -TERM -$VLLM_PID; then
            echo "Sent TERM signal to process group -$VLLM_PID."
            sleep 5
            if kill -0 -$VLLM_PID > /dev/null 2>&1; then
                echo "Process group -$VLLM_PID still running after TERM, sending KILL signal."
                kill -KILL -$VLLM_PID
            else
                echo "Process group -$VLLM_PID terminated gracefully."
            fi
        else
            echo "Failed to send TERM signal (maybe already stopped)."
        fi
    elif [[ -n "$VLLM_PID" ]]; then
         echo "vLLM server process (PID: $VLLM_PID) was not found during cleanup."
    else
         echo "VLLM_PID variable not set, cannot perform cleanup."
    fi
    echo "Cleanup function finished."
}}

# Trap signals (INTERRUPT, TERM, HUP, EXIT) to call the cleanup function
trap cleanup SIGINT SIGTERM SIGHUP EXIT

echo "--- PBS Server Job Start ---"
echo "Job ID: $PBS_JOBID"
echo "Job Name: {server_job_name}"
{f'echo "Depends on Job ID: {dependency_job_id}"' if dependency_job_id else ""}
echo "Running on host: $(hostname)"
echo "PBS work directory: $PBS_O_WORKDIR"
echo "Main PBS Log File: $PBS_O_WORKDIR/$MAIN_PBS_LOG_RELPATH"
echo "vLLM Serve Output Log: $PBS_O_WORKDIR/$VLLM_SERVE_LOG_RELPATH" # Indicate the specific log
echo "Server IP File: $PBS_O_WORKDIR/$SERVER_IP_FILE_RELPATH"
echo "Eval Type: {eval_type}" # Log eval type
echo "Run Name: {run_name if run_name else 'Error: Could not determine'}" # Log derived run name
echo "Target KVC Usage File: {target_kvc_file_path}" # Log the target path
echo "----------------------------"

# Go to the submission directory
cd $PBS_O_WORKDIR || {{ echo "Error changing to $PBS_O_WORKDIR"; exit 1; }}

# Ensure the logs directory exists (PBS might create it for -o, but good practice)
mkdir -p "{logs_subdir}" || {{ echo "Error creating logs directory {logs_subdir}"; exit 1; }}

# Ensure the vLLM log file is clean before starting
echo "Checking for existing vLLM serve log file..."
if [ -f "$VLLM_SERVE_LOG_RELPATH" ]; then
    echo "Removing existing file: $VLLM_SERVE_LOG_RELPATH"
    rm -f "$VLLM_SERVE_LOG_RELPATH"
fi

# Ensure Run Directory Exists
echo "Ensuring target run directory for KVC log exists..."
{create_run_dir_command} || {{ echo "Error creating target run directory for KVC log: {os.path.dirname(target_kvc_file_path)}"; exit 1; }}
echo "Directory ensured: {os.path.dirname(target_kvc_file_path)}"

# Activate Conda
echo "Sourcing Conda..."
source "{conda_init_script}" || {{ echo "Error sourcing conda"; exit 1; }}
echo "Activating Conda environment: {CONDA_ENV_NAME}..."
conda activate {CONDA_ENV_NAME} || {{ echo "Error activating conda env"; exit 1; }}
conda info | grep "active environment"

# Set Environment Variables
echo "Setting environment variables..."
{textwrap.indent(os.linesep.join(exports), '    ')}

# Get IP Address
echo "Retrieving host IP address..."
HOST_NAME=$(hostname -s)
HOST_IP=$(getent hosts "$HOST_NAME" | awk '{{print $1}}' | head -n 1 || hostname -i | awk '{{print $1}}') # Fallback
if [ -z "$HOST_IP" ]; then
   echo "Error: Unable to determine IP address." >&2 # Error to stderr (goes to pbs_log_file)
   echo "ERROR_NO_IP" > "$SERVER_IP_FILE_RELPATH"
   exit 1
fi
echo "Host IP: $HOST_IP"
echo "$HOST_IP" > "$SERVER_IP_FILE_RELPATH"
echo "IP address saved to $SERVER_IP_FILE_RELPATH"

# Start vLLM Server in Background using setsid
echo "Starting vLLM server in background..."
echo "Command: {vllm_base_command}"
echo "Output directed to: $VLLM_SERVE_LOG_RELPATH"

# Use setsid to run vllm in a new session and process group.
setsid {vllm_exec_command}

# Capture the PID of the background process (setsid)
VLLM_PID=$!
echo "vLLM server process group started with PID: $VLLM_PID"

# Brief check if the process started successfully
sleep 2 # Give it a moment to potentially fail immediately
if ! kill -0 $VLLM_PID > /dev/null 2>&1; then
    echo "Error: vLLM server process (PID $VLLM_PID) not found shortly after start." >&2
    echo "Check logs:" >&2
    echo "  vLLM output: $PBS_O_WORKDIR/$VLLM_SERVE_LOG_RELPATH" >&2
    echo "  PBS script log: $PBS_O_WORKDIR/$MAIN_PBS_LOG_RELPATH" >&2
    exit 1
fi

echo "[$(date)] vLLM Server started (PID: $VLLM_PID). Waiting for termination signal or job end..."

# Wait for the backgrounded vLLM process (or rather, the setsid leader) to finish.
wait $VLLM_PID
WAIT_EXIT_CODE=$?

echo "[$(date)] 'wait \\$VLLM_PID' command finished with exit code: $WAIT_EXIT_CODE."

# --- Delete vLLM Serve Log ---
# This happens after the wait command finishes, meaning the server process ended
# (either normally, via cancellation triggering cleanup, or error).
# We delete it regardless of exit code, as requested (delete once server.log is saved).
# The main server.log (-o target) should already contain most script execution logs.
echo "[$(date)] Attempting to delete vLLM serve log file: $VLLM_SERVE_LOG_RELPATH"
if [ -f "$VLLM_SERVE_LOG_RELPATH" ]; then
    rm -f "$VLLM_SERVE_LOG_RELPATH"
    if [ $? -eq 0 ]; then
        echo "Successfully deleted $VLLM_SERVE_LOG_RELPATH."
    else
        echo "Warning: Failed to delete $VLLM_SERVE_LOG_RELPATH." >&2
    fi
else
    echo "vLLM serve log file $VLLM_SERVE_LOG_RELPATH not found, nothing to delete."
fi
# ----------------------------

echo "--- PBS Server Job Finished ---"

# Explicitly exit with the wait command's status (optional, trap EXIT handles cleanup)
# exit $WAIT_EXIT_CODE
"""
    # Return the path to the specific log file vLLM writes to, relative to $PBS_O_WORKDIR
    return pbs_script_content, relative_server_ip_file, relative_pbs_log_file, relative_vllm_serve_log_file


def create_client_pbs_script(
    job_name_prefix: str,
    server_job_id: str,
    relative_server_ip_file: str, # Relative to $PBS_O_WORKDIR (includes logs_subdir)
    relative_vllm_serve_log_file: str, # Relative to $PBS_O_WORKDIR (includes logs_subdir)
    client_hours: int,
    client_gpus: int,
    client_mem: str,
    eval_type: str,
    eval_script_args: dict, # Contains eval args from YAML
    initial_wait_seconds: int,
    logs_subdir: str, # New argument
) -> str:
    """
    Creates the PBS script content for the evaluation client job.
    Paths are relative to $PBS_O_WORKDIR and include the logs_subdir.
    """
    # File paths are relative to $PBS_O_WORKDIR, inside logs_subdir
    client_job_name = f"{job_name_prefix}_client"
    relative_client_log_file = os.path.join(logs_subdir, f"{client_job_name}.log")
    # Use the passed relative paths for server files
    server_ip_file_to_check = relative_server_ip_file
    server_log_to_check = relative_vllm_serve_log_file # Client checks the specific vllm log
    # The main server PBS log (for debugging client-side if needed)
    relative_main_server_pbs_log = os.path.join(logs_subdir, f"{job_name_prefix}_server.log")

    user = os.environ.get("USER", "default_user")
    conda_init_script = CONDA_INIT_PATH.format(user=user)
    pbs_project = f"{PBS_PROJECT_PREFIX}-{user}"
    SERVER_READY_STRING = "Application startup complete."

    # --- Determine Resource Request ---
    # (Resource request logic remains the same)
    if eval_type == "sc_control":
        resource_select = "select=1:ncpus=2"
        mem_request = client_mem if client_mem else "8gb"
        resource_mem = f":mem={mem_request}"
        gpu_request_line = f"#PBS -l {resource_select}{resource_mem}"
        cuda_export = "# No GPU needed for sc_control client"
    else:
        client_gpus = max(1, client_gpus)
        resource_select = f"select=1:ngpus={client_gpus}"
        mem_request = client_mem if client_mem else "8gb"
        resource_mem = f":mem={mem_request}"
        gpu_request_line = f"#PBS -l {resource_select}{resource_mem}"
        cuda_export = f"export CUDA_VISIBLE_DEVICES=$(seq -s , 0 {client_gpus - 1})"

    # --- Construct Evaluation Command ---
    # Ensure paths in args are quoted if necessary
    quoted_eval_args = {}
    for k, v in eval_script_args.items():
        if isinstance(v, str):
            quoted_eval_args[k] = shlex.quote(v)
        else:
            quoted_eval_args[k] = v # Keep numbers, etc., as is

    # Build command using quoted args
    if eval_type == "similarity":
        eval_module = "slimsc.prune.evaluation.similarity_prune_eval"
        eval_command_parts = [
            "python", "-m", eval_module,
            f"--n_start {quoted_eval_args['n_start']}",
            f"--threshold {quoted_eval_args['threshold']}",
            f"--pruning_strategy {quoted_eval_args['pruning_strategy']}",
            f"--model_name {quoted_eval_args['model_name']}",
            f"--model_identifier {quoted_eval_args['model_identifier']}",
            f"--tokenizer_path {quoted_eval_args['tokenizer_path']}",
            "--vllm_url $VLLM_URL", # Use exported variable
            f"--dataset_name {quoted_eval_args['dataset_name']}",
        ]
    elif eval_type == "sc_control":
        eval_module = "slimsc.prune.evaluation.sc_control_eval"
        eval_command_parts = [
            "python", "-m", eval_module,
            f"--n_start {quoted_eval_args['n_start']}",
            f"--model_name {quoted_eval_args['model_name']}",
            f"--model_identifier {quoted_eval_args['model_identifier']}",
            f"--tokenizer_path {quoted_eval_args['tokenizer_path']}" if quoted_eval_args.get('tokenizer_path') else "",
            "--vllm_url $VLLM_URL", # Use exported variable
            f"--dataset_name {quoted_eval_args['dataset_name']}",
        ]
        eval_command_parts = [part for part in eval_command_parts if part] # Clean empty
    else:
        raise ValueError(f"Invalid eval_type '{eval_type}' in create_client_pbs_script")

    # Add common optional arguments using quoted_eval_args
    if quoted_eval_args.get("output_dir"):
        eval_command_parts.append(f"--output_dir {quoted_eval_args['output_dir']}")
    if quoted_eval_args.get("num_qns"):
        eval_command_parts.append(f"--num_qns {quoted_eval_args['num_qns']}")
    elif quoted_eval_args.get("iterations"):
        eval_command_parts.append(f"--iterations {quoted_eval_args['iterations']}")
    else:
        if quoted_eval_args.get("start"):
            eval_command_parts.append(f"--start {quoted_eval_args['start']}")
        if quoted_eval_args.get("end"):
            eval_command_parts.append(f"--end {quoted_eval_args['end']}")

    eval_command = " ".join(eval_command_parts)

    # --- Start PBS Script Content ---
    # Note: All paths referenced are relative to $PBS_O_WORKDIR
    pbs_script_content = f"""#!/bin/bash
{gpu_request_line}
#PBS -l walltime={client_hours}:00:00
#PBS -P {pbs_project}
#PBS -q normal
#PBS -N {client_job_name}
#PBS -j oe
#PBS -o {relative_client_log_file}
#PBS -W depend=after:{server_job_id}

# Define relative paths used within the script
SERVER_IP_FILE_RELPATH="{server_ip_file_to_check}"
SERVER_VLLM_LOG_RELPATH="{server_log_to_check}"
SERVER_PBS_LOG_RELPATH="{relative_main_server_pbs_log}"
CLIENT_LOG_RELPATH="{relative_client_log_file}"

export SERVER_JOB_ID_TO_CANCEL="{server_job_id}"

echo "--- PBS Client Job Start ---"
echo "Job ID: $PBS_JOBID"
echo "Job Name: {client_job_name}"
echo "Depends on Server Job ID: {server_job_id}"
echo "Evaluation Type: {eval_type}"
echo "Running on host: $(hostname)"
echo "PBS work directory: $PBS_O_WORKDIR"
echo "Client Log File: $PBS_O_WORKDIR/$CLIENT_LOG_RELPATH"
echo "Reading Server IP From: $PBS_O_WORKDIR/$SERVER_IP_FILE_RELPATH"
echo "Checking Server Output Log: $PBS_O_WORKDIR/$SERVER_VLLM_LOG_RELPATH" # Client checks this specific log
echo "Project Root Relative Path: {PROJECT_ROOT_REL_PATH}"
echo "Server Job to Cancel on Success: $SERVER_JOB_ID_TO_CANCEL"
echo "Server Ready String to Check: '{SERVER_READY_STRING}'"
echo "----------------------------"

cd $PBS_O_WORKDIR || {{ echo "Error changing to $PBS_O_WORKDIR"; exit 1; }}
echo "Current directory: $(pwd)"

# Ensure the logs directory exists (should be created by server, but belt-and-suspenders)
mkdir -p "{logs_subdir}" || {{ echo "Error ensuring logs directory {logs_subdir} exists"; exit 1; }}

# --- Conda Activation ---
echo "Sourcing Conda..."
source "{conda_init_script}" || {{ echo "Error sourcing conda"; exit 1; }}
echo "Activating Conda environment: {CONDA_ENV_NAME}..."
conda activate {CONDA_ENV_NAME} || {{ echo "Error activating conda env"; exit 1; }}
conda info | grep "active environment"
{cuda_export}

INITIAL_WAIT_SECONDS={initial_wait_seconds}
if [ "$INITIAL_WAIT_SECONDS" -gt 0 ]; then
    echo "[$(date)] Starting initial wait period of $INITIAL_WAIT_SECONDS seconds before polling server..."
    sleep $INITIAL_WAIT_SECONDS
    echo "[$(date)] Initial wait finished."
else
    echo "[$(date)] No initial wait period configured (or set to 0)."
fi

# --- Wait for Server IP File (Phase 1) ---
echo "Waiting for server IP file: $SERVER_IP_FILE_RELPATH"
MAX_IP_WAIT_SEC=180 # 3 minutes timeout for IP file
IP_WAIT_INTERVAL=60 # Check less frequently now
elapsed_ip_wait=0
server_ip_found=0
while [ $server_ip_found -eq 0 ]; do
    # Check if server job still exists
    qstat "$SERVER_JOB_ID_TO_CANCEL" > /dev/null 2>&1
    SERVER_JOB_EXISTS=$?
    if [ $SERVER_JOB_EXISTS -ne 0 ]; then
         echo "[$(date)] Error: Server job $SERVER_JOB_ID_TO_CANCEL no longer exists while waiting for IP file '$SERVER_IP_FILE_RELPATH'." >&2
         # Optional: Check server logs for early failure clues
         vllm_server_log="$SERVER_VLLM_LOG_RELPATH" # vLLM log might have been deleted already
         main_pbs_server_log="$SERVER_PBS_LOG_RELPATH"
         if [ -f "$vllm_server_log" ]; then echo "Last lines of vLLM log ($vllm_server_log):"; tail -n 20 "$vllm_server_log"; else echo "vLLM log ($vllm_server_log) not found or deleted by server."; fi >&2
         if [ -f "$main_pbs_server_log" ]; then echo "Last lines of main server PBS log ($main_pbs_server_log):"; tail -n 20 "$main_pbs_server_log"; else echo "Main server PBS log ($main_pbs_server_log) not found."; fi >&2
         exit 1
    fi

    # Check if IP file exists and is non-empty
    if [ -s "$SERVER_IP_FILE_RELPATH" ]; then
        # Check if file contains error marker written by server script
        if grep -q "ERROR_NO_IP" "$SERVER_IP_FILE_RELPATH"; then
            echo "[$(date)] Error: Server IP file '$SERVER_IP_FILE_RELPATH' contains error marker." >&2
            exit 1
        fi
        echo "[$(date)] Found non-empty server IP file: $SERVER_IP_FILE_RELPATH."
        server_ip_found=1
    else
        # Check for timeout
        if [ $elapsed_ip_wait -ge $MAX_IP_WAIT_SEC ]; then
            echo "[$(date)] Error: Timeout ($MAX_IP_WAIT_SEC sec) waiting for non-empty server IP file $SERVER_IP_FILE_RELPATH" >&2
            echo "Server job $SERVER_JOB_ID_TO_CANCEL status:" >&2
            qstat "$SERVER_JOB_ID_TO_CANCEL" >&2 || echo "Server job $SERVER_JOB_ID_TO_CANCEL not found by qstat." >&2
            exit 1
        fi
        echo "[$(date)] Waiting for $SERVER_IP_FILE_RELPATH to be created/populated by server job $SERVER_JOB_ID_TO_CANCEL... ($elapsed_ip_wait/$MAX_IP_WAIT_SEC sec)"
        sleep $IP_WAIT_INTERVAL
        elapsed_ip_wait=$((elapsed_ip_wait + IP_WAIT_INTERVAL))
    fi
done

# Read IP and set URL
SERVER_IP=$(cat "$SERVER_IP_FILE_RELPATH")
if [ -z "$SERVER_IP" ] || [ "$SERVER_IP" == "ERROR_NO_IP" ]; then # Double check
    echo "[$(date)] Error: Server IP is empty or indicates an error after file found." >&2
    exit 1
fi
export VLLM_URL="http://${{SERVER_IP}}:8000"
echo "Read Server IP: $SERVER_IP"
echo "Set VLLM_URL=$VLLM_URL"

# --- Wait for Server Ready String in Log (Phase 2) ---
# IMPORTANT: The server might delete SERVER_VLLM_LOG_RELPATH upon finishing.
# The client needs to see the "ready" string before the server finishes and deletes the log.
# If the server starts fast and the client has a delay, this check might fail after the server exits.
# We add a check to see if the server job still exists during the wait loop.
echo "Waiting for vLLM server to be ready (checking for '{SERVER_READY_STRING}' in $SERVER_VLLM_LOG_RELPATH)..."
MAX_LOG_WAIT_SEC=600 # 10 minutes total timeout for server to become ready
LOG_WAIT_INTERVAL=30
elapsed_log_wait=0
server_ready=0

while [ $server_ready -eq 0 ]; do
    # Check if server job still exists before checking the log
    qstat "$SERVER_JOB_ID_TO_CANCEL" > /dev/null 2>&1
    SERVER_JOB_EXISTS=$?
    if [ $SERVER_JOB_EXISTS -ne 0 ]; then
         echo "[$(date)] Warning: Server job $SERVER_JOB_ID_TO_CANCEL no longer exists while waiting for ready string in '$SERVER_VLLM_LOG_RELPATH'." >&2
         echo "The vLLM log might have been deleted by the server upon exit." >&2
         echo "Assuming server might have been ready if IP file was created." >&2
         echo "Proceeding with caution - evaluation may fail if server wasn't actually ready." >&2
         # Decide whether to proceed or fail here. Proceeding cautiously.
         # If we proceed, the eval command will likely fail quickly if the server isn't up.
         server_ready=1 # Break the loop and try to run eval
         continue
    fi

    # Check if the specific vLLM log file exists and contains the ready string
    # Use -F for fixed string search, -q for quiet (just exit status)
    if [ -f "$SERVER_VLLM_LOG_RELPATH" ] && grep -qF "{SERVER_READY_STRING}" "$SERVER_VLLM_LOG_RELPATH"; then
        echo "[$(date)] Found '{SERVER_READY_STRING}' in $SERVER_VLLM_LOG_RELPATH. Server is ready."
        server_ready=1
    else
        # Check for total timeout
        if [ $elapsed_log_wait -ge $MAX_LOG_WAIT_SEC ]; then
            echo "[$(date)] Error: Timeout ($MAX_LOG_WAIT_SEC sec) waiting for '{SERVER_READY_STRING}' in $SERVER_VLLM_LOG_RELPATH." >&2
             # Check logs upon timeout
             vllm_server_log="$SERVER_VLLM_LOG_RELPATH"
             main_pbs_server_log="$SERVER_PBS_LOG_RELPATH"
             echo "Server job $SERVER_JOB_ID_TO_CANCEL status:" >&2
             qstat "$SERVER_JOB_ID_TO_CANCEL" >&2 || echo "Server job $SERVER_JOB_ID_TO_CANCEL not found by qstat." >&2
             if [ -f "$vllm_server_log" ]; then echo "Last lines of vLLM log ($vllm_server_log):"; tail -n 30 "$vllm_server_log"; else echo "vLLM log ($vllm_server_log) not found (possibly deleted by server)."; fi >&2
             if [ -f "$main_pbs_server_log" ]; then echo "Last lines of main server PBS log ($main_pbs_server_log):"; tail -n 30 "$main_pbs_server_log"; else echo "Main server PBS log ($main_pbs_server_log) not found."; fi >&2
            exit 1
        fi

        # Log file might not exist yet, or string not found, or server job might be starting up
        if [ ! -f "$SERVER_VLLM_LOG_RELPATH" ]; then
            echo "[$(date)] Waiting for server output log '$SERVER_VLLM_LOG_RELPATH' to appear... ($elapsed_log_wait/$MAX_LOG_WAIT_SEC sec)"
        else
            echo "[$(date)] Checking server output log '$SERVER_VLLM_LOG_RELPATH' for ready string... ($elapsed_log_wait/$MAX_LOG_WAIT_SEC sec)"
            # Optional: Display last few lines of the log while waiting?
            # echo "--- Last 5 lines of $SERVER_VLLM_LOG_RELPATH ---"; tail -n 5 "$SERVER_VLLM_LOG_RELPATH"; echo "--- End ---"
        fi
        sleep $LOG_WAIT_INTERVAL
        elapsed_log_wait=$((elapsed_log_wait + LOG_WAIT_INTERVAL))
    fi
done

# --- Run Evaluation ---
echo "[$(date)] Server presumed ready. Changing to project root: {PROJECT_ROOT_REL_PATH}"
cd "{PROJECT_ROOT_REL_PATH}" || {{ echo "Error changing directory to project root"; exit 1; }}
echo "Current directory for python execution: $(pwd)"

echo "[$(date)] Running evaluation command..."
echo "Command: {eval_command}"
    {textwrap.indent(eval_command, '    ')} # Execute the command
EVAL_EXIT_CODE=$?

echo "[$(date)] Evaluation script exited with code: $EVAL_EXIT_CODE"

# --- Attempt Server Cancellation ONLY if Eval Succeeded ---
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Evaluation successful. Attempting to cancel server job: $SERVER_JOB_ID_TO_CANCEL"
    qdel $SERVER_JOB_ID_TO_CANCEL
    QDEL_EXIT_CODE=$?
    if [ $QDEL_EXIT_CODE -eq 0 ]; then
        echo "Successfully sent cancellation request for server job $SERVER_JOB_ID_TO_CANCEL."
    else
        # Check if it's already gone
        if ! qstat $SERVER_JOB_ID_TO_CANCEL > /dev/null 2>&1; then
             echo "qdel failed, but server job $SERVER_JOB_ID_TO_CANCEL seems to be already completed or deleted."
        else
             echo "Warning: qdel command for server job $SERVER_JOB_ID_TO_CANCEL exited with code $QDEL_EXIT_CODE. Job might require manual cancellation." >&2
        fi
    fi
else
    echo "[$(date)] Evaluation script failed (exit code $EVAL_EXIT_CODE). Server job $SERVER_JOB_ID_TO_CANCEL will NOT be cancelled automatically." >&2
    exit $EVAL_EXIT_CODE # Exit with the evaluation script's error code
fi

echo "--- PBS Client Job Finished ---"
# Exit with the evaluation script's exit code (0 if successful)
exit $EVAL_EXIT_CODE
"""
    return pbs_script_content


def write_pbs_script(job_name_prefix: str, pbs_script_content: str, suffix: str, workdir: str, logs_subdir: str) -> str:
    """Writes the PBS script content to a file in the specified logs subdirectory."""
    # workdir is the base script directory, logs_subdir is the relative subdir name
    log_path_base = os.path.join(workdir, logs_subdir)
    # Ensure the target log directory exists (main_yaml should also do this)
    os.makedirs(log_path_base, exist_ok=True)
    pbs_script_path = os.path.join(log_path_base, f"{job_name_prefix}_{suffix}.pbs")
    try:
        with open(pbs_script_path, "w") as f:
            f.write(pbs_script_content)
        print(f"PBS script written to: {pbs_script_path}")
        return pbs_script_path # Return the full path to the script
    except IOError as e:
        print(f"Error writing PBS script {pbs_script_path}: {e}")
        sys.exit(1)


def submit_pbs_job(pbs_script_path: str) -> str | None:
    """
    Submits the PBS job. Assumes pbs_script_path is the correct location
    of the script (e.g., inside the logs directory). qsub should be run
    from the parent directory of the logs dir (the script dir).
    """
    submit_command = ["qsub", pbs_script_path]
    print(f"Submitting command: {' '.join(submit_command)}")
    try:
        # Run qsub from the directory containing the script and the logs subdir.
        # $PBS_O_WORKDIR will be set to this directory.
        submission_dir = os.path.dirname(os.path.dirname(pbs_script_path)) # Go up one level from logs/script.pbs
        print(f"Running qsub from directory: {submission_dir}")
        process = subprocess.Popen(
            submit_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=submission_dir
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error submitting job {pbs_script_path}:")
            print(f"Stdout: {stdout.decode()}")
            print(f"Stderr: {stderr.decode()}")
            return None

        job_id = stdout.decode().strip()
        if not job_id:
             print(f"Error: qsub for {pbs_script_path} succeeded but returned no job ID.")
             print(f"Stderr: {stderr.decode()}")
             return None
        print(f"PBS Job submitted: {pbs_script_path} -> Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Exception during job submission for {pbs_script_path}: {e}")
        return None


def get_config_value(cfg, keys, default=None):
    """Safely get a nested value from a dictionary."""
    val = cfg
    try:
        for key in keys:
            val = val[key]
        # Handle cases where YAML might have 'null' explicitly
        if val is None and default is not None:
             return default
        return val
    except (KeyError, TypeError):
        return default


def validate_job_config(job_config, job_name_prefix, eval_type):
     """Checks for essential keys in the job config dictionary."""
     if not get_config_value(job_config, ['model_path']):
         print(f"Error: 'model_path' missing for job '{job_name_prefix}'. Skipping.")
         return False

     eval_cfg = job_config.get('eval', {})
     required_eval_args = ['n_start', 'model_name', 'model_identifier', 'dataset_name']
     if eval_type == 'similarity':
         required_eval_args.extend(['threshold', 'pruning_strategy', 'tokenizer_path'])
     # Add checks for sc_control if needed, e.g., model_identifier might be optional

     missing_args = [arg for arg in required_eval_args if arg not in eval_cfg]
     if missing_args:
         print(f"Error: Missing required eval arguments for job '{job_name_prefix}' (type: {eval_type}): {missing_args}. Skipping.")
         return False
     return True


def main_yaml():
    """Main function to orchestrate job submission from YAML."""
    # Determine the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running submission script from: {script_dir}")

    parser = argparse.ArgumentParser(
        description="Submit multiple two-stage PBS jobs defined in a YAML file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", "-c", default="experiments.yaml",
        help="Path to the YAML configuration file (relative to script location)."
    )
    parser.add_argument(
        "--jobid_file", default=DEFAULT_JOBID_FILE,
        help="File to track the last client job ID for chaining (relative to script location)."
             " Set to '' to disable chaining between runs."
    )
    parser.add_argument(
        "--start_job_index", type=int, default=0,
        help="0-based index of the job in the YAML file to start processing from."
    )
    parser.add_argument(
        "--max_jobs", type=int, default=None,
        help="Maximum number of jobs to process from the YAML file starting at start_job_index."
    )

    cli_args = parser.parse_args()

    # Construct absolute paths for config and jobid file relative to script dir
    config_path = os.path.join(script_dir, cli_args.config)
    jobid_file_path = os.path.join(script_dir, cli_args.jobid_file) if cli_args.jobid_file else None

    # --- Create Logs Directory ---
    logs_dir_abs_path = os.path.join(script_dir, LOGS_DIR_NAME)
    try:
        os.makedirs(logs_dir_abs_path, exist_ok=True)
        print(f"Ensured logs directory exists: {logs_dir_abs_path}")
    except OSError as e:
        print(f"Error creating logs directory {logs_dir_abs_path}: {e}")
        sys.exit(1)

    # --- Load YAML Config ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or 'jobs' not in config or not isinstance(config['jobs'], list):
            print(f"Error: YAML file '{config_path}' must contain a top-level 'jobs' list.")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: YAML file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
         print(f"An unexpected error occurred loading the YAML file: {e}")
         sys.exit(1)

    all_jobs = config.get('jobs', [])
    total_jobs_in_yaml = len(all_jobs)
    start_index = max(0, cli_args.start_job_index)
    end_index = total_jobs_in_yaml
    if cli_args.max_jobs is not None:
        end_index = min(start_index + cli_args.max_jobs, total_jobs_in_yaml)
    jobs_to_process = all_jobs[start_index:end_index]
    num_to_process = len(jobs_to_process)

    if num_to_process == 0:
        print("No jobs selected to process based on --start_job_index and --max_jobs.")
        sys.exit(0)

    print(f"Loaded {total_jobs_in_yaml} job definitions from {config_path}.")
    print(f"Processing {num_to_process} jobs (index {start_index} to {end_index - 1}).")

    # --- Read Previous Job ID for Chaining ---
    previous_client_jobid = None
    if jobid_file_path:
        previous_client_jobid = read_previous_jobid(jobid_file_path)
        if previous_client_jobid:
             print(f"Initial dependency for the first server job: {previous_client_jobid}")
        else:
             print(f"No previous job ID found in '{jobid_file_path}' or chaining disabled.")
    else:
        print("Job ID file not specified, chaining between runs is disabled.")

    # --- Loop Through Selected Jobs ---
    last_successful_client_id = None
    # workdir is where the script runs and where $PBS_O_WORKDIR will point
    workdir = script_dir

    for i, job_config in enumerate(jobs_to_process):
        current_job_index = start_index + i
        print(f"\n===== Processing Job {current_job_index + 1}/{total_jobs_in_yaml} =====")

        job_name_prefix = get_config_value(job_config, ['name_prefix'], f"yaml_job_{current_job_index+1}")
        eval_cfg = job_config.get('eval', {})
        eval_type = get_config_value(eval_cfg, ['type'])

        if not eval_type or eval_type not in ['similarity', 'sc_control']:
            print(f"Error: 'eval.type' missing or invalid ('{eval_type}') for job '{job_name_prefix}'. Must be 'similarity' or 'sc_control'. Skipping.")
            continue

        if not validate_job_config(job_config, job_name_prefix, eval_type):
             continue # Skip if essential config missing

        # Check for existing files in the target logs subdirectory
        if not check_existing_files(job_name_prefix, eval_type, workdir, LOGS_DIR_NAME):
             print(f"Skipping job '{job_name_prefix}' due to existing files.")
             continue

        # Extract parameters (using defaults where appropriate)
        model_path = get_config_value(job_config, ['model_path']) # Already validated
        server_cfg = job_config.get('server', {})
        tp_size = get_config_value(server_cfg, ['tensor_parallel_size'], 2)
        server_hours = get_config_value(server_cfg, ['hours'], 8)
        gpu_mem_util = get_config_value(server_cfg, ['gpu_memory_utilization']) # Allow None default
        enable_reasoning = get_config_value(server_cfg, ['enable_reasoning'], False)
        reasoning_parser = get_config_value(server_cfg, ['reasoning_parser'])
        no_vllm_use_v1_0 = get_config_value(server_cfg, ['no_vllm_use_v1_0'], False)
        vllm_use_v1 = not no_vllm_use_v1_0

        client_cfg = job_config.get('client', {})
        client_hours = get_config_value(client_cfg, ['hours'], 8)
        client_gpus = get_config_value(client_cfg, ['gpus'], 1)
        client_mem = get_config_value(client_cfg, ['mem'], "8gb")
        client_initial_delay_minutes = get_config_value(client_cfg, ['initial_delay_minutes'], 0) # Default 0
        client_initial_wait_seconds = int(client_initial_delay_minutes * 60)

        # --- Extract eval params needed for KVC path ---
        eval_output_dir = get_config_value(eval_cfg, ['output_dir'], os.path.join(os.path.expanduser("~"), "slimsc/prune/results"))
        eval_model_name = get_config_value(eval_cfg, ['model_name'])
        eval_dataset_name = get_config_value(eval_cfg, ['dataset_name'])
        eval_n_start = get_config_value(eval_cfg, ['n_start']) # Needed by both

        # --- Conditionally extract similarity params ---
        eval_pruning_strategy = None
        eval_threshold = None
        if eval_type == 'similarity':
            eval_pruning_strategy = get_config_value(eval_cfg, ['pruning_strategy'])
            eval_threshold = get_config_value(eval_cfg, ['threshold'])
            if None in [eval_pruning_strategy, eval_n_start, eval_threshold]:
                 print(f"Error: Missing similarity params (pruning_strategy, n_start, threshold) in eval config for job '{job_name_prefix}'. Skipping.")
                 continue
        elif eval_type == 'sc_control':
             if eval_n_start is None:
                 print(f"Error: Missing required 'n_start' in eval config for sc_control job '{job_name_prefix}'. Skipping.")
                 continue

        print(f"--- Job Details ({job_name_prefix}) ---")
        print(f"  Model: {model_path}")
        print(f"  Server: TP={tp_size}, Hours={server_hours}, Reasoning={enable_reasoning}")
        print(f"  Client: Type={eval_type}, Hours={client_hours}, GPUs={client_gpus if eval_type != 'sc_control' else 'N/A'}, Mem={client_mem}, Initial Delay={client_initial_delay_minutes}min")

        # --- Create and Submit Server ---
        # print(f"Dependency for this server: {previous_client_jobid if previous_client_jobid else 'None'}")
        # Pass logs_subdir name; returns relative paths including logs_subdir
        server_pbs_content, rel_server_ip_file, rel_pbs_server_log, rel_vllm_serve_log = create_server_pbs_script(
            job_name_prefix, model_path, tp_size, server_hours, gpu_mem_util,
            enable_reasoning, reasoning_parser, vllm_use_v1,
            dependency_job_id=None,
            logs_subdir=LOGS_DIR_NAME,
            eval_type=eval_type,
            base_output_dir=eval_output_dir,
            model_name=eval_model_name,
            dataset_name=eval_dataset_name,
            n_start=eval_n_start,
            pruning_strategy=eval_pruning_strategy,
            threshold=eval_threshold
        )
        # Write script into the logs directory
        server_pbs_path = write_pbs_script(job_name_prefix, server_pbs_content, "server", workdir, LOGS_DIR_NAME)
        # Submit the script (qsub runs from workdir)
        server_job_id = submit_pbs_job(server_pbs_path)

        if not server_job_id:
            print(f"Error submitting server job for '{job_name_prefix}'. Stopping further processing.")
            break # Or continue, depending on desired behavior

        # --- Create and Submit Client ---
        # Pass the relative vllm_serve_log file path (returned above) to the client script creator
        client_pbs_content = create_client_pbs_script(
            job_name_prefix, server_job_id,
            rel_server_ip_file, # Pass relative path
            rel_vllm_serve_log, # Pass relative path
            client_hours, client_gpus, client_mem,
            eval_type,
            eval_cfg,
            client_initial_wait_seconds,
            logs_subdir=LOGS_DIR_NAME # Pass the subdir name
        )
         # Write script into the logs directory
        client_pbs_path = write_pbs_script(job_name_prefix, client_pbs_content, "client", workdir, LOGS_DIR_NAME)
         # Submit the script (qsub runs from workdir)
        client_job_id = submit_pbs_job(client_pbs_path)

        if not client_job_id:
            print(f"Error submitting client job for '{job_name_prefix}'. Server job {server_job_id} may remain running.")
            print("Stopping further processing.")
            # Optional: Automatically cancel the server if client submission fails
            print(f"Attempting to cancel server job {server_job_id} due to client submission failure...")
            subprocess.run(["qdel", server_job_id], check=False, capture_output=True) # Use check=False, capture output
            break # Or continue

        # --- SUCCESS ---
        print(f"Successfully submitted job pair for '{job_name_prefix}': Server={server_job_id}, Client={client_job_id}")
        previous_client_jobid = client_job_id
        last_successful_client_id = client_job_id
        time.sleep(1)

    # --- After Loop ---
    if jobid_file_path and last_successful_client_id:
        write_jobid(jobid_file_path, last_successful_client_id)
        print(f"\nSaved final successful client ID ({last_successful_client_id}) to {jobid_file_path} for next run.")
    elif jobid_file_path:
         print(f"\nNo client jobs were successfully submitted in this run. Job ID file '{jobid_file_path}' was not updated.")

    print("\n===== YAML Job Submission Finished =====")


if __name__ == "__main__":
    main_yaml()