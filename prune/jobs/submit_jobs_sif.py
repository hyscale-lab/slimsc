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
from string import Template

# --- Configuration ---
# These paths are relative to where the PBS job starts ($PBS_O_WORKDIR),
# which will be the directory where this script is run (e.g., slimsc/prune/jobs/).
# Path to the project root directory (containing the 'slimsc' package)
PROJECT_ROOT_REL_PATH = "../../.."
# Default location for the job ID chaining file, relative to this script's location
DEFAULT_JOBID_FILE = ".last_client_jobid"
# Subdirectory name for logs, scripts, etc., relative to this script's location
LOGS_DIR_NAME = "logs"

# PBS Project configuration (adjust if necessary)
PBS_PROJECT_PREFIX = "personal"


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
    sif_image_path: str,
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
    threshold_schedule: Optional[str],
    num_steps_to_delay_pruning_for_naming: Optional[int],
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
        schedule_suffix = ""
        threshold_for_naming = threshold

        if threshold_schedule == 'annealing':
            schedule_suffix = f"_{threshold_schedule}" # Adds "_annealing"
            threshold_for_naming = 0.9 # Use fixed 0.9 for naming convention

        # Ensure required values are present before formatting
        if pruning_strategy is not None and n_start is not None and threshold_for_naming is not None and num_steps_to_delay_pruning_for_naming is not None:
            run_name = f"{pruning_strategy}{schedule_suffix}_n{n_start}_thresh{threshold_for_naming:.2f}_delay{num_steps_to_delay_pruning_for_naming}"
            print(f"Constructed server run_name: {run_name}") # Add log for debugging
        else:
            print(f"Warning: Could not construct run_name for server KVC path due to missing params (strategy={pruning_strategy}, n_start={n_start}, threshold_for_naming={threshold_for_naming}), delay_for_naming={num_steps_to_delay_pruning_for_naming}). Using 'unknown_run'.")
            run_name = "unknown_run"

    elif eval_type == "sc_control":
        # Assuming sc_control run_name is based on n_start
        if n_start is not None:
            run_name = f"sc_{n_start}_control"
            print(f"Constructed server run_name: {run_name}") # Add log for debugging
        else:
            print("Warning: Missing n_start for sc_control run_name.")
            run_name = "unknown_run"

    # Fallback if run_name wasn't generated for some reason
    if not run_name:
        print("Error: run_name could not be determined. Defaulting to 'unknown_run'.")
        run_name = "unknown_run"

    model_dataset_dir = os.path.join(base_output_dir, model_name, dataset_name, run_name if run_name else "unknown_run")
    results_zip_path = os.path.join("~/slimsc-results", model_name, dataset_name, run_name if run_name else "unknown_run")
    target_kvc_file_path = os.path.join(model_dataset_dir, "kvcache_usages.csv")
    quoted_target_kvc_file_path = shlex.quote(target_kvc_file_path)

    # Quote paths that might contain spaces or special characters for shell safety
    quoted_model_path = shlex.quote(os.path.expandvars(model_path))
    quoted_reasoning_parser = shlex.quote(os.path.expandvars(reasoning_parser)) if reasoning_parser else None
    quoted_vllm_serve_log = shlex.quote(os.path.expandvars(relative_vllm_serve_log_file))
    quoted_sif_image_path = shlex.quote(os.path.expandvars(sif_image_path))

    user = os.environ.get("USER", "default_user")
    pbs_project = f"{PBS_PROJECT_PREFIX}-{user}"

    dependency_directive = ""
    if dependency_job_id:
        dependency_directive = f"#PBS -W depend=afterok:{dependency_job_id}"

    # Construct vLLM Command
    singularity_command_parts = [
        "singularity", "exec", "--nv",
        "-B", f'{base_output_dir}:{base_output_dir}', # bind kv cache usage output directory
        "-B", f'{logs_subdir}:{logs_subdir}', # bind logs subdir
    ]
    vllm_command_parts = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", f'{quoted_model_path}',
        f"--tensor-parallel-size {tensor_parallel_size}",
        "--port $PORT", # Use dynamic PORT variable instead of hardcoded 8000
        "--seed 42"
    ]
    # Redirection using tee to capture stdout/stderr to the specific log file,
    # and run in the background (&)
    # Ensure the log directory exists before redirecting (although PBS should handle the main log dir)
    vllm_log_redirection = f"> >(tee -a {quoted_vllm_serve_log}) 2>&1 &"

    if gpu_memory_utilization is not None and 0 < gpu_memory_utilization <= 1:
         vllm_command_parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")
    if enable_reasoning:
        vllm_command_parts.append("--enable-reasoning")
    if quoted_reasoning_parser:
        vllm_command_parts.append(f'--reasoning-parser {quoted_reasoning_parser}')

    # The base command without redirection
    singularity_command = " ".join(singularity_command_parts)
    vllm_serve_command = " ".join(vllm_command_parts)

    exports = [
        f"export CUDA_VISIBLE_DEVICES=$(seq -s , 0 {tensor_parallel_size - 1})",
        f"export KVC_USAGE_FILE={quoted_target_kvc_file_path}",
        "export VLLM_USE_V1=0",
    ]

    kvc_dirname = os.path.dirname(target_kvc_file_path)

    create_run_dir_command = f'mkdir -p {shlex.quote(kvc_dirname)}'

    modules = ["singularity", "cuda/12.2.2"]

    # Create the dictionary for template substitution
    template_vars = {
        "tensor_parallel_size": tensor_parallel_size,
        "server_hours": server_hours,
        "pbs_project": pbs_project,
        "server_job_name": server_job_name,
        "relative_pbs_log_file": relative_pbs_log_file,
        "dependency_directive": dependency_directive,

        "relative_vllm_serve_log_file": relative_vllm_serve_log_file,
        "relative_server_ip_file": relative_server_ip_file,
        "relative_pbs_log_file": relative_pbs_log_file,

        "dependency_job_id": dependency_job_id if dependency_job_id else None,
        "eval_type": eval_type,
        "run_name": run_name,
        "target_kvc_file_path": target_kvc_file_path,

        "logs_subdir": logs_subdir,
        "kvc_dirname": kvc_dirname, # Directory for KVC usage output
        "create_run_dir_command": create_run_dir_command, # Command to create the KVC usage directory
        "exports": "\n".join(exports), # Join export commands with newlines
        "modules": " ".join(modules),

        "singularity_command": singularity_command,
        "sif_image_path": quoted_sif_image_path,
        "vllm_serve_command": vllm_serve_command,
        "vllm_log_redirection": vllm_log_redirection,
    }

    # --- PBS Script Content ---
    # NOTE: All paths referenced inside the script now need to be relative to $PBS_O_WORKDIR
    # and include the logs_subdir prefix where appropriate.
    with open('jobs/templates/server_sif.pbs', 'r') as f:
        src = Template(f.read())
        pbs_script_content = src.substitute(template_vars)
    # Return the path to the specific log file vLLM writes to, relative to $PBS_O_WORKDIR
    return pbs_script_content, relative_server_ip_file, relative_pbs_log_file, relative_vllm_serve_log_file


def create_client_pbs_script(
    sif_image_path: str,
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
    # Get the vllm server hostname
    server_hostname = get_job_hostname(server_job_id)

    user = os.environ.get("USER", "default_user")
    pbs_project = f"{PBS_PROJECT_PREFIX}-{user}"
    quoted_sif_image_path = shlex.quote(os.path.expandvars(sif_image_path))

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
        elif v is None: # if seed or num_steps_to_delay_pruning is None from YAML
            quoted_eval_args[k] = None # Keep as None, to be skipped later
        else:
            quoted_eval_args[k] = v # Keep numbers, etc., as is

    singularity_command_parts = [
        "singularity", "exec", "--nv", 
        "-B", f'{logs_subdir}:{logs_subdir}', # Bind logs subdir
        "-B", f'{os.path.abspath(PROJECT_ROOT_REL_PATH)}:{os.path.abspath(PROJECT_ROOT_REL_PATH)}', # Bind project root
    ]
    eval_command_parts = []
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
            f"--threshold_schedule {quoted_eval_args['threshold_schedule']}" if quoted_eval_args.get('threshold_schedule') else "",
        ]
        if quoted_eval_args.get('seed') is not None:
            eval_command_parts.append(f"--seed {quoted_eval_args['seed']}")
        if quoted_eval_args.get('num_steps_to_delay_pruning') is not None:
            eval_command_parts.append(f"--num_steps_to_delay_pruning {quoted_eval_args['num_steps_to_delay_pruning']}")

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

    eval_command = " ".join(singularity_command_parts) + quoted_sif_image_path + " ".join(eval_command_parts)
    template_vars = {
        "gpu_request_line": gpu_request_line,
        "client_hours": client_hours,
        "pbs_project": pbs_project,
        "client_job_name": client_job_name,
        "relative_client_log_file": relative_client_log_file,
        "server_job_id": server_job_id,

        "server_ip_file_to_check": server_ip_file_to_check,
        "server_log_to_check": server_log_to_check,
        "relative_main_server_pbs_log": relative_main_server_pbs_log,
        "relative_client_log_file": relative_client_log_file,

        "server_hostname": server_hostname,
        "eval_type": eval_type,
        "logs_subdir": logs_subdir,
        "project_root_rel_path": PROJECT_ROOT_REL_PATH, # Relative path to project root

        "cuda_export": cuda_export,
        "initial_wait_seconds": initial_wait_seconds, # Initial wait in seconds

        "eval_command": textwrap.indent(eval_command, '    '),
    }

    # --- Start PBS Script Content ---
    # Note: All paths referenced are relative to $PBS_O_WORKDIR
    with open('jobs/templates/client_sif.pbs', 'r') as f:
        src = Template(f.read())
        pbs_script_content = src.substitute(template_vars)
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

def get_job_hostname(job_id: str) -> str:
    """Extracts the server hostname from the qstat output for a given job ID."""
    get_hostname_command = ["qstat", "-f", job_id, "|", "grep", "exec_host"]
    try:
        process = subprocess.Popen(
            get_hostname_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error getting hostname for job {job_id}: {stderr.decode()}")
            return None

        # Extract the hostname from the output
        output = stdout.decode().strip()
        if not output:
            print(f"No exec_host found for job {job_id}.")
            return None

        # Example output: "exec_host = x1000c0s6b0n1/1*16"
        hostname = output.split('=')[1].strip().split('/')[0]
        return hostname
    except Exception as e:
        print(f"Exception during hostname retrieval for {job_id}: {e}")
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


# TODO: Add vllm sif image to required args
def validate_job_config(job_config, job_name_prefix, eval_type):
     """Checks for essential keys in the job config dictionary."""
     if not get_config_value(job_config, ['model_path']):
         print(f"Error: 'model_path' missing for job '{job_name_prefix}'. Skipping.")
         return False

     eval_cfg = job_config.get('eval', {})
     required_eval_args = ['n_start', 'model_name', 'model_identifier', 'dataset_name']
     if eval_type == 'similarity':
         required_eval_args.extend(['threshold', 'pruning_strategy', 'tokenizer_path'])

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
        server_sif_image_path = get_config_value(server_cfg, ['sif_image_path'])
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
        client_sif_image_path = get_config_value(client_cfg, ['sif_image_path'])

        # --- Extract eval params needed for KVC path ---
        eval_output_dir = get_config_value(eval_cfg, ['output_dir'], os.path.join(os.path.expanduser("~"), "slimsc/prune/results"))
        eval_model_name = get_config_value(eval_cfg, ['model_name'])
        eval_dataset_name = get_config_value(eval_cfg, ['dataset_name'])
        eval_n_start = get_config_value(eval_cfg, ['n_start']) # Needed by both

        eval_seed = None
        eval_num_steps_to_delay_pruning = None
        eval_pruning_strategy = None
        eval_threshold = None
        eval_threshold_schedule = None

        # --- Conditionally extract similarity params ---
        eval_threshold_schedule = None # Initialize
        if eval_type == 'similarity':
            eval_pruning_strategy = get_config_value(eval_cfg, ['pruning_strategy'])
            eval_threshold = get_config_value(eval_cfg, ['threshold'])
            eval_threshold_schedule = get_config_value(eval_cfg, ['threshold_schedule'], 'fixed')
            eval_seed = get_config_value(eval_cfg, ['seed'], None) # Default to None if not in YAML
            DEFAULT_DELAY_FOR_NAMING = 20 # Should match similarity_prune_eval.py argparse default
            eval_num_steps_to_delay_pruning_for_naming = get_config_value(eval_cfg, ['num_steps_to_delay_pruning'], DEFAULT_DELAY_FOR_NAMING)
            if None in [eval_pruning_strategy, eval_n_start, eval_threshold]:
                 print(f"Error: Missing similarity params (pruning_strategy, n_start, threshold) in eval config for job '{job_name_prefix}'. Skipping.")
                 continue
        elif eval_type == 'sc_control':
            if eval_n_start is None:
                print(f"Error: Missing required 'n_start' in eval config for sc_control job '{job_name_prefix}'. Skipping.")
                continue

            eval_threshold = None
            eval_threshold_schedule = None
            eval_num_steps_to_delay_pruning_for_naming = None

        print(f"--- Job Details ({job_name_prefix}) ---")
        print(f"  Model: {model_path}")
        print(f"  Server: TP={tp_size}, Hours={server_hours}, Reasoning={enable_reasoning}")
        print(f"  Client: Type={eval_type}, Hours={client_hours}, GPUs={client_gpus if eval_type != 'sc_control' else 'N/A'}, Mem={client_mem}, Initial Delay={client_initial_delay_minutes}min")
        if eval_type == 'similarity':
            print(f"    Seed: {eval_seed if eval_seed is not None else 'Default'}")
            print(f"    Num Steps to Delay Pruning: {eval_num_steps_to_delay_pruning if eval_num_steps_to_delay_pruning is not None else 'Default (20)'}")
        # --- Create and Submit Server ---
        # print(f"Dependency for this server: {previous_client_jobid if previous_client_jobid else 'None'}")
        # Pass logs_subdir name; returns relative paths including logs_subdir
        server_pbs_content, rel_server_ip_file, rel_pbs_server_log, rel_vllm_serve_log = create_server_pbs_script(
            server_sif_image_path,
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
            threshold=eval_threshold,
            threshold_schedule=eval_threshold_schedule,
            num_steps_to_delay_pruning_for_naming=eval_num_steps_to_delay_pruning_for_naming if eval_type == 'similarity' else None
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
            client_sif_image_path,
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