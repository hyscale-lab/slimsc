# slimsc/prune/jobs/generate_jobs_sif.py
import os
import sys
import argparse
import yaml
import shlex
from typing import Dict

# --- Configuration ---
PROJECT_ROOT_REL_PATH = "../../.."
LOGS_DIR_NAME = "logs"
PBS_PROJECT_PREFIX = "personal"
GPUS_PER_NODE = 4

# --- Helper Functions ---

def get_template(template_name: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "templates", template_name)
    try:
        with open(template_path, 'r') as f: return f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}"); sys.exit(1)

def check_existing_files(job_name_prefix: str, workdir: str):
    final_log_file = os.path.join(workdir, LOGS_DIR_NAME, f"{job_name_prefix}.log")
    if os.path.exists(final_log_file):
        print(f"Error: The final log file already exists: {final_log_file}")
        print(f"Please remove or rename it, or use a different name_prefix.")
        return False
    return True

def expandvars(path: str) -> str:
    """Expands environment variables in the given path."""
    if not path: return path
    expanded_path = os.path.expandvars(path)
    if not os.path.isabs(expanded_path):
        raise ValueError(f"Path '{path}' must be absolute after variable expansion.")
    return expanded_path

def create_pbs_script_from_template(job_config: Dict, job_name_prefix: str) -> str:
    """Creates a PBS script by populating a template with values from the job config."""
    # --- Extract Configs ---
    server_cfg = job_config.get('server', {})
    client_cfg = job_config.get('client', {})
    eval_cfg = job_config.get('eval', {})
    eval_type = eval_cfg['type']

    # --- Define File Paths early ---
    workdir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(workdir, LOGS_DIR_NAME)
    host_logs_dir = expandvars(get_config_value(eval_cfg, ['host_logs_dir'], os.path.join(os.path.expanduser("~"), "slimsc/logs")))
    pbs_log_file = os.path.join(host_logs_dir, f"{job_name_prefix}.log")
    vllm_serve_log_file = os.path.join(host_logs_dir, f"{job_name_prefix}_vllm_serve.log")
    server_ip_file = os.path.join(host_logs_dir, f"{job_name_prefix}_server_ip.txt")
    client_done_file = os.path.join(host_logs_dir, f"{job_name_prefix}_client.done")
    base_output_dir = get_config_value(eval_cfg, ['output_dir'], os.path.join(os.path.expanduser("~"), "slimsc/prune/results"))
    host_output_dir = expandvars(get_config_value(eval_cfg, ['host_output_dir'], os.path.join(os.path.expanduser("~"), "slimsc/prune/results")))
    hf_home = expandvars(get_config_value(eval_cfg, ['hf_home'], os.path.join(os.path.expanduser("~"), ".cache/huggingface")))

    # --- Primary Parameters ---
    model_path = job_config['model_path']
    tensor_parallel_size = get_config_value(server_cfg, ['tensor_parallel_size'], 2)
    client_gpus = get_config_value(client_cfg, ['gpus'], 1)
    
    # --- Determine Node and GPU layout ---
    total_gpus = tensor_parallel_size + client_gpus
    is_multi_node = eval_type == "similarity" and total_gpus > GPUS_PER_NODE
    
    # --- Build Commands ---
    # VLLM Server Singularity Command
    vllm_singularity_start_command_parts = [
        "singularity", "instance", "start", "--nv",
        "-B", f'{model_path}:{model_path}', # bind model path
        "-B", f'{host_logs_dir}:{logs_dir}', # bind logs subdir
    ]
    vllm_sif_path = os.path.abspath(expandvars(server_cfg.get('sif_path')))
    vllm_instance_name = f"vllm_{job_name_prefix}"
    vllm_start_instance_command = " ".join(vllm_singularity_start_command_parts) + f" {vllm_sif_path} {vllm_instance_name}"
    if not vllm_sif_path:
        raise ValueError(f"Missing 'sif_path' in server configuration for job '{job_name_prefix}'.")
    vllm_sif_path = os.path.abspath(vllm_sif_path)
    if not os.path.exists(vllm_sif_path):
        raise ValueError(f"Singularity image not found at {vllm_sif_path} for job '{job_name_prefix}'.")

    # vLLM Server Command (now uses the final path variable directly)
    vllm_singularity_exec_command_parts = [
        "setsid",
        "singularity", "exec", "--nv",
        f'instance://{vllm_instance_name}'
    ]
    vllm_parts = ["vllm", "serve", shlex.quote(model_path), f"--tensor-parallel-size {tensor_parallel_size}", "--port $PORT", "--seed 42"]
    if get_config_value(server_cfg, ['gpu_memory_utilization']):
        vllm_parts.append(f"--gpu-memory-utilization {server_cfg['gpu_memory_utilization']}")
    if get_config_value(server_cfg, ['enable_reasoning'], False):
        vllm_parts.append("--enable-reasoning")
    if get_config_value(server_cfg, ['reasoning_parser']):
        vllm_parts.append(f'--reasoning-parser {shlex.quote(server_cfg["reasoning_parser"])}')
    vllm_exec_command = " ".join(vllm_singularity_exec_command_parts) + " " + " ".join(vllm_parts) + f" > >(tee -a {shlex.quote(vllm_serve_log_file)}) 2>&1 &"

    # Client Evaluation Command
    client_singularity_start_command_parts = [
        "singularity", "instance", "start", "--nv", "--no-home",
        "-B", f'{hf_home}:{hf_home}', # bind hf_home
        "-B", f'{model_path}:{model_path}', # bind model path
        "-B", f'{host_output_dir}:{base_output_dir}', # bind kv cache usage output directory
        "-B", f'{host_logs_dir}:{logs_dir}', # bind logs subdir
    ]
    client_sif_path = os.path.abspath(expandvars(client_cfg.get('sif_path')))
    client_instance_name = f"client_{job_name_prefix}"
    client_start_instance_command = " ".join(client_singularity_start_command_parts) + f" {client_sif_path} {client_instance_name}"

    q_args = {k: shlex.quote(str(os.path.expandvars(v))) if isinstance(v, str) else v for k, v in eval_cfg.items()}
    client_singularity_exec_command_parts = [
        "singularity", "exec", "--nv", "--no-home",
        f'instance://{client_instance_name}',
    ]
    eval_parts = []
    if eval_type == "similarity":
        eval_parts = ["python -m slimsc.prune.evaluation.similarity_prune_eval", f"--n_start {q_args['n_start']}", f"--threshold {q_args['threshold']}", f"--pruning_strategy {q_args['pruning_strategy']}", f"--tokenizer_path {q_args['tokenizer_path']}", f"--threshold_schedule {q_args.get('threshold_schedule', 'fixed')}"]
        if q_args.get('seed') is not None: eval_parts.append(f"--seed {q_args['seed']}")
        if q_args.get('num_steps_to_delay_pruning') is not None: eval_parts.append(f"--num_steps_to_delay_pruning {q_args['num_steps_to_delay_pruning']}")
    elif eval_type == "sc_control":
        eval_parts = ["python -m slimsc.prune.evaluation.sc_control_eval", f"--n_start {q_args['n_start']}"]
        if q_args.get('tokenizer_path'): eval_parts.append(f"--tokenizer_path {q_args['tokenizer_path']}")

    eval_parts.extend([f"--model_name {q_args['model_name']}", f"--model_identifier {q_args['model_identifier']}", "--vllm_url $VLLM_URL", f"--dataset_name {q_args['dataset_name']}"])
    if q_args.get("output_dir"): eval_parts.append(f"--output_dir {q_args['output_dir']}")
    if q_args.get("num_qns"): eval_parts.append(f"--num_qns {q_args['num_qns']}")
    eval_command = " ".join(client_singularity_exec_command_parts) + " " + " ".join(filter(None, eval_parts))

    # --- Prepare All Template Variables in a Single Dictionary ---
    pruning_strategy = get_config_value(eval_cfg, ['pruning_strategy'])
    threshold = get_config_value(eval_cfg, ['threshold'])
    threshold_schedule = get_config_value(eval_cfg, ['threshold_schedule'])
    
    run_name = ""
    if eval_type == "similarity":
        schedule_suffix = f"_{threshold_schedule}" if threshold_schedule == 'annealing' else ""
        threshold_for_naming = 0.9 if threshold_schedule == 'annealing' else threshold
        run_name = f"{pruning_strategy}{schedule_suffix}_n{eval_cfg['n_start']}_thresh{threshold_for_naming:.2f}_delay{get_config_value(eval_cfg, ['num_steps_to_delay_pruning'], 20)}"
    elif eval_type == "sc_control":
        run_name = f"sc_{eval_cfg['n_start']}_control"
    model_dataset_dir = os.path.join(base_output_dir, eval_cfg['model_name'], eval_cfg['dataset_name'], run_name or "unknown_run")
    host_dataset_dir = os.path.join(host_output_dir, eval_cfg['model_name'], eval_cfg['dataset_name'], run_name or "unknown_run")

    template_vars = {
        "JOB_NAME": job_name_prefix,
        "SERVER_HOURS": get_config_value(server_cfg, ['hours'], 8),
        "PBS_PROJECT": f"{PBS_PROJECT_PREFIX}-{os.environ.get('USER', 'default')}",
        "PROJECT_ROOT_REL_PATH": PROJECT_ROOT_REL_PATH,
        "VLLM_START_INSTANCE_COMMAND": vllm_start_instance_command,
        "VLLM_EXEC_COMMAND": vllm_exec_command,
        "VLLM_INSTANCE_NAME": vllm_instance_name,
        "EVAL_COMMAND": eval_command,
        "CLIENT_START_INSTANCE_COMMAND": client_start_instance_command,
        "CLIENT_INSTANCE_NAME": client_instance_name,
        "EVAL_TYPE": eval_type,
        "SERVER_GPU_INDICES": ",".join(map(str, range(tensor_parallel_size))),
        "IS_MULTI_NODE": "true" if is_multi_node else "false",
        "TARGET_KVC_FILE_PATH": shlex.quote(os.path.join(model_dataset_dir, "kvcache_usages.csv")),
        "HOST_KVC_FILE_PATH": shlex.quote(os.path.join(host_dataset_dir, "kvcache_usages.csv")),
        "HF_HOME": shlex.quote(hf_home),
        # Add path variables directly
        "PBS_LOG_FILE": shlex.quote(pbs_log_file),
        "VLLM_SERVE_LOG_FILE": shlex.quote(vllm_serve_log_file),
        "SERVER_IP_FILE": shlex.quote(server_ip_file),
        "CLIENT_DONE_FILE": shlex.quote(client_done_file),
    }

    # Node-dependent variables
    if is_multi_node:
        if tensor_parallel_size > GPUS_PER_NODE or client_gpus > GPUS_PER_NODE:
            raise ValueError(f"Job '{job_name_prefix}' requests more GPUs for server ({tensor_parallel_size}) or client ({client_gpus}) than available on a single node ({GPUS_PER_NODE}).")
        template_vars["GPU_REQUEST_LINE"] = f"select=1:ngpus={tensor_parallel_size}+1:ngpus={client_gpus}"
        template_vars["CLIENT_GPU_INDICES_MULTI_NODE"] = ",".join(map(str, range(client_gpus)))
        template_vars["CLIENT_GPU_INDICES_SINGLE_NODE"] = "" # Dummy value
    else: # Single-node
        if eval_type == "similarity":
            template_vars["GPU_REQUEST_LINE"] = f"select=1:ngpus={total_gpus}"
            template_vars["CLIENT_GPU_INDICES_SINGLE_NODE"] = ",".join(map(str, range(tensor_parallel_size, total_gpus)))
        else: # sc_control
            template_vars["GPU_REQUEST_LINE"] = f"select=1:ngpus={tensor_parallel_size}"
            template_vars["CLIENT_GPU_INDICES_SINGLE_NODE"] = "" # Not used
        template_vars["CLIENT_GPU_INDICES_MULTI_NODE"] = "" # Dummy value
    
    # --- Render Final Script in a Single Pass ---
    template = get_template("combined_job_sif.pbs.template")
    return template.format(**template_vars)

def write_pbs_script(job_name_prefix: str, pbs_script_content: str, workdir: str) -> str:
    log_path_base = os.path.join(workdir, LOGS_DIR_NAME)
    os.makedirs(log_path_base, exist_ok=True)
    pbs_script_path = os.path.join(log_path_base, f"{job_name_prefix}.pbs")
    try:
        with open(pbs_script_path, "w") as f: f.write(pbs_script_content)
        print(f"PBS script written to: {pbs_script_path}")
        return pbs_script_path
    except IOError as e:
        print(f"Error writing PBS script {pbs_script_path}: {e}"); sys.exit(1)

def get_config_value(cfg, keys, default=None):
    val = cfg
    try:
        for key in keys: val = val[key]
        return val if val is not None else default
    except (KeyError, TypeError): return default

def validate_job_config(job_config, job_name_prefix, eval_type):
    if not get_config_value(job_config, ['model_path']):
        print(f"Error: 'model_path' missing for job '{job_name_prefix}'. Skipping."); return False
    eval_cfg = job_config.get('eval', {})
    required = ['n_start', 'model_name', 'model_identifier', 'dataset_name']
    if eval_type == 'similarity': required.extend(['threshold', 'pruning_strategy', 'tokenizer_path'])
    missing = [arg for arg in required if arg not in eval_cfg]
    if missing:
        print(f"Error: Missing eval args for '{job_name_prefix}': {missing}. Skipping."); return False
    return True

def output_pbs_script_path(job_uuid: str, host_log_dir: str, pbs_script_path: str):
    log_path_base = os.path.dirname(pbs_script_path)
    print(f"PBS script created at: {pbs_script_path} (host dir: {host_log_dir})")
    print(f"Logs will be saved in: {log_path_base} (host dir: {host_log_dir})")
    print(f"To submit the job, run: qsub {os.path.basename(pbs_script_path)}")
    # write into a file for access from nscc
    output_file = os.path.join(log_path_base, f"{job_uuid}_pbs_scripts.txt")
    try:
        with open(output_file, "a") as f:
            f.write(os.path.join(host_log_dir, os.path.basename(pbs_script_path)) + "\n")
        print(f"PBS script path saved to: {output_file}")
    except IOError as e:
        print(f"Error writing PBS script path to {output_file}: {e}")
    return pbs_script_path

def main_yaml():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Submit optimized PBS tasks from a YAML file using a template.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", default="experiments.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--start_job_index", type=int, default=0, help="0-based index of the job to start from.")
    parser.add_argument("--max_jobs", type=int, default=None, help="Max number of jobs to process.")
    parser.add_argument("--job_uuid", type=str, default=None, help="UUID of the job to process.")
    cli_args = parser.parse_args()

    config_path = os.path.join(script_dir, cli_args.config)
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
        if 'jobs' not in config: raise ValueError("YAML must contain a top-level 'jobs' list.")
    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        print(f"Error loading YAML file {config_path}: {e}"); sys.exit(1)

    all_jobs = config.get('jobs', [])
    start_index = max(0, cli_args.start_job_index)
    end_index = min(start_index + (cli_args.max_jobs or len(all_jobs)), len(all_jobs))
    job_uuid = cli_args.job_uuid
    jobs_to_process = all_jobs[start_index:end_index]
    if not jobs_to_process:
        print("No jobs selected to process."); sys.exit(0)
    print(f"Processing {len(jobs_to_process)} jobs (index {start_index} to {end_index - 1}).")

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
        if not check_existing_files(job_name_prefix, workdir):
             print(f"Skipping job '{job_name_prefix}' due to existing files."); continue

        try:
            pbs_script_content = create_pbs_script_from_template(job_config, job_name_prefix)
        except ValueError as e:
            print(f"Error creating script for '{job_name_prefix}': {e}. Skipping job."); continue

        host_log_dir = expandvars(get_config_value(eval_cfg, ['host_logs_dir'], os.path.join(os.path.expanduser("~"), "slimsc/logs")))
        pbs_script_path = write_pbs_script(job_name_prefix, pbs_script_content, workdir)
        output_pbs_script_path(job_uuid, host_log_dir, pbs_script_path)

    print("\n===== YAML Job PBS Script Creation Finished =====")

if __name__ == "__main__":
    main_yaml()