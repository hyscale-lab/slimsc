# slimsc

## Setup

### 1. Create and Activate Conda Environment

First, set up a Conda environment with the required Python dependencies.

```bash
# Create a new conda environment (e.g., named "slimsc")
conda create -n slimsc python=3.10 -y

# Activate the environment
conda activate slimsc

# Install the dependencies
pip install -r requirements/requirements.txt
pip install -r requirements/torch.requirements.txt
```

### 2. Modified vLLM Installation

The `slimsc` evaluation process relies on the vLLM server logging KV cache usage at each step. This requires a modification to the vLLM source code. You must install vLLM from source in an editable mode and apply the necessary code change.

*   **Install vLLM in Editable Mode:** Follow the vLLM documentation to install it from source using `pip install -e .` (from the root of the cloned vLLM repository). This allows changes to the source files to take effect without re-installing.
*   **Locate and Edit `llm_engine.py`:** Find the file `vllm/vllm/engine/llm_engine.py` within your vLLM source directory.
*   **Add Imports:** At the top of the file, add the following import statements:
    ```python
    import csv
    import os
    import time
    import logging # Needed for logger.error
    ```
*   **Initialize Logger:** Below the imports (or near other logger initialization), initialize the logger if it's not already done:
    ```python
    logger = logging.getLogger(__name__)
    ```
*   **Insert KV Logging Code:** Locate the `step()` function within `llm_engine.py`. **Insert the following code block right before the `return ctx.request_outputs` line at the very end of the function:**

    ```python
    # --- START slimsc KV Usage Logging ---
    # KV Cache Usage in %
    num_total_gpu = self.cache_config.num_gpu_blocks
    gpu_cache_usage_perc = 0.
    if num_total_gpu:
        num_free_gpu = sum(
            scheduler.block_manager.get_num_free_gpu_blocks()
            for scheduler in self.scheduler)
        gpu_cache_usage_perc = 1.0 - (num_free_gpu / num_total_gpu)
    if gpu_cache_usage_perc > 0.0:
        try:
            # Get the CSV file path from an environment variable, with a default fallback
            # NOTE: Default fallback path below might not be appropriate for your system.
            # It's strongly recommended to always set the KVC_USAGE_FILE environment variable.
            default_path = "~/scratch/kvcache_usage.csv"
            path = os.getenv("KVC_USAGE_FILE", default_path)
            path = os.path.expanduser(path)  # Expand '~' for user directories

            file_exists = os.path.exists(path)
            # Use 'a' mode for appending, 'w' only if file doesn't exist (or handle header writing explicitly)
            # Ensure directory exists before writing
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Write header only if the file did not exist before opening or is empty
                if not file_exists or os.path.getsize(path) == 0:
                    writer.writerow(["timestamp", "gpu_cache_usage_perc"])
                writer.writerow([time.time(), gpu_cache_usage_perc])
        except Exception as e:
            # Use the logger initialized earlier
            logger.error(f"Failed to write KV cache usage to {path}: {e}")
    # --- END slimsc KV Usage Logging ---
    ```
*   **Save the file.** Because you installed in editable mode, these changes should be reflected when you run vLLM again.

## Running Experiments

There are two ways to run experiments:

1.  **Manual / Local Execution:** Recommended for public reproducibility and running on a local machine.
2.  **HPC / Cluster Execution:** For users with a PBS/TORQUE-based HPC cluster, using the provided job submission script.

---

## Manual / Local Execution

This method involves manually starting the vLLM server and the evaluation client in separate terminals. It gives you full control over the process.

### 1. Configure Your Experiment

Copy the example configuration file `prune/jobs/experiments.yaml` to a new file (e.g., `my_experiment.yaml`).

```bash
cp prune/jobs/experiments.yaml my_experiment.yaml
```

Open `my_experiment.yaml` and edit the parameters for your experiment. Pay close attention to:
- `model_path`: The path to your model weights.
- `eval`: All the evaluation parameters like `type`, `n_start`, `threshold`, etc.
- `server`: Server configuration like `tensor_parallel_size`.

### 2. Start the vLLM Server

The vLLM server must be running before you can launch the client.

**A. Set Environment Variables:**
You need to set `CUDA_VISIBLE_DEVICES` to control which GPUs are used, and `KVC_USAGE_FILE` to tell the server where to save the KV cache statistics. The path for `KVC_USAGE_FILE` depends on your experiment configuration. You can derive it from your YAML file (see `prune/jobs/submit_jobs.py` for the exact logic) or simply choose a path.

```bash
# Activate your conda environment
conda activate slimsc

# Select the GPUs for the server (matches tensor_parallel_size)
export CUDA_VISIBLE_DEVICES=0,1 # For TP=2

# Set the path for the KV cache log file
export KVC_USAGE_FILE=/path/to/your/results/kvcache_usages.csv

# Create the directory for the log file
mkdir -p "$(dirname "$KVC_USAGE_FILE")"
```

**B. Launch the Server:**
Construct the `vllm serve` command using the `model_path` and `server` settings from your YAML file.

```bash
vllm serve /path/to/your/model \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
```
Wait until you see the message `Application startup complete.` before proceeding.

### 3. Run the Evaluation Client

Open a **new terminal**.

**A. Set Environment Variables:**

```bash
# Activate your conda environment
conda activate slimsc

# Set the URL for the vLLM server
export VLLM_URL="http://localhost:8000"
```

**B. Run the Client Script:**
Navigate to the root of the `slimsc` project. The command you run depends on the `eval.type` in your YAML file.

```bash
cd /path/to/your/slimsc/project/root
```

**For `similarity` evaluation:**

The `similarity` evaluation client requires one GPU to load the sentence transformer model. Set `CUDA_VISIBLE_DEVICES` to an available GPU that is not being used by the vLLM server.

```bash
# Allocate one GPU for the client (e.g., GPU 2)
export CUDA_VISIBLE_DEVICES=2

python -m prune.evaluation.similarity_prune_eval \
    --n_start <eval.n_start> \
    --threshold <eval.threshold> \
    --pruning_strategy <eval.pruning_strategy> \
    --model_name <eval.model_name> \
    --model_identifier <eval.model_identifier> \
    --tokenizer_path <eval.tokenizer_path> \
    --dataset_name <eval.dataset_name> \
    # ... and other arguments from your YAML file
```

**For `sc_control` evaluation:**
```bash
python -m prune.evaluation.sc_control_eval \
    --n_start <eval.n_start> \
    --model_name <eval.model_name> \
    --model_identifier <eval.model_identifier> \
    --dataset_name <eval.dataset_name> \
    # ... and other arguments from your YAML file
```

Replace the placeholders `<...>` with the actual values from your YAML configuration.

### 4. Cleanup

Once the client script is finished, stop the vLLM server by pressing `Ctrl+C` in its terminal.

---

## HPC / Cluster Execution (PBS/TORQUE)

For users on a compatible HPC cluster, you can use the provided script to submit jobs from a YAML file.

### 1. Configure Your Experiment

Edit `prune/jobs/experiments.yaml` or create a new YAML file with your job configurations. Ensure the paths (like `model_path`) are correct for the cluster's filesystem.

### 2. Submit the Job

The `submit_jobs.py` script will read your YAML file, generate the necessary PBS scripts, and submit them to the queue.

```bash
# Activate your conda environment
conda activate slimsc

# Run the submission script
python prune/jobs/submit_jobs.py --config /path/to/your/experiments.yaml
```

The script will create log files and PBS scripts in the `prune/jobs/logs` directory. You can monitor the status of your jobs using `qstat`.