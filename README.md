# slimsc

## Prerequisites

1.  **slimsc Code:** You have cloned or have access to the `slimsc` project code.
2.  **Conda:** You have Conda installed.
3.  **Conda Environment:** You have a Conda environment (`vllm` by default, as configured in `submit_jobs.py`) containing vLLM and the necessary slimsc Python dependencies. You should know the path to your `conda.sh` initialization script (e.g., `$HOME/miniconda3/etc/profile.d/conda.sh`).
4.  **Model Weights:** The language model weights are available on your system.
5.  **GPUs:** You have sufficient GPU resources available on the machine where you will run vLLM (`tensor_parallel_size` GPUs). The client might also need GPUs depending on the evaluation type (`client.gpus` in YAML, though `sc_control` doesn't use them).
6.  **Target Directories:** You should decide on a base directory for experiment results (`base_output_dir`, matching `eval.output_dir` in the YAML logic).
7.  **Modified vLLM Installation:** The slimsc evaluation process relies on the vLLM server logging KV cache usage at each step. This requires a modification to the vLLM source code. You must install vLLM from source in an editable mode and apply the necessary code change.

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

## The Two-Step Process

An experiment job pair consists of two main steps that must run sequentially:

1.  **Start the vLLM Server:** This process loads the model onto GPUs and waits for API requests. It must be running and ready before step 2 begins. This process also writes the KVC usage data to a specific file.
2.  **Run the Evaluation Client:** This script connects to the running vLLM server and performs the evaluation task (e.g., inference with pruning, or control runs).

These steps will typically run in separate terminals or background processes.

## Step 1: Starting the vLLM Server

The vLLM server needs to be started with specific configuration matching the desired experiment. The path where the server logs KVC usage is determined by the experiment parameters, so this environment variable (`KVC_USAGE_FILE`) must be set correctly *before* starting the `vllm serve` command.

**1. Set up the environment:**

Open a new terminal.
Source your Conda initialization script and activate the environment:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm
```

(Adjust the path `$HOME/miniconda3` if necessary).

Set the `CUDA_VISIBLE_DEVICES` environment variable to control which GPUs vLLM uses. This should correspond to the `tensor_parallel_size` configured in the YAML. For `TP=2`, use:

```bash
export CUDA_VISIBLE_DEVICES=0,1 # Use GPUs 0 and 1
```

Adjust `0,1` based on the required `tensor_parallel_size` and available GPUs.

**2. Determine the `KVC_USAGE_FILE` path:**

The path for the KVC usage CSV file is constructed based on `eval.output_dir`, `eval.model_name`, `eval.dataset_name`, and a `run_name` derived from the experiment configuration.

*   `base_output_dir`: This corresponds to the `eval.output_dir` value from the YAML. Let's assume it's `~/slimsc/prune/results` as a common default.
*   `model_name`: From `eval.model_name`.
*   `dataset_name`: From `eval.dataset_name`.
*   `run_name`: This is constructed as follows (referencing the logic in `create_server_pbs_script`):
    *   If `eval_type` is `similarity`: `{pruning_strategy}{schedule_suffix}_n{n_start}_thresh{threshold_for_naming:.2f}_delay{num_steps_to_delay_pruning_for_naming}`
        *   `pruning_strategy`: From `eval.pruning_strategy`.
        *   `schedule_suffix`: `_annealing` if `eval.threshold_schedule` is `annealing`, otherwise `""`.
        *   `n_start`: From `eval.n_start`.
        *   `threshold_for_naming`: Use `eval.threshold`. Format to 2 decimal places.
        *   `num_steps_to_delay_pruning_for_naming`: From `eval.num_steps_to_delay_pruning` (defaults to 20 if not specified).
    *   If `eval_type` is `sc_control`: `sc_{n_start}_control`
        *   `n_start`: From `eval.n_start`.

Let's say you are running a `similarity` experiment with:
*   `eval.output_dir`: `~/my_results`
*   `eval.model_name`: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
*   `eval.dataset_name`: `gpqa_diamond`
*   `eval.n_start`: 8
*   `eval.pruning_strategy`: `random`
*   `eval.threshold`: 0.98
*   `eval.threshold_schedule`: `fixed`
*   `eval.num_steps_to_delay_pruning`: 20

The `run_name` would be `random_n8_thresh0.98_delay20`.

The target KVC file path would be: `~/my_results/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/gpqa_diamond/random_n8_thresh0.98_delay20/kvcache_usages.csv`

Set the `KVC_USAGE_FILE` environment variable *before* starting the server:

```bash
export KVC_USAGE_FILE="$HOME/my_results/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/gpqa_diamond/random_n8_thresh0.98_delay20/kvcache_usages.csv"
# Ensure the directory exists
mkdir -p "$(dirname "$KVC_USAGE_FILE")"
```

**3. Start the vLLM server:**

Construct the `vllm serve` command based on the `server` and `eval` parameters from your intended experiment configuration. The base command derived in the PBS script was:

`vllm serve <quoted_model_path> --tensor-parallel-size <tp_size> --port $PORT --seed 42 [optional args]`

For a local run, you can use the default port 8000 (`--port 8000` or omit `--port` entirely, as 8000 is the default).

Example command (using the parameters from step 2, assuming model path is `/path/to/your/model`):

```bash
vllm serve /path/to/your/model \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
```

Run this command in the terminal where you set up the environment variables. It will start the server. Wait until you see the message `Application startup complete.` in the server's output. This indicates the server is ready to accept connections. Keep this terminal open and the server running.

## Step 2: Running the Evaluation Client

The evaluation client script (`similarity_prune_eval.py` or `sc_control_eval.py`) needs to be run from the `slimsc` project root directory and configured via command-line arguments and the `VLLM_URL` environment variable.

**1. Set up the environment:**

Open a *new* terminal.
Source your Conda initialization script and activate the environment:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm
```

(Adjust the path `$HOME/miniconda3` if necessary).

Navigate to the `slimsc` project root directory (the one containing the `slimsc` package subdirectory):

```bash
cd /path/to/your/slimsc/project/root # Replace with actual path
# Example: if you are in slimsc/prune/jobs, run: cd ../../..
```

Set the `VLLM_URL` environment variable to point to your running server. If it's on the same machine using the default port:

```bash
export VLLM_URL="http://localhost:8000"
```

If the server is on a different machine or port, replace `localhost:8000` with the correct IP:Port (e.g., `http://192.168.1.100:8001`).

If your evaluation type is *not* `sc_control` and requires GPUs (this is less common as the model is on the server, but the client script *can* be configured to use GPUs via the YAML `client.gpus`), set `CUDA_VISIBLE_DEVICES` accordingly:

```bash
export CUDA_VISIBLE_DEVICES=0 # Example for client using one GPU
```
(Check the `create_client_pbs_script` logic for `cuda_export` - it only sets this for non-'sc_control' types if `client_gpus` > 0). For most use cases, this export might not be needed if the client script doesn't directly use GPUs.

**2. Construct and run the evaluation command:**

The command is `python -m slimsc.prune.evaluation.<eval_script_name>`. The arguments correspond to the `eval` section of the YAML.

*   **For `similarity` evaluation:**

    ```bash
    python -m slimsc.prune.evaluation.similarity_prune_eval \
        --n_start <eval.n_start> \
        --threshold <eval.threshold> \
        --pruning_strategy <eval.pruning_strategy> \
        --model_name <eval.model_name> \
        --model_identifier <eval.model_identifier> \
        --tokenizer_path <eval.tokenizer_path> \
        --dataset_name <eval.dataset_name> \
        --threshold_schedule <eval.threshold_schedule> \
        [--seed <eval.seed>] \
        [--num_steps_to_delay_pruning <eval.num_steps_to_delay_pruning>] \
        [--output_dir <eval.output_dir>] \
        [--num_qns <eval.num_qns> | --iterations <eval.iterations> | --start <eval.start> --end <eval.end>]
    ```

    Replace placeholders with values from your configuration. Square brackets `[]` denote optional arguments. Use quotes (`"`) around paths or arguments that might contain spaces. Ensure `tokenizer_path` is correct relative to the `slimsc` root directory if it's a relative path. The `output_dir` is usually an absolute path or relative to the *user's home directory* (`~`).

*   **For `sc_control` evaluation:**

    ```bash
    python -m slimsc.prune.evaluation.sc_control_eval \
        --n_start <eval.n_start> \
        --model_name <eval.model_name> \
        --model_identifier <eval.model_identifier> \
        [--tokenizer_path <eval.tokenizer_path>] \
        --dataset_name <eval.dataset_name> \
        [--output_dir <eval.output_dir>] \
        [--num_qns <eval.num_qns> | --iterations <eval.iterations> | --start <eval.start> --end <eval.end>]
    ```

    Replace placeholders with values from your configuration. `tokenizer_path` is optional for `sc_control_eval`.

Run the chosen command in the terminal where you set up the client environment and changed directory to the `slimsc` root.

The script will connect to the server and start the evaluation. Its output will appear in this terminal.

## Manual Cleanup

After the client script finishes successfully and you have handled the results, you *must* manually stop the vLLM server process running in the other terminal. Simply press `Ctrl+C` in the server terminal.

If the client script fails, the server will continue running. You should investigate the client failure and then manually stop the server (`Ctrl+C`).