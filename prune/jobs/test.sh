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
if [ -z "$SERVER_IP" ] || [ "$SERVER_IP" == "ERROR_NO_IP" ] || [ "$SERVER_IP" == "ERROR_NO_PORT" ]; then # Check for both error types
    echo "[$(date)] Error: Server IP file contains error indicator: '$SERVER_IP'" >&2
    exit 1
fi

# Check if it contains both IP and port in format IP:PORT
if [[ "$SERVER_IP" == *":"* ]]; then
    # Extract the port from the IP:PORT format
    SERVER_PORT=$(echo "$SERVER_IP" | cut -d':' -f2)
    SERVER_IP=$(echo "$SERVER_IP" | cut -d':' -f1)
    export VLLM_URL="http://${{SERVER_IP}}:${{SERVER_PORT}}"
    echo "Read Server IP: $SERVER_IP, Port: $SERVER_PORT"
else
    # Fallback to default port 8000 for backward compatibility
    export VLLM_URL="http://${{SERVER_IP}}:8000"
    echo "Read Server IP: $SERVER_IP (using default port 8000)"
fi
echo "Set VLLM_URL=$VLLM_URL"

# --- Wait for Server Ready String in Log (Phase 2) ---
# IMPORTANT: The server might delete SERVER_VLLM_LOG_RELPATH upon finishing.
# The client needs to see the "ready" string before the server finishes and deletes the log.
# If the server starts fast and the client has a delay, this check might fail after the server exits.
# We add a check to see if the server job still exists during the wait loop.
echo "Waiting for vLLM server to be ready (checking for '{SERVER_READY_STRING}' in $SERVER_VLLM_LOG_RELPATH)..."
MAX_LOG_WAIT_SEC=720 # 12 minutes total timeout for server to become ready
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