FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements/ ./requirements
RUN pip install --no-cache-dir -r requirements/requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "slimsc.prune.evaluation.sc_control_eval"]
