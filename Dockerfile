FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements/ ./requirements
RUN pip install --no-cache-dir -r requirements/requirements.txt

RUN mkdir -p slimsc

COPY . slimsc

RUN mkdir -p slimsc/prune/results

CMD ["/bin/bash"]
