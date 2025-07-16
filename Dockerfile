FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /slimsc

COPY requirements/ ./requirements
RUN pip install --no-cache-dir -r requirements/requirements.txt

COPY . .

RUN mkdir -p prune/results

CMD ["/bin/bash"]
