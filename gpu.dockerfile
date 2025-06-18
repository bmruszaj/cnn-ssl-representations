FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update -qq \
    && apt-get install -yqq \
      git \
      python3.12 \
      python3-pip \
      gcc \
      graphviz \
      graphviz-dev

RUN pip install -U pip

WORKDIR /assignment

COPY ./requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt
RUN rm ./requirements-gpu.txt

ENTRYPOINT ["jupyter", "lab", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--notebook-dir", "/assignment"]
