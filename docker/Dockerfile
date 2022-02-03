FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-pip \
    wget

WORKDIR /src

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

ENV CONDA_PREFIX /root/miniconda3/envs/default

# Clear .bashrc (it refuses to run non-interactively otherwise).
RUN echo > ~/.bashrc

# Add conda logic to .bashrc.
RUN conda init bash

# Create new environment and install some dependencies.
RUN conda create -y -n transformers python=3.9 \
    nodejs \
    jupyterlab_widgets \
    ipywidgets \
    -c conda-forge

# Activate environment in .bashrc.
RUN echo "conda activate transformers" >> /root/.bashrc

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

RUN pip install jupyterlab \
                torch==1.10.1+cu113 \
                torchvision==0.11.2+cu113 \
                torchaudio==0.10.1+cu113 \
                pytorch_lightning \
                wandb \
                transformers \
                datasets \
                setuptools==59.5.0 \
                -f https://download.pytorch.org/whl/cu113/torch_stable.html

ENTRYPOINT ["/bin/bash", "-c", "jupyter lab --ip=* --allow-root"]
