ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=22.04_20240212

FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}

ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND noninteractive

# Install required ubuntu packages for setting up python 3.10
RUN apt update && \
    apt install -y curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    # TODO REMOVE GIT
    apt update && apt install -y python3.10 git && \
    # apt update && apt install -y python3.11 && \
    apt-get clean

# Install pip for python3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install uv 
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="/root/.cargo/bin:$PATH"' >> /root/.bashrc && \
    export PATH="/root/.cargo/bin:$PATH"

RUN rm -rf /var/lib/apt/lists/*

# Uninstall build packages
RUN apt autoremove -y curl software-properties-common

# Download the sources of apt packages within the container for standard legal compliance
RUN sed -i 's/# deb-src/deb-src/g' /etc/apt/sources.list
RUN apt update
# xz-utils is needed to pull the source and unpack them correctly
RUN apt install xz-utils -y
RUN mkdir -p /legal/source
WORKDIR /legal/source
# Read installed packages, strip all but the package names, pipe to 'apt source' to download respective packages
RUN apt list --installed |  grep -i installed | sed 's|\(.*\)/.*|\1|' | xargs apt source --download-only
# The source is saved in directories as well as tarballs in the current dir
RUN rm xz-utils*
COPY LICENSE-3rd-party.txt /legal/

# Copy and Install agent specific modules
# COPY src/aiq_agent /opt/src/aiq_agent

WORKDIR /opt/src/aiq_agent

ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# Verify uv installation
RUN uv --version

RUN uv venv --seed .venv --python 3.11

# Create entrypoint script
RUN echo '#!/bin/bash\n\
source .venv/bin/activate\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# # Clone AIQ toolkit
RUN git clone -b develop https://github.com/NVIDIA/AIQToolkit.git aiqtoolkit && \
    cd aiqtoolkit && \
    git checkout 67f5e9b && \
    cd ..

# RUN uv pip install -e aiva_agent

ENTRYPOINT ["/entrypoint.sh"]
