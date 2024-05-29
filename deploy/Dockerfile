# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND noninteractive

# Update packages and install necessary tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa

# Update packages again and install Python 3.9 and pip
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    redis-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python3.9 to python3
RUN rm -rf /usr/bin/python3
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Set python3.9 as default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip
RUN python3.9 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install project dependencies
RUN python3.9 -m pip install -r requirements.txt

# Prepare directories and dump data
RUN mkdir -p resources/datasets resources/checkpoints resources/db logs \
    && python3 run_dump_data.py

# Download pretrained model
RUN wget -O resources/checkpoints/bert4rec_ml-25m_best.ckpt https://uithcm-my.sharepoint.com/:u:/g/personal/18520216_ms_uit_edu_vn/EVCvHZg7QFZGlis704IiPdIBMJxIK37tcVGUM9zY-LzlCw?e=tCgA0J&download=1

# Default command (will be overridden by services in docker-compose.yml)
CMD ["sh", "-c", "echo 'No default command specified'"]
