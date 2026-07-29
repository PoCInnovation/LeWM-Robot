FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
ARG GROOT_REF=ead52833afbbf4243f8cd5e7664f48a94de03b19
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto "=https" --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /

RUN apt-get remove -y python3-cryptography python3-cryptography-dev 2>/dev/null || true
RUN python3 -m pip install --ignore-installed --break-system-packages cryptography

RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git /Isaac-GR00T
WORKDIR /Isaac-GR00T
RUN git checkout "${GROOT_REF}" && git rev-parse --verify HEAD
RUN uv sync --python 3.10
RUN uv pip install -e .

ENV VIRTUAL_ENV="/Isaac-GR00T/.venv"
ENV PATH="/Isaac-GR00T/.venv/bin:$PATH"
RUN echo "source /Isaac-GR00T/.venv/bin/activate" >> /root/.bashrc

WORKDIR /Isaac-GR00T/gr00t/eval/real_robot/SO100
RUN uv pip install -e .
RUN uv pip install feetech-servo-sdk

# for rerun GUI
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxcb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libgl1 \
    libegl1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY docker/utils.sh /root/tmp/utils.sh
RUN cat /root/tmp/utils.sh >> /root/.bashrc && rm /root/tmp/utils.sh
