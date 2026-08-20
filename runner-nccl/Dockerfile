FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential libnccl2 libnccl-dev python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && make -C /opt/nccl-tests -j"$(nproc)"

RUN pip3 install --no-cache-dir "pydantic>=2.7" "httpx>=0.27"
COPY src/gppb /opt/gppb/gppb
ENV PYTHONPATH=/opt/gppb

COPY runner-nccl/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
