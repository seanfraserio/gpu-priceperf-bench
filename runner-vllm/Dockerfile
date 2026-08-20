FROM vllm/vllm-openai:v0.27.1

ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir hf_transfer "pydantic>=2.7" "httpx>=0.27"

COPY src/gppb /opt/gppb/gppb
ENV PYTHONPATH=/opt/gppb

COPY runner-vllm/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
