ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY reef_marlin_patch.py /opt/nvfp4-marlin-order/reef_marlin_patch.py
COPY vendor/ /opt/nvfp4-marlin-order/vendor/
RUN python3 /opt/nvfp4-marlin-order/reef_marlin_patch.py apply \
      --source /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py \
      --manifest /opt/nvfp4-marlin-order/manifest.json \
    && python3 /opt/nvfp4-marlin-order/reef_marlin_patch.py verify \
      --source /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py \
      --manifest /opt/nvfp4-marlin-order/manifest.json
