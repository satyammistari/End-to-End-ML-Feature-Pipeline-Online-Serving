#!/usr/bin/env bash
# Generate Python gRPC stubs from feature_service.proto
set -euo pipefail

PROTO_DIR="src/api/proto"
OUT_DIR="src/api/proto"

# Install grpcio-tools if not present
pip install grpcio-tools --quiet

python -m grpc_tools.protoc \
  -I "${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  "${PROTO_DIR}/feature_service.proto"

echo "Stubs generated in ${OUT_DIR}:"
ls -1 "${OUT_DIR}"/*.py
