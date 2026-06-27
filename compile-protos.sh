#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Compiling gRPC Python stubs from ./protos..."

# Explicitly use the venv Python to ensure grpc_tools is found
./.venv/bin/python -m grpc_tools.protoc \
  -I./protos \
  --python_out=. \
  --grpc_python_out=. \
  ./protos/voice.proto

echo "✅ Successfully generated voice_pb2.py and voice_pb2_grpc.py!"

