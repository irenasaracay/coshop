#!/usr/bin/env bash
# Launch the CoShop vector search server under gunicorn.
#
# EMBEDDING_API_URL must point at any OpenAI-compatible embeddings endpoint.
# "/v1" is appended automatically if the URL does not already end in it.
#
# EMBEDDING_MODEL is the model id sent to that endpoint; it must match the name
# the backend serves the model under (e.g. vLLM's --served-model-name, or
# "text-embedding-3-small" for OpenAI). It is required.
#
# Auth uses the standard openai client (OPENAI_API_KEY); set a dummy value for
# local servers that do not check it.
#
# Usage (local vLLM serving an embedding model):
#   vllm serve Qwen/Qwen3-Embedding-0.6B --served-model-name my-embed-model --port 8000
#   EMBEDDING_API_URL=http://localhost:8000 EMBEDDING_MODEL=my-embed-model ./launch_vector_search_server.sh
#
# A different embedding server, e.g. a remote vLLM/TEI host:
#   EMBEDDING_API_URL=http://my-embed-host:8000/v1 EMBEDDING_MODEL=my-embed-model \
#     CUDA_VISIBLE_DEVICES=0 ./launch_vector_search_server.sh
#
# OpenAI's hosted embeddings:
#   OPENAI_API_KEY=sk-... EMBEDDING_API_URL=https://api.openai.com \
#     EMBEDDING_MODEL=text-embedding-3-small ./launch_vector_search_server.sh

set -euo pipefail

: "${EMBEDDING_API_URL:?EMBEDDING_API_URL must be set}"
: "${EMBEDDING_MODEL:?EMBEDDING_MODEL must be set (the model id the backend serves, e.g. the vLLM --served-model-name)}"
export EMBEDDING_MODEL

exec gunicorn \
  --workers 1 \
  --threads 1 \
  --bind 0.0.0.0:3004 \
  "coshop.tools.vector_search_server:create_app()"
