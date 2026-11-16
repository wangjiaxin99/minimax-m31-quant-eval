m31-eval reference --dataset data/mixed/quick240.jsonl \
  --base-url http://localhost:8000/v1 --reference-kind mxfp8 \
  --metric coarse --tokens 32 --positions {0..31} \
  --max-prompt-tokens 65536 --concurrency 2 \
  --time-budget 1800 --output cache/mxfp8

m31-eval evaluate --reference cache/mxfp8 \
  --base-url http://localhost:8001/v1 --concurrency 2 \
  --time-budget 1800 --output logs/mxfp4-r1
