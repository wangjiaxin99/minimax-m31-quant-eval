"""Long teacher-forced KL scored inside vLLM.

Reference answers are generated once, then the reference model and every candidate read the same token sequences
through one prefill path (prompt log-probabilities). A worker extension reduces the logits on the GPU:
the reference keeps its Top-K per position; a candidate returns per-position KL for several K, top-1 flips and the
target token's log-probability. Nothing vocabulary-sized leaves the worker.
"""
