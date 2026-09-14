---
language:
- vi
- en
tags:
- video-retrieval
- aic2026
- model-bundle
---

# AIC runtime model bundle

This is a deployment bundle of existing model files, not a newly trained model.
No competition video, image, transcript, embedding, submission, credentials or
environment is included. `bundle-manifest.json` records file sizes and SHA-256.
Use `scripts/model_bundle.py restore` from the companion GitHub code repository.

## Models and provenance

| Component | Source / identity |
| --- | --- |
| MetaCLIP2 | https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu/tree/c139061af7b10fdb2e754b60d2b1182a3d5526c2 |
| BGE-VL-large | https://huggingface.co/BAAI/BGE-VL-large/tree/40fb48217f521df22a2a5bf15edd52ed1146ef05 |
| Perception Encoder | https://huggingface.co/facebook/PE-Core-B16-224/tree/a16450b46fef32363459920c2685a1b4ef13dcd9 |
| BEiT-3 COCO retrieval base | https://github.com/microsoft/unilm/tree/833df7e7832e5064a281131ee64a481afa8e5b95/beit3 ; checkpoint identity is recorded by file hash |
| CRAFT | EasyOCR detector `craft_mlt_25k.pth`; https://github.com/JaidedAI/EasyOCR |
| PARSeq-VN | Team-provided `best-parseq.ckpt`; architecture https://github.com/baudm/parseq/tree/1572487d138cdf0340fb2a2f1437cc0cca05eabf |
| ChunkFormer | https://huggingface.co/khanhld/chunkformer-ctc-large-vie/tree/311fc03558a895dc2b32957f2fb4236c7fb1455b ; deployed local files are identified by their hashes |
| RF-DETR Medium | Local deployed `rf-detr-medium.pth`; https://github.com/roboflow/rf-detr ; checkpoint identity is recorded by file hash |

Original authors retain their rights. MetaCLIP2 and ChunkFormer model cards state
CC-BY-NC-4.0. Preserve upstream attribution and applicable notices; do not assign
one blanket code license to this bundle. The publisher must add the provenance
and redistribution terms for the team's PARSeq-VN checkpoint before publishing;
the architecture's license alone does not establish the checkpoint's terms.

## Layout

`project/` contains files restored to their runtime paths inside the code repo.
`hub/` contains only selected pinned snapshots, restored to `HF_HUB_CACHE` under
their original repository IDs. These are weights, model configs, tokenizer files
and required BGE-VL Python loaders. Python environments and third-party source
checkouts are installed separately from the GitHub instructions.

Optional Ollama `qwen2.5:3b-instruct` is provisioned separately with `ollama pull`.
An Ollama model tag is not interchangeable with a Transformers checkpoint.
External VLM is disabled in the baseline and has no selected weight in this bundle.
