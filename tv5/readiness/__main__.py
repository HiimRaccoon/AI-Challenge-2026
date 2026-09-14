from __future__ import annotations
import argparse, json
from pathlib import Path
from .validator import ReadinessConfig, validate_readiness

parser = argparse.ArgumentParser(description="Read-only WP13 artifact readiness status")
parser.add_argument("root", type=Path); parser.add_argument("--kind", choices=("wp03", "wp04"), default="wp03")
parser.add_argument("--component", default="WP03 Visual"); parser.add_argument("--expected-video", action="append", default=[])
parser.add_argument("--manifest-name", default="manifest.json"); parser.add_argument("--expected-run-id"); parser.add_argument("--expected-digest")
parser.add_argument("--known-handover", action="store_true"); parser.add_argument("--upstream-capability", action="store_true")
parser.add_argument("--modality", choices=("OCR", "ASR", "Object", "Metadata")); parser.add_argument("--records-name", default="records.json"); parser.add_argument("--index-name")
args = parser.parse_args()
config = ReadinessConfig(args.component, args.kind, args.root, tuple(args.expected_video), args.known_handover, args.upstream_capability, args.expected_run_id, args.manifest_name, args.expected_digest, args.modality, args.records_name, args.index_name)
print(json.dumps(validate_readiness(config).to_dict(), indent=2, default=str))
