# Thêm vào src/wp04/cli.py

import os

def _doctor(arguments: argparse.Namespace) -> int:
    config = load_config(Path(arguments.config))
    modalities = config["modalities"]
    checks: dict[str, Any] = {}
    ok = True
    for name in ("ocr", "asr", "object"):
        settings = modalities.get(name, {})
        enabled = bool(settings.get("enabled", False))
        command_env = settings.get("settings", {}).get("command_env")
        configured_command = os.environ.get(command_env, "").strip() if command_env else ""
        status = "disabled" if not enabled else ("ok" if configured_command else "missing_command")
        if enabled and not configured_command:
            ok = False
        checks[name] = {
            "enabled": enabled,
            "command_env": command_env,
            "configured": bool(configured_command),
            "status": status,
        }
    result = {"production_ready": ok, "checks": checks}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 2


# Trong build_parser():
doctor = subcommands.add_parser("doctor", help="check WP04_* command env vars before preprocess")
doctor.add_argument("--config", required=True)

# Trong main():
if arguments.command == "doctor":
    return _doctor(arguments)
