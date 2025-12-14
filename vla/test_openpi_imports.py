#!/usr/bin/env python3
"""Quick test to verify OpenPI imports work correctly."""

import sys

print("Testing OpenPI imports...")

try:
    from openpi.training import config as openpi_config
    print("[OK] openpi.training.config imported")
except ImportError as e:
    print(f"[X] Failed to import openpi.training.config: {e}")
    sys.exit(1)

try:
    from openpi.policies import policy_config
    print("[OK] openpi.policies.policy_config imported")
except ImportError as e:
    print(f"[X] Failed to import openpi.policies.policy_config: {e}")
    sys.exit(1)

try:
    from openpi.shared import download
    print("[OK] openpi.shared.download imported")
except ImportError as e:
    print(f"[X] Failed to import openpi.shared.download: {e}")
    sys.exit(1)

print("\n[SUCCESS] All OpenPI imports successful!")

# Test getting a config
print("\nTesting config loading...")
try:
    config = openpi_config.get_config("pi05_droid")
    print(f"[OK] Loaded config: pi05_droid")
    print(f"     Model type: {config.model.model_type}")
except Exception as e:
    print(f"[X] Failed to load config: {e}")
    sys.exit(1)

print("\n[SUCCESS] OpenPI integration is working!")
