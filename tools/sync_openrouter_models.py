"""
OpenRouter Models Synchronization Script
=========================================

Safely syncs OpenRouter models to model_specs.json with validation and backup.

This script:
1. Fetches current models from OpenRouter API
2. Filters only desired providers (meta-llama, mistralai, deepseek, qwen)
3. Updates model_specs.json while preserving existing providers
4. Creates backups before modification
5. Validates data integrity throughout the process

Usage:
    python tools/sync_openrouter_models.py [--dry-run] [--verbose] [--providers PROVIDERS]

Options:
    --dry-run     Show what would be updated without making changes
    --verbose     Show detailed progress information
    --providers   Comma-separated list of providers to sync (default: meta-llama,mistralai,deepseek,qwen)
"""

import argparse
import os
import sys

# Add parent directory to path for json_utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

import json_utils as json

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Default providers to sync from OpenRouter
DEFAULT_PROVIDERS = [
    "meta-llama",
    "mistralai",
    "deepseek",
    "qwen",
    "cohere",
    "nousresearch"
]

# Minimum required fields for a model spec
REQUIRED_MODEL_FIELDS = [
    "model_id",
    "name",
    "description",
    "input_tokens",
    "output_tokens",
    "context_window",
    "pricing",
    "capabilities",
    "provider"
]


class OpenRouterSync:
    """Handles synchronization of OpenRouter models to local model_specs.json"""

    def __init__(self, api_key: str, verbose: bool = False):
        self.api_key = api_key
        self.verbose = verbose
        self.models_url = "https://openrouter.ai/api/v1/models"
        self.project_root = Path(__file__).parent.parent
        self.specs_file = self.project_root / "model_specs.json"
        self.backup_dir = self.project_root / "backups"

    def log(self, message: str, force: bool = False):
        """Print message if verbose mode is enabled or force is True"""
        if self.verbose or force:
            print(message)

    def fetch_openrouter_models(self) -> List[Dict[str, Any]]:
        """Fetch current models from OpenRouter API with error handling"""
        self.log("Fetching models from OpenRouter API...")

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.get(self.models_url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if "data" not in data:
                raise ValueError("API response missing 'data' field")

            models = data["data"]

            if not isinstance(models, list):
                raise ValueError("API 'data' field is not a list")

            if len(models) == 0:
                raise ValueError("API returned empty models list")

            self.log(f"Successfully fetched {len(models)} models from OpenRouter")
            return models

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch models from OpenRouter: {str(e)}")
        except ValueError as e:
            raise Exception(f"Invalid API response: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error fetching models: {str(e)}")

    def filter_models_by_providers(
        self, models: List[Dict[str, Any]], providers: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Filter models to only include specified providers (None = all providers)"""
        if providers is None or len(providers) == 0:
            self.log(f"Syncing ALL models (no provider filter)")
        else:
            self.log(f"Filtering models for providers: {', '.join(providers)}")

        filtered = {}

        for model in models:
            model_id = model.get("id", "")

            # If no provider filter, include all models
            if providers is None or len(providers) == 0:
                converted = self.convert_openrouter_model(model)
                if converted:
                    filtered[model_id] = converted
                continue

            # Check if model belongs to one of our target providers
            provider_prefix = model_id.split("/")[0] if "/" in model_id else None

            if provider_prefix not in providers:
                continue

            # Convert OpenRouter format to our format
            converted = self.convert_openrouter_model(model)
            if converted:
                filtered[model_id] = converted

        self.log(f"Filtered to {len(filtered)} models from target providers")
        return filtered

    def convert_openrouter_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenRouter model format to our model_specs.json format"""
        try:
            model_id = model.get("id", "")
            if not model_id:
                self.log(f"Skipping model without ID: {model}")
                return None

            # Extract pricing (OpenRouter uses different units)
            pricing_data = model.get("pricing", {})
            pricing = {
                "input_per_million": float(pricing_data.get("prompt", 0)) * 1000000,
                "output_per_million": float(pricing_data.get("completion", 0)) * 1000000
            }

            # Extract context window info
            context_length = model.get("context_length", 8192)
            top_provider = model.get("top_provider", {})
            max_completion = top_provider.get("max_completion_tokens")

            # If max_completion is None or 0, use reasonable default
            if not max_completion or max_completion <= 0:
                max_completion = min(context_length // 2, 16384)  # Default to half context or 16K max

            # Extract capabilities from architecture
            architecture = model.get("architecture", {})
            modalities = architecture.get("input_modalities", [])
            capabilities = ["text"]  # All models support text

            if "image" in modalities:
                capabilities.append("vision")
            if "audio" in modalities:
                capabilities.append("audio")

            # Add coding capability for models that typically code well
            if any(keyword in model_id.lower() for keyword in ["coder", "code", "instruct"]):
                capabilities.append("coding")

            # Build the model spec
            spec = {
                "model_id": model_id,
                "name": model.get("name", model_id),
                "description": model.get("description", f"Model {model_id} via OpenRouter"),
                "input_tokens": context_length,
                "output_tokens": max_completion,
                "context_window": context_length,
                "pricing": pricing,
                "capabilities": capabilities,
                "provider": "openrouter",
                "verified_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "source": f"https://openrouter.ai/models/{model_id}"
            }

            # Validate required fields
            for field in REQUIRED_MODEL_FIELDS:
                if field not in spec:
                    self.log(f"Model {model_id} missing required field: {field}")
                    return None

            return spec

        except Exception as e:
            self.log(f"Error converting model {model.get('id', 'unknown')}: {str(e)}")
            return None

    def load_current_specs(self) -> Dict[str, Any]:
        """Load current model_specs.json with validation"""
        self.log(f"Loading current specs from {self.specs_file}")

        try:
            with open(self.specs_file, "r", encoding="utf-8") as f:
                specs = json.load(f)

            if "model_specifications" not in specs:
                raise ValueError("model_specs.json missing 'model_specifications' key")

            self.log("Successfully loaded current specs")
            return specs

        except FileNotFoundError:
            raise Exception(f"model_specs.json not found at {self.specs_file}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in model_specs.json: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading model_specs.json: {str(e)}")

    def create_backup(self) -> Path:
        """Create timestamped backup of model_specs.json"""
        self.backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"model_specs_backup_{timestamp}.json"

        self.log(f"Creating backup at {backup_file}")

        try:
            with open(self.specs_file, "r", encoding="utf-8") as src:
                with open(backup_file, "w", encoding="utf-8") as dst:
                    dst.write(src.read())

            self.log("Backup created successfully")
            return backup_file

        except Exception as e:
            raise Exception(f"Failed to create backup: {str(e)}")

    def update_specs(
        self, specs: Dict[str, Any], new_models: Dict[str, Dict[str, Any]], dry_run: bool = False
    ) -> Dict[str, Any]:
        """Update specs with new OpenRouter models while preserving others"""
        self.log("Updating model specifications...")

        if "openrouter" not in specs["model_specifications"]:
            specs["model_specifications"]["openrouter"] = {}

        old_count = len(specs["model_specifications"]["openrouter"])

        # Update or add models
        specs["model_specifications"]["openrouter"] = new_models

        new_count = len(new_models)

        self.log(f"OpenRouter models: {old_count} -> {new_count}", force=True)
        self.log(f"Models added/updated: {new_count}", force=True)

        # Show sample of changes
        if self.verbose and new_models:
            print("\nSample of updated models:")
            for i, (model_id, spec) in enumerate(list(new_models.items())[:5]):
                print(f"  - {model_id}: {spec['name']}")
            if len(new_models) > 5:
                print(f"  ... and {len(new_models) - 5} more")

        return specs

    def save_specs(self, specs: Dict[str, Any], dry_run: bool = False):
        """Save updated specs to file"""
        if dry_run:
            self.log("DRY RUN: Would save updated specs to file", force=True)
            return

        self.log(f"Saving updated specs to {self.specs_file}")

        try:
            with open(self.specs_file, "w", encoding="utf-8") as f:
                json.dump(specs, f, indent=2, ensure_ascii=False)

            self.log("Specs saved successfully", force=True)

        except Exception as e:
            raise Exception(f"Failed to save specs: {str(e)}")

    def validate_update(
        self, old_specs: Dict[str, Any], new_specs: Dict[str, Any]
    ) -> bool:
        """Validate that update didn't break anything"""
        self.log("Validating update...")

        try:
            # Check main structure preserved
            assert "model_specifications" in new_specs
            assert "aliases" in new_specs
            assert "default_models" in new_specs

            # Check other providers not affected
            for provider in old_specs["model_specifications"]:
                if provider != "openrouter":
                    assert provider in new_specs["model_specifications"]
                    assert len(old_specs["model_specifications"][provider]) == len(
                        new_specs["model_specifications"][provider]
                    )

            self.log("Validation passed")
            return True

        except AssertionError as e:
            self.log(f"Validation failed: {str(e)}", force=True)
            return False

    def sync(
        self, providers: List[str], dry_run: bool = False
    ) -> bool:
        """Main synchronization process"""
        try:
            # Step 1: Fetch models from OpenRouter
            models = self.fetch_openrouter_models()

            # Step 2: Filter to target providers
            filtered_models = self.filter_models_by_providers(models, providers)

            if len(filtered_models) == 0:
                self.log("WARNING: No models found for target providers!", force=True)
                return False

            # Step 3: Load current specs
            old_specs = self.load_current_specs()

            # Step 4: Create backup
            if not dry_run:
                self.create_backup()

            # Step 5: Update specs
            new_specs = self.update_specs(old_specs.copy(), filtered_models, dry_run)

            # Step 6: Validate
            if not self.validate_update(old_specs, new_specs):
                self.log("ERROR: Validation failed, aborting update", force=True)
                return False

            # Step 7: Save
            self.save_specs(new_specs, dry_run)

            self.log("\nSynchronization completed successfully!", force=True)
            return True

        except Exception as e:
            self.log(f"\nERROR: Synchronization failed: {str(e)}", force=True)
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Sync OpenRouter models to model_specs.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )
    parser.add_argument(
        "--providers",
        type=str,
        help=f"Comma-separated list of providers (default: {','.join(DEFAULT_PROVIDERS)})"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sync ALL models from ALL providers (ignores --providers)"
    )

    args = parser.parse_args()

    # Load API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables")
        print("Please set OPENROUTER_API_KEY in your .env file")
        sys.exit(1)

    # Parse providers list
    if args.all:
        providers = None  # None means sync ALL providers
    elif args.providers:
        providers = args.providers.split(",")
    else:
        providers = DEFAULT_PROVIDERS

    # Create syncer and run
    syncer = OpenRouterSync(api_key, verbose=args.verbose)

    print("=" * 80)
    print("OpenRouter Models Synchronization")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE UPDATE'}")
    if providers is None:
        print(f"Target providers: ALL PROVIDERS (342 models)")
    else:
        print(f"Target providers: {', '.join(providers)}")
    print("=" * 80)
    print()

    success = syncer.sync(providers, dry_run=args.dry_run)

    if success:
        print("\nSync completed successfully!")
        sys.exit(0)
    else:
        print("\nSync failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
