"""
Quick test to reproduce JSON schema error with minimal prompt
"""
import json
import requests
import time

API_BASE = "http://localhost:8000"

# Schema extracted from the original problematic request
SCHEMA = {
    "type": "object",
    "required": ["unit", "summary", "entities", "situations", "notes", "topic_cards"],
    "properties": {
        "unit": {
            "type": "object",
            "required": ["type", "chapter_index", "section_index", "title"],
            "properties": {
                "type": {"type": "string"},
                "chapter_index": {"type": "integer"},
                "section_index": {"type": ["integer", "null"]},
                "title": {"type": "string"},
                "role": {"type": "string"}
            }
        },
        "distance": {"type": "integer"},
        "detail_level": {"type": "string"},
        "summary": {
            "type": "object",
            "required": ["lines"],
            "properties": {
                "lines": {
                    "type": "array",
                    "minItems": 8,
                    "maxItems": 10,
                    "items": {"type": "string", "minLength": 1}
                },
                "beats": {"type": "array", "items": {"type": "string"}}
            }
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "entity_type", "coverage", "importance"],
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "coverage": {
                        "type": "string",
                        "enum": ["mention", "overview", "detail", "analysis"]
                    },
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            }
        },
        "situations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "coverage", "importance"],
                "properties": {
                    "label": {"type": "string"},
                    "coverage": {"type": "string"},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                }
            }
        },
        "notes": {
            "type": "object",
            "required": ["open_questions", "continuity_hooks"],
            "properties": {
                "open_questions": {"type": "array", "items": {"type": "string"}},
                "continuity_hooks": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        },
        "topic_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["topic_id", "label", "facet", "narrative_function", "summary", "importance", "duplication_sensitivity", "linked_events"],
                "properties": {
                    "topic_id": {"type": "string"},
                    "label": {"type": "string"},
                    "facet": {"type": "string"},
                    "narrative_function": {
                        "type": "string",
                        "enum": ["introduce", "elaborate", "reference"]
                    },
                    "summary": {"type": "string"},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "duplication_sensitivity": {"type": "boolean"},
                    "evidence_spans": {"type": "array", "items": {"type": "string"}},
                    "linked_events": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    },
    "additionalProperties": False
}

PROMPT = """You are an editorial analyst. Summarize the provided UNIT as AI-friendly JSON so another model can maintain continuity without copying this text into its own prose.

CRITICAL RULES:
- Return ONLY a JSON object (no extra text or fences).
- The summary.lines field must contain 8-10 clear, non-redundant lines (strings, not objects).
- Score 'entities' with coverage (mention/overview/detail/analysis) and importance [0..1].
- Do not invent facts; ground every statement in the unit text.

UNIT:
In 2025, Francisco worked on two major projects: CronoReal and Metavox.

CronoReal is a VR platform for historical time travel. It uses photogrammetry from real archaeological sites to create precise 3D reconstructions. Francisco collaborated with archaeologists and historians to validate proportions, colors, and historical rituals. Even simple details like stone tones required hours of discussion to ensure accuracy.

Metavox is a conversational AI for NPCs in VR/AR environments. It enables fluid, contextual dialogues tested in museums, theme parks, and video games. The technical challenge was massive: responses needed to be instant, immersive, and error-free to maintain the magic of the experience.

Paradoxically, Metavox crystallized when Francisco stopped living in a marathon of 14-16 hour days. He learned to truly delegate to his team. He started meditating each morning, reading calmly at night, and walking his dog Tadeo even when emails kept piling up.

He transitioned from exhausting marathons to a sustainable pace that sharpens his hunger for innovation rather than burning him out. He continues to approach every project with curiosity and humility. Immersive technology remains his obsession, but now he lives it from a more sustainable, conscious place.

Now the challenge is to sustain this human rhythm while CronoReal and Metavox expand. He's focused on building independent teams, maintaining passion without burnout, and creating impact that lasts.
"""

def run_model_test(model_name, test_name):
    """Test a specific model"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}\n")

    request_data = {
        "prompt": PROMPT,
        "generator_model": model_name,
        "json_schema": SCHEMA,
        "qa_layers": [],  # Bypass QA for faster testing
        "max_iterations": 3,
        "verbose": True
    }

    try:
        print("Sending request...")
        response = requests.post(f"{API_BASE}/generate", json=request_data, timeout=10)

        if response.status_code != 200:
            print(f"[X] ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return None

        result = response.json()
        session_id = result.get('session_id')
        print(f"[OK] Session created: {session_id}")

        # Wait for completion
        max_wait = 90
        start_time = time.time()
        last_status = None

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{API_BASE}/status/{session_id}")

            if status_response.status_code != 200:
                print(f"[X] Failed to get status")
                return None

            status_data = status_response.json()
            current_status = status_data.get('status')

            if current_status != last_status:
                print(f"  Status: {current_status}")
                last_status = current_status

            if current_status == 'approved':
                print(f"\n[SUCCESS] Model generated valid JSON!")
                return {'status': 'approved', 'session_id': session_id}

            elif current_status in ['failed', 'stopped']:
                print(f"\n[FAILED]")

                # Get verbose log to see errors
                if 'verbose_log' in status_data:
                    print(f"\nAnalyzing errors from verbose log:")
                    for entry in status_data['verbose_log']:
                        if entry.get('event') == 'json_guard_result':
                            iteration = entry.get('data', {}).get('iteration')
                            valid = entry.get('data', {}).get('valid')
                            errors = entry.get('data', {}).get('errors', [])

                            if not valid:
                                print(f"\n  Iteration {iteration} - INVALID JSON:")
                                for error in errors:
                                    print(f"    - {error}")

                return {'status': current_status, 'session_id': session_id, 'errors': errors}

            time.sleep(2)

        print(f"\n[TIMEOUT]")
        return {'status': 'timeout', 'session_id': session_id}

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None

def main():
    print("="*80)
    print("QUICK JSON SCHEMA ERROR REPRODUCTION")
    print("="*80)

    print("\nSchema overview:")
    print("  - entities[].coverage enum: ['mention', 'overview', 'detail', 'analysis']")
    print("  - topic_cards[].narrative_function enum: ['introduce', 'elaborate', 'reference']")
    print("  - notes.open_questions: array of strings")
    print("  - notes.continuity_hooks: array of strings")

    tests = [
        ('grok-4-1-fast-non-reasoning', 'Grok-4-1-fast (original failing model)'),
        ('gpt-5-mini', 'GPT-5-mini (cheaper alternative)'),
        ('gpt-5.1', 'GPT-5.1 (full version)'),
        ('claude-opus-4.1', 'Claude Opus 4.1'),
    ]

    results = {}

    for model, name in tests:
        result = run_model_test(model, name)
        results[model] = result
        time.sleep(2)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for model, name in tests:
        result = results.get(model)
        if result:
            status = result.get('status', 'unknown')
            symbol = "[OK]" if status == 'approved' else "[X] "
            print(f"{symbol} {name}: {status}")
        else:
            print(f"[X]  {name}: error")

    # Identify which models work
    working_models = [m for m, r in results.items() if r and r.get('status') == 'approved']
    failing_models = [m for m, r in results.items() if r and r.get('status') != 'approved']

    if working_models:
        print(f"\n[OK] Models that work: {', '.join(working_models)}")
    if failing_models:
        print(f"\n[X]  Models that fail: {', '.join(failing_models)}")

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
