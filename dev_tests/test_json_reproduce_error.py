"""
Reproduce the JSON schema error from session 0097ba67-ce49-462d-ae26-e8741e0bcce9
Test with multiple models and Gran Sabio fallback
"""
import sqlite3
import json
import requests
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "debugger_history.db"
SESSION_ID = "0097ba67-ce49-462d-ae26-e8741e0bcce9"
API_BASE = "http://localhost:8000"

def extract_original_request():
    """Extract the original request from database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT request_json FROM sessions WHERE session_id = ?", (SESSION_ID,))
    session = cursor.fetchone()
    conn.close()

    if session:
        return json.loads(session['request_json'])
    return None

def run_model_test(model_name, request_data, test_name):
    """Test a specific model with the request"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}\n")

    # Modify request to use specific model
    test_request = request_data.copy()
    test_request['generator_model'] = model_name

    # Reduce prompt size for faster testing (use first 1000 chars)
    if len(test_request['prompt']) > 5000:
        print("PROMPT TOO LONG - Using truncated version for faster testing")
        # Keep the schema and instructions, truncate the middle content
        parts = test_request['prompt'].split('UNIT:')
        if len(parts) > 1:
            instruction_part = parts[0]
            unit_part = parts[1]
            # Keep first 2000 chars of unit
            unit_part = unit_part[:2000] + "... [TRUNCATED FOR TESTING]"
            test_request['prompt'] = instruction_part + 'UNIT:' + unit_part

    # Make request
    try:
        print("Sending request to API...")
        response = requests.post(f"{API_BASE}/generate", json=test_request, timeout=10)

        if response.status_code != 200:
            print(f"ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return None

        result = response.json()
        session_id = result.get('session_id')

        if not session_id:
            print(f"ERROR: No session_id in response")
            print(f"Response: {result}")
            return None

        print(f"Session created: {session_id}")
        print(f"Status: {result.get('status')}")

        # Wait for completion
        max_wait = 120  # 2 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{API_BASE}/status/{session_id}")

            if status_response.status_code != 200:
                print(f"ERROR: Failed to get status")
                return None

            status_data = status_response.json()
            current_status = status_data.get('status')

            print(f"Status: {current_status}")

            if current_status == 'approved':
                print(f"\n[SUCCESS] Content approved!")

                # Get result
                result_response = requests.get(f"{API_BASE}/result/{session_id}")
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    print(f"\nGenerated content preview:")
                    content_str = json.dumps(result_data.get('approved_content', {}), indent=2)
                    print(content_str[:500] + "..." if len(content_str) > 500 else content_str)

                return {'status': 'approved', 'session_id': session_id, 'data': status_data}

            elif current_status in ['failed', 'stopped', 'preflight_rejected']:
                print(f"\n[FAILED] Status: {current_status}")
                print(f"Message: {status_data.get('message', 'No message')}")

                if 'verbose_log' in status_data:
                    print(f"\nVerbose log:")
                    for entry in status_data['verbose_log'][-5:]:  # Last 5 entries
                        print(f"  [{entry.get('timestamp', 'N/A')}] {entry.get('event', 'N/A')}")

                return {'status': current_status, 'session_id': session_id, 'data': status_data}

            time.sleep(3)

        print(f"\n[TIMEOUT] Session did not complete in {max_wait}s")
        return {'status': 'timeout', 'session_id': session_id}

    except requests.exceptions.Timeout:
        print(f"\n[TIMEOUT] REQUEST TIMEOUT")
        return None
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("REPRODUCING JSON SCHEMA ERROR")
    print("="*80)

    # Extract original request
    print("\nExtracting original request from database...")
    original_request = extract_original_request()

    if not original_request:
        print("ERROR: Could not extract original request")
        return

    print(f"[OK] Original request extracted")
    print(f"  Generator model: {original_request.get('generator_model')}")
    print(f"  JSON schema present: {'json_schema' in original_request}")
    print(f"  Prompt length: {len(original_request.get('prompt', ''))}")

    # Test cases
    test_cases = [
        {
            'name': 'Original: Grok-4-1-fast (reproduce error)',
            'model': 'grok-4-1-fast-non-reasoning',
            'description': 'Reproduce the original error'
        },
        {
            'name': 'Alternative: GPT-5-mini',
            'model': 'gpt-5-mini',
            'description': 'Test if GPT-5-mini handles it correctly'
        },
        {
            'name': 'Alternative: GPT-5.1',
            'model': 'gpt-5.1',
            'description': 'Test with full GPT-5.1'
        },
        {
            'name': 'Alternative: Claude Opus 4.1',
            'model': 'claude-opus-4.1',
            'description': 'Test with Claude Opus 4.1'
        }
    ]

    results = {}

    for test_case in test_cases:
        result = run_model_test(
            model_name=test_case['model'],
            request_data=original_request,
            test_name=test_case['name']
        )

        results[test_case['model']] = result

        # Wait between tests
        time.sleep(2)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_case in test_cases:
        model = test_case['model']
        result = results.get(model)

        if result:
            status = result.get('status', 'unknown')
            session_id = result.get('session_id', 'N/A')
            print(f"\n{test_case['name']}")
            print(f"  Model: {model}")
            print(f"  Status: {status}")
            print(f"  Session: {session_id}")
        else:
            print(f"\n{test_case['name']}")
            print(f"  Model: {model}")
            print(f"  Status: ERROR/TIMEOUT")

    # Test with Gran Sabio fallback
    print("\n" + "="*80)
    print("TESTING WITH GRAN SABIO FALLBACK")
    print("="*80)

    # Add Gran Sabio config to request
    gran_sabio_request = original_request.copy()
    gran_sabio_request['generator_model'] = 'grok-4-1-fast-non-reasoning'  # Original failing model
    gran_sabio_request['max_iterations'] = 2  # Reduce iterations for faster test
    gran_sabio_request['gran_sabio_config'] = {
        'enabled': True,
        'model': 'gpt-5-mini'  # Use cheap model for Gran Sabio
    }

    print("\nTesting Grok with Gran Sabio fallback (gpt-5-mini)...")
    gran_sabio_result = run_model_test(
        model_name='grok-4-1-fast-non-reasoning',
        request_data=gran_sabio_request,
        test_name='Grok + Gran Sabio (gpt-5-mini)'
    )

    if gran_sabio_result:
        print(f"\n[OK] Gran Sabio test completed: {gran_sabio_result.get('status')}")

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
