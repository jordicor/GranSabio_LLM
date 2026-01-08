"""
Detailed analysis of API calls to AI models for a specific project
Extracts information from payload_json to count actual model API calls
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

# Project ID to analyze
PROJECT_ID = "ed8e20631f714ec5b9017aaa268d21e1"

# Database path
DB_PATH = Path(__file__).parent.parent / "debugger_history.db"

def analyze_api_calls_detailed():
    """Detailed analysis of actual API calls to AI models"""

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 80)
    print(f"DETAILED API ANALYSIS FOR PROJECT: {PROJECT_ID}")
    print("=" * 80)

    # Get session IDs for this project
    cursor.execute("""
        SELECT session_id, created_at, status
        FROM sessions
        WHERE project_id = ?
        ORDER BY created_at
    """, (PROJECT_ID,))
    sessions = cursor.fetchall()

    if not sessions:
        print("\n[WARNING] No sessions found for this project_id")
        conn.close()
        return

    session_ids = [s[0] for s in sessions]
    first_session_time = sessions[0][1]
    last_session_time = sessions[-1][1]

    print(f"\nTotal Sessions: {len(sessions)}")
    print(f"First Session: {first_session_time}")
    print(f"Last Session: {last_session_time}")

    # Session status breakdown
    status_counts = Counter(s[2] for s in sessions)
    print("\nSession Status Breakdown:")
    for status, count in status_counts.most_common():
        print(f"  - {status}: {count}")

    # Now analyze events with payloads that contain API call information
    placeholders = ','.join('?' * len(session_ids))

    # Get all events that represent actual AI model calls
    # Key event types: generator_output, qa_evaluation_completed, consensus_completed
    cursor.execute(f"""
        SELECT
            session_id,
            event_type,
            payload_json,
            created_at
        FROM session_events
        WHERE session_id IN ({placeholders})
        AND event_type IN (
            'generator_output',
            'qa_evaluation_completed',
            'consensus_completed',
            'gran_sabio_regeneration_completed'
        )
        ORDER BY created_at
    """, session_ids)

    events = cursor.fetchall()

    print(f"\n\n{'=' * 80}")
    print("API CALLS TO AI MODELS")
    print('=' * 80)

    # Counters for analysis
    model_calls = Counter()
    provider_calls = Counter()
    total_tokens = defaultdict(lambda: {'input': 0, 'output': 0})
    first_api_call = None
    last_api_call = None
    total_api_calls = 0

    for session_id, event_type, payload_str, created_at in events:
        try:
            payload = json.loads(payload_str) if payload_str else {}

            # Extract model information based on event type
            if event_type == 'generator_output':
                model = payload.get('model', 'unknown')
                usage = payload.get('usage', {})

                model_calls[model] += 1
                total_api_calls += 1

                # Extract provider from model name
                if 'gpt' in model.lower() or 'o1' in model.lower() or 'o3' in model.lower():
                    provider_calls['OpenAI'] += 1
                elif 'claude' in model.lower():
                    provider_calls['Anthropic'] += 1
                elif 'gemini' in model.lower():
                    provider_calls['Google'] += 1
                elif 'grok' in model.lower():
                    provider_calls['xAI'] += 1

                # Track tokens
                if usage:
                    input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                    total_tokens[model]['input'] += input_tokens
                    total_tokens[model]['output'] += output_tokens

            elif event_type == 'qa_evaluation_completed':
                # QA can have multiple evaluator results
                evaluator_results = payload.get('evaluator_results', [])
                for result in evaluator_results:
                    model = result.get('evaluator_model', 'unknown')
                    usage = result.get('usage', {})

                    model_calls[model] += 1
                    total_api_calls += 1

                    if 'gpt' in model.lower() or 'o1' in model.lower():
                        provider_calls['OpenAI'] += 1
                    elif 'claude' in model.lower():
                        provider_calls['Anthropic'] += 1
                    elif 'gemini' in model.lower():
                        provider_calls['Google'] += 1
                    elif 'grok' in model.lower():
                        provider_calls['xAI'] += 1

                    if usage:
                        input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                        output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                        total_tokens[model]['input'] += input_tokens
                        total_tokens[model]['output'] += output_tokens

            elif event_type == 'consensus_completed':
                # Consensus has multiple evaluator results
                evaluator_results = payload.get('evaluator_results', [])
                for result in evaluator_results:
                    model = result.get('evaluator_model', 'unknown')
                    usage = result.get('usage', {})

                    model_calls[model] += 1
                    total_api_calls += 1

                    if 'gpt' in model.lower() or 'o1' in model.lower():
                        provider_calls['OpenAI'] += 1
                    elif 'claude' in model.lower():
                        provider_calls['Anthropic'] += 1
                    elif 'gemini' in model.lower():
                        provider_calls['Google'] += 1
                    elif 'grok' in model.lower():
                        provider_calls['xAI'] += 1

                    if usage:
                        input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                        output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                        total_tokens[model]['input'] += input_tokens
                        total_tokens[model]['output'] += output_tokens

            elif event_type == 'gran_sabio_regeneration_completed':
                model = payload.get('model', 'unknown')
                usage = payload.get('usage', {})

                model_calls[model] += 1
                total_api_calls += 1

                if 'gpt' in model.lower() or 'o1' in model.lower():
                    provider_calls['OpenAI'] += 1
                elif 'claude' in model.lower():
                    provider_calls['Anthropic'] += 1
                elif 'gemini' in model.lower():
                    provider_calls['Google'] += 1
                elif 'grok' in model.lower():
                    provider_calls['xAI'] += 1

                if usage:
                    input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
                    total_tokens[model]['input'] += input_tokens
                    total_tokens[model]['output'] += output_tokens

            # Track first and last API call times
            if first_api_call is None:
                first_api_call = created_at
            last_api_call = created_at

        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Warning: Error processing event: {e}")

    print(f"\nTOTAL API CALLS TO AI MODELS: {total_api_calls}")
    print(f"\nFirst API Call: {first_api_call}")
    print(f"Last API Call:  {last_api_call}")

    if first_api_call and last_api_call:
        try:
            start = datetime.fromisoformat(first_api_call.replace('Z', '+00:00'))
            end = datetime.fromisoformat(last_api_call.replace('Z', '+00:00'))
            duration = end - start
            print(f"\nAPI Activity Duration: {duration}")
            print(f"  - Hours: {duration.total_seconds() / 3600:.2f}")
            print(f"  - Minutes: {duration.total_seconds() / 60:.2f}")
        except:
            pass

    print("\n\nAPI CALLS BY PROVIDER:")
    for provider, count in provider_calls.most_common():
        print(f"  - {provider}: {count} calls")

    print("\n\nAPI CALLS BY MODEL:")
    for model, count in model_calls.most_common():
        print(f"  - {model}: {count} calls")

    print("\n\nTOKEN USAGE BY MODEL:")
    total_input_tokens = 0
    total_output_tokens = 0

    for model, tokens in sorted(total_tokens.items(), key=lambda x: x[1]['input'] + x[1]['output'], reverse=True):
        input_t = tokens['input']
        output_t = tokens['output']
        total_t = input_t + output_t

        total_input_tokens += input_t
        total_output_tokens += output_t

        print(f"\n  {model}:")
        print(f"    - Input tokens:  {input_t:,}")
        print(f"    - Output tokens: {output_t:,}")
        print(f"    - Total tokens:  {total_t:,}")

    print(f"\n\nTOTAL TOKENS ACROSS ALL MODELS:")
    print(f"  - Input tokens:  {total_input_tokens:,}")
    print(f"  - Output tokens: {total_output_tokens:,}")
    print(f"  - Total tokens:  {total_input_tokens + total_output_tokens:,}")

    print("\n" + "=" * 80)
    conn.close()

if __name__ == "__main__":
    analyze_api_calls_detailed()
