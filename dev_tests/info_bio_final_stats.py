"""
Final comprehensive analysis of API calls and usage for a specific project
Combines session data, events, and usage_json for complete statistics
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

def format_datetime(dt_str):
    """Format datetime string for display"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str

def analyze_final_stats():
    """Complete analysis with all available data"""

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=" * 80)
    print(f"COMPREHENSIVE API ANALYSIS - PROJECT: {PROJECT_ID}")
    print("=" * 80)

    # Get all sessions with their data
    cursor.execute("""
        SELECT
            session_id,
            created_at,
            updated_at,
            status,
            usage_json
        FROM sessions
        WHERE project_id = ?
        ORDER BY created_at
    """, (PROJECT_ID,))

    sessions = cursor.fetchall()

    if not sessions:
        print("\n[WARNING] No sessions found for this project_id")
        conn.close()
        return

    total_sessions = len(sessions)
    first_session = sessions[0][1]
    last_session = sessions[-1][1]

    print(f"\n{'=' * 80}")
    print("SESSION OVERVIEW")
    print('=' * 80)
    print(f"\nTotal Sessions: {total_sessions}")
    print(f"First Session:  {format_datetime(first_session)}")
    print(f"Last Session:   {format_datetime(last_session)}")

    # Calculate duration
    try:
        start = datetime.fromisoformat(first_session.replace('Z', '+00:00'))
        end = datetime.fromisoformat(last_session.replace('Z', '+00:00'))
        duration = end - start
        print(f"\nProject Duration: {duration}")
        print(f"  - Days:    {duration.days}")
        print(f"  - Hours:   {duration.total_seconds() / 3600:.2f}")
        print(f"  - Minutes: {duration.total_seconds() / 60:.2f}")
    except:
        pass

    # Session status breakdown
    status_counts = Counter(s[3] for s in sessions)
    print("\nSession Status Breakdown:")
    for status, count in status_counts.most_common():
        percentage = (count / total_sessions) * 100
        print(f"  - {status:15s}: {count:4d} ({percentage:5.1f}%)")

    # Aggregate usage data from usage_json
    print(f"\n{'=' * 80}")
    print("TOKEN USAGE & COST ANALYSIS")
    print('=' * 80)

    total_input = 0
    total_output = 0
    total_cost = 0.0
    total_reasoning = 0

    phase_totals = defaultdict(lambda: {
        'input': 0,
        'output': 0,
        'cost': 0.0,
        'reasoning': 0
    })

    sessions_with_usage = 0
    sessions_without_usage = 0

    for session_id, created_at, updated_at, status, usage_json in sessions:
        if usage_json and usage_json.strip():
            try:
                usage = json.loads(usage_json)
                sessions_with_usage += 1

                # Grand totals
                if 'grand_totals' in usage:
                    gt = usage['grand_totals']
                    total_input += gt.get('input_tokens', 0)
                    total_output += gt.get('output_tokens', 0)
                    total_cost += gt.get('cost', 0.0)
                    reasoning = gt.get('reasoning_tokens', 0)
                    if reasoning:
                        total_reasoning += reasoning

                # Phase breakdown
                if 'phases' in usage:
                    for phase, data in usage['phases'].items():
                        phase_totals[phase]['input'] += data.get('input_tokens', 0)
                        phase_totals[phase]['output'] += data.get('output_tokens', 0)
                        phase_totals[phase]['cost'] += data.get('cost', 0.0)
                        reasoning = data.get('reasoning_tokens', 0)
                        if reasoning:
                            phase_totals[phase]['reasoning'] += reasoning

            except json.JSONDecodeError:
                sessions_without_usage += 1
        else:
            sessions_without_usage += 1

    print(f"\nSessions with usage data: {sessions_with_usage}")
    print(f"Sessions without usage data: {sessions_without_usage}")

    print(f"\n{'-' * 80}")
    print("GRAND TOTALS")
    print('-' * 80)
    print(f"Input Tokens:      {total_input:>15,}")
    print(f"Output Tokens:     {total_output:>15,}")
    if total_reasoning > 0:
        print(f"Reasoning Tokens:  {total_reasoning:>15,}")
    print(f"Total Tokens:      {(total_input + total_output):>15,}")
    print(f"\nTotal Cost:        ${total_cost:>14.4f} USD")

    if sessions_with_usage > 0:
        print(f"\nAverage per session:")
        print(f"  - Tokens: {(total_input + total_output) / sessions_with_usage:,.0f}")
        print(f"  - Cost:   ${total_cost / sessions_with_usage:.4f}")

    print(f"\n{'-' * 80}")
    print("BREAKDOWN BY PHASE")
    print('-' * 80)

    for phase in sorted(phase_totals.keys()):
        data = phase_totals[phase]
        phase_total = data['input'] + data['output']
        phase_cost = data['cost']

        print(f"\n{phase.upper()}:")
        print(f"  Input Tokens:     {data['input']:>12,}")
        print(f"  Output Tokens:    {data['output']:>12,}")
        if data['reasoning'] > 0:
            print(f"  Reasoning Tokens: {data['reasoning']:>12,}")
        print(f"  Total Tokens:     {phase_total:>12,}")
        print(f"  Cost:             ${phase_cost:>11.4f}")

        if total_input + total_output > 0:
            percentage = (phase_total / (total_input + total_output)) * 100
            print(f"  % of total:       {percentage:>11.1f}%")

    # Event analysis
    print(f"\n{'=' * 80}")
    print("API CALL EVENTS ANALYSIS")
    print('=' * 80)

    session_ids = [s[0] for s in sessions]
    placeholders = ','.join('?' * len(session_ids))

    # Count events by type
    cursor.execute(f"""
        SELECT event_type, COUNT(*) as count
        FROM session_events
        WHERE session_id IN ({placeholders})
        GROUP BY event_type
        ORDER BY count DESC
    """, session_ids)

    events = cursor.fetchall()

    print("\nEvent Type Distribution:")
    total_events = sum(count for _, count in events)
    for event_type, count in events:
        percentage = (count / total_events) * 100
        print(f"  {event_type:40s}: {count:5d} ({percentage:5.1f}%)")

    print(f"\nTotal Events: {total_events:,}")

    # Extract generator model usage from events
    print(f"\n{'-' * 80}")
    print("GENERATOR MODEL USAGE")
    print('-' * 80)

    cursor.execute(f"""
        SELECT payload_json
        FROM session_events
        WHERE session_id IN ({placeholders})
        AND event_type = 'generator_output'
    """, session_ids)

    model_usage = Counter()
    for row in cursor.fetchall():
        if row[0]:
            try:
                payload = json.loads(row[0])
                model = payload.get('data', {}).get('model', 'unknown')
                model_usage[model] += 1
            except:
                pass

    if model_usage:
        print("\nGenerator Model Call Count:")
        for model, count in model_usage.most_common():
            print(f"  {model:30s}: {count:4d} calls")
    else:
        print("\n(No generator output events with model information found)")

    # First and last API call times
    cursor.execute(f"""
        SELECT MIN(created_at), MAX(created_at)
        FROM session_events
        WHERE session_id IN ({placeholders})
        AND event_type IN ('generator_output', 'qa_evaluation_completed', 'consensus_completed')
    """, session_ids)

    result = cursor.fetchone()
    if result[0] and result[1]:
        first_api = result[0]
        last_api = result[1]

        print(f"\n{'-' * 80}")
        print(f"First API Call: {format_datetime(first_api)}")
        print(f"Last API Call:  {format_datetime(last_api)}")

        try:
            start = datetime.fromisoformat(first_api.replace('Z', '+00:00'))
            end = datetime.fromisoformat(last_api.replace('Z', '+00:00'))
            duration = end - start
            print(f"\nAPI Activity Duration: {duration}")
            print(f"  - Hours:   {duration.total_seconds() / 3600:.2f}")
            print(f"  - Minutes: {duration.total_seconds() / 60:.2f}")
        except:
            pass

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    conn.close()

if __name__ == "__main__":
    analyze_final_stats()
