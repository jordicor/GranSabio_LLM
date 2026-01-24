"""
Test script to analyze minority deal-breaker handling in the QA system.

This script:
1. Analyzes existing sessions for minority deal-breaker scenarios
2. Verifies the logic for detecting minority vs majority deal-breakers
3. Checks if Gran Sabio escalation is triggered correctly
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "debugger_history.db"


def analyze_deal_breaker_logic():
    """
    Demonstrate the logic for detecting minority vs majority deal-breakers.

    With 3 models:
    - majority_threshold = 3 / 2 = 1.5
    - is_minority = 0 < deal_breaker_count < 1.5 -> only count=1 is minority
    - immediate_stop = deal_breaker_count > 1.5 -> count >= 2 triggers stop
    """
    print("=== Deal Breaker Logic Analysis ===")
    print()

    for total_models in [2, 3, 4]:
        print(f"With {total_models} models:")
        majority_threshold = total_models / 2
        print(f"  majority_threshold = {majority_threshold}")

        for db_count in range(total_models + 1):
            is_minority = total_models > 0 and 0 < db_count < (total_models / 2)
            is_tie = total_models > 0 and total_models % 2 == 0 and db_count * 2 == total_models
            immediate_stop = db_count > majority_threshold  # QA engine logic

            classification = []
            if db_count == 0:
                classification.append("NO_DEAL_BREAKER")
            elif is_minority:
                classification.append("MINORITY -> should escalate to GranSabio inline")
            elif is_tie:
                classification.append("TIE -> should escalate to GranSabio inline")
            elif immediate_stop:
                classification.append("MAJORITY -> force iteration")
            else:
                classification.append("UNEXPECTED")

            print(f"  {db_count}/{total_models}: {', '.join(classification)}")
        print()


def analyze_session_flow(session_id: str):
    """Analyze the QA flow for a specific session to trace deal-breaker handling."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all events
    cursor.execute('''
        SELECT event_order, event_type, payload_json
        FROM session_events
        WHERE session_id = ?
        ORDER BY event_order
    ''', (session_id,))

    events = list(cursor.fetchall())
    print(f"\n=== Session {session_id[:20]}... Analysis ===")
    print(f"Total events: {len(events)}")

    # Find iterations with deal-breakers
    qa_events = [(o, p) for o, t, p in events if t == 'qa_evaluation_completed']

    print(f"\nQA Evaluations: {len(qa_events)}")

    minority_found = False
    gran_sabio_called = False

    for order, pjson in qa_events:
        payload = json.loads(pjson)
        data = payload.get('data', {})
        iteration = data.get('iteration', '?')
        qa_results = data.get('qa_results', {})

        for layer_name, model_results in qa_results.items():
            if not isinstance(model_results, dict):
                continue

            db_models = []
            non_db_models = []

            for model, eval_data in model_results.items():
                if isinstance(eval_data, dict):
                    is_db = eval_data.get('deal_breaker', False)
                else:
                    is_db = getattr(eval_data, 'deal_breaker', False)

                if is_db:
                    db_models.append(model)
                else:
                    non_db_models.append(model)

            total = len(db_models) + len(non_db_models)
            db_count = len(db_models)

            if db_count > 0:
                is_minority = total > 0 and 0 < db_count < (total / 2)
                is_tie = total > 0 and total % 2 == 0 and db_count * 2 == total

                type_str = "MINORITY" if is_minority else ("TIE" if is_tie else "MAJORITY")

                if is_minority or is_tie:
                    minority_found = True
                    print(f"\nIteration {iteration}, Layer {layer_name}:")
                    print(f"  Deal breakers: {db_count}/{total} -> {type_str}")
                    print(f"  DB models: {db_models}")
                    print(f"  Non-DB models: {non_db_models}")

    # Check for Gran Sabio events
    for order, etype, pjson in events:
        if 'gran_sabio' in etype.lower():
            gran_sabio_called = True
            print(f"\nGran Sabio event found: [{order}] {etype}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Minority/Tie deal-breakers found: {minority_found}")
    print(f"Gran Sabio explicitly called: {gran_sabio_called}")

    if minority_found and not gran_sabio_called:
        print("\n*** POTENTIAL ISSUE: Minority deal-breaker detected but Gran Sabio was NOT called! ***")
        print("This could indicate a bug in the inline escalation logic.")

    conn.close()
    return minority_found, gran_sabio_called


def find_all_minority_sessions():
    """Find all sessions with minority deal-breakers."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all sessions
    cursor.execute('''
        SELECT s.session_id, s.status, s.created_at
        FROM sessions s
        ORDER BY s.created_at DESC
    ''')

    sessions = cursor.fetchall()
    print(f"\n=== Scanning {len(sessions)} sessions for minority deal-breakers ===")

    minority_sessions = []

    for sid, status, created in sessions:
        cursor.execute('''
            SELECT payload_json
            FROM session_events
            WHERE session_id = ? AND event_type = 'qa_evaluation_completed'
        ''', (sid,))

        for (pjson,) in cursor.fetchall():
            payload = json.loads(pjson)
            data = payload.get('data', {})
            qa_results = data.get('qa_results', {})

            for layer_name, model_results in qa_results.items():
                if not isinstance(model_results, dict):
                    continue

                total = len(model_results)
                db_count = sum(
                    1 for m, e in model_results.items()
                    if (isinstance(e, dict) and e.get('deal_breaker')) or
                       (hasattr(e, 'deal_breaker') and e.deal_breaker)
                )

                is_minority = total > 0 and 0 < db_count < (total / 2)

                if is_minority:
                    minority_sessions.append({
                        'session_id': sid,
                        'status': status,
                        'created': created,
                        'layer': layer_name,
                        'db_count': db_count,
                        'total': total
                    })
                    break
            else:
                continue
            break

    print(f"\nFound {len(minority_sessions)} sessions with minority deal-breakers:\n")

    for ms in minority_sessions[:10]:  # Limit to 10
        print(f"  {ms['session_id'][:20]}... ({ms['status']})")
        print(f"    Layer: {ms['layer']}, DB: {ms['db_count']}/{ms['total']}")

    conn.close()
    return minority_sessions


def main():
    print("=" * 70)
    print("MINORITY DEAL-BREAKER ANALYSIS")
    print("=" * 70)

    # First, explain the logic
    analyze_deal_breaker_logic()

    # Find sessions with minority deal-breakers
    minority_sessions = find_all_minority_sessions()

    if minority_sessions:
        # Analyze the first few in detail
        print("\n" + "=" * 70)
        print("DETAILED ANALYSIS OF SESSIONS WITH MINORITY DEAL-BREAKERS")
        print("=" * 70)

        for ms in minority_sessions[:3]:
            analyze_session_flow(ms['session_id'])


if __name__ == "__main__":
    main()
