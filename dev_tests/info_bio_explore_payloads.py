"""
Explore payload structure to understand how API call data is stored
"""
import sqlite3
import json
from pathlib import Path

# Project ID to analyze
PROJECT_ID = "ed8e20631f714ec5b9017aaa268d21e1"

# Database path
DB_PATH = Path(__file__).parent.parent / "debugger_history.db"

def explore_payloads():
    """Explore the structure of event payloads"""

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get session IDs for this project
    cursor.execute("""
        SELECT session_id FROM sessions WHERE project_id = ?
    """, (PROJECT_ID,))
    session_ids = [row[0] for row in cursor.fetchall()]

    if not session_ids:
        print("No sessions found")
        conn.close()
        return

    placeholders = ','.join('?' * len(session_ids))

    # Get sample events of each type
    cursor.execute(f"""
        SELECT DISTINCT event_type FROM session_events
        WHERE session_id IN ({placeholders})
        ORDER BY event_type
    """, session_ids)

    event_types = [row[0] for row in cursor.fetchall()]

    print("=" * 80)
    print("EXPLORING PAYLOAD STRUCTURES")
    print("=" * 80)

    for event_type in event_types:
        print(f"\n\n{'=' * 80}")
        print(f"EVENT TYPE: {event_type}")
        print('=' * 80)

        cursor.execute(f"""
            SELECT payload_json
            FROM session_events
            WHERE session_id IN ({placeholders})
            AND event_type = ?
            LIMIT 1
        """, session_ids + [event_type])

        result = cursor.fetchone()
        if result and result[0]:
            try:
                payload = json.loads(result[0])
                print("\nPAYLOAD STRUCTURE:")
                # Use ensure_ascii=True to avoid Unicode issues on Windows
                print(json.dumps(payload, indent=2, ensure_ascii=True)[:2000])  # First 2000 chars

                # Print keys at root level
                if isinstance(payload, dict):
                    print(f"\n\nROOT KEYS: {list(payload.keys())}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
        else:
            print("(No payload)")

    # Also check if sessions table has usage_json
    print(f"\n\n{'=' * 80}")
    print("CHECKING SESSIONS TABLE FOR USAGE DATA")
    print('=' * 80)

    cursor.execute(f"""
        SELECT session_id, status, usage_json
        FROM sessions
        WHERE project_id = ?
        AND usage_json IS NOT NULL
        AND usage_json != ''
        LIMIT 3
    """, (PROJECT_ID,))

    sessions_with_usage = cursor.fetchall()

    if sessions_with_usage:
        for session_id, status, usage_json in sessions_with_usage:
            print(f"\n\nSession: {session_id}")
            print(f"Status: {status}")
            print("\nUsage JSON:")
            try:
                usage = json.loads(usage_json)
                print(json.dumps(usage, indent=2, ensure_ascii=True)[:2000])
            except:
                print(usage_json[:2000])
    else:
        print("\nNo sessions with usage_json found")

    print("\n" + "=" * 80)
    conn.close()

if __name__ == "__main__":
    explore_payloads()
