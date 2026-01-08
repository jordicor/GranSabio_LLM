"""
Script to analyze API calls from debugger_history.db for a specific project
Provides statistics on total calls, start time, and end time
"""
import sqlite3
from datetime import datetime
from pathlib import Path

# Project ID to analyze
PROJECT_ID = "ed8e20631f714ec5b9017aaa268d21e1"

# Database path
DB_PATH = Path(__file__).parent.parent / "debugger_history.db"

def analyze_project_calls():
    """Analyze API calls for the specified project"""

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # First, let's explore the database structure
    print("=" * 80)
    print("DATABASE STRUCTURE")
    print("=" * 80)

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\nTables found: {[t[0] for t in tables]}\n")

    # Get schema for each table
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")

    print("\n" + "=" * 80)
    print(f"ANALYSIS FOR PROJECT: {PROJECT_ID}")
    print("=" * 80)

    # Query sessions table for the project
    print("\nQuerying sessions table...")
    try:
        cursor.execute("""
            SELECT
                COUNT(*) as total_sessions,
                MIN(created_at) as first_session,
                MAX(created_at) as last_session
            FROM sessions
            WHERE project_id = ?
        """, (PROJECT_ID,))

        result = cursor.fetchone()
        if result and result[0] > 0:
            total_sessions, first_session, last_session = result

            print(f"\n[SUCCESS] SESSIONS FOUND FOR PROJECT")
            print(f"\nTotal Sessions: {total_sessions}")
            print(f"\nFirst Session Created: {first_session}")
            print(f"Last Session Created:  {last_session}")

            # Calculate duration
            if first_session and last_session:
                try:
                    start = datetime.fromisoformat(first_session.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(last_session.replace('Z', '+00:00'))
                    duration = end - start
                    print(f"\nTotal Duration: {duration}")
                    print(f"  - Days: {duration.days}")
                    print(f"  - Hours: {duration.total_seconds() / 3600:.2f}")
                    print(f"  - Minutes: {duration.total_seconds() / 60:.2f}")
                except Exception as e:
                    print(f"\nCould not parse duration: {e}")

            # Get session IDs for this project
            cursor.execute("""
                SELECT session_id FROM sessions WHERE project_id = ?
            """, (PROJECT_ID,))
            session_ids = [row[0] for row in cursor.fetchall()]

            # Now count API calls from session_events for these sessions
            print("\n\nQuerying session_events for API calls...")
            placeholders = ','.join('?' * len(session_ids))
            cursor.execute(f"""
                SELECT
                    COUNT(*) as total_events,
                    MIN(created_at) as first_event,
                    MAX(created_at) as last_event
                FROM session_events
                WHERE session_id IN ({placeholders})
            """, session_ids)

            events_result = cursor.fetchone()
            if events_result and events_result[0] > 0:
                total_events, first_event, last_event = events_result

                print(f"\n[SUCCESS] EVENTS FOUND")
                print(f"\nTotal Events: {total_events}")
                print(f"\nFirst Event: {first_event}")
                print(f"Last Event:  {last_event}")

                # Event types breakdown
                cursor.execute(f"""
                    SELECT event_type, COUNT(*) as count
                    FROM session_events
                    WHERE session_id IN ({placeholders})
                    GROUP BY event_type
                    ORDER BY count DESC
                """, session_ids)

                event_types = cursor.fetchall()
                print("\n\nEvent Types Breakdown:")
                for event_type, count in event_types:
                    print(f"  - {event_type}: {count}")
            else:
                print("\n[WARNING] No events found for these sessions")

        else:
            print("\n[WARNING] No sessions found for this project_id")
            print("Let's check all project IDs in the database...")

            cursor.execute("""
                SELECT DISTINCT project_id, COUNT(*) as session_count
                FROM sessions
                WHERE project_id IS NOT NULL
                GROUP BY project_id
                ORDER BY session_count DESC
            """)
            all_projects = cursor.fetchall()

            if all_projects:
                print("\nAll project IDs in database:")
                for proj_id, count in all_projects:
                    indicator = " <-- TARGET" if proj_id == PROJECT_ID else ""
                    print(f"  - {proj_id}: {count} sessions{indicator}")
            else:
                print("\n[INFO] No projects found in database")

    except Exception as e:
        print(f"\n[ERROR] Query failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    conn.close()

if __name__ == "__main__":
    analyze_project_calls()
