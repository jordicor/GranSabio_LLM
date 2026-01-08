"""
Extract full details from problematic session to analyze JSON schema issues
Session: 0097ba67-ce49-462d-ae26-e8741e0bcce9
"""
import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "debugger_history.db"
SESSION_ID = "0097ba67-ce49-462d-ae26-e8741e0bcce9"

def extract_json_schema_from_prompt(prompt_text):
    """Extract JSON schema from prompt if present"""
    if "JSON_SCHEMA:" in prompt_text:
        parts = prompt_text.split("JSON_SCHEMA:")
        if len(parts) > 1:
            schema_part = parts[1].split("\n\n")[0]
            try:
                return json.loads(schema_part.strip())
            except:
                return schema_part.strip()
    return None

def analyze_json_error(content, error_msg):
    """Analyze JSON parsing error"""
    if "Expecting ',' delimiter" in error_msg:
        # Extract position
        import re
        match = re.search(r'column (\d+) \(char (\d+)\)', error_msg)
        if match:
            col = int(match.group(1))
            char_pos = int(match.group(2))

            # Show context around error
            start = max(0, char_pos - 150)
            end = min(len(content), char_pos + 150)
            context = content[start:end]

            print(f"\n>>> ERROR POSITION: Column {col}, Character {char_pos}")
            print(f">>> CONTEXT (showing ±150 chars around error):")
            print("=" * 80)
            print(context)
            print("=" * 80)
            print(f">>> Character at error position: {repr(content[char_pos] if char_pos < len(content) else 'EOF')}")

            # Try to identify the issue
            if char_pos > 0 and char_pos < len(content):
                before = content[max(0, char_pos - 50):char_pos]
                after = content[char_pos:min(len(content), char_pos + 50)]
                print(f"\n>>> BEFORE ERROR: ...{repr(before)}")
                print(f">>> AFTER ERROR: {repr(after)}...")

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get session request to extract schema
    cursor.execute("SELECT request_json FROM sessions WHERE session_id = ?", (SESSION_ID,))
    session = cursor.fetchone()

    if not session:
        print(f"Session {SESSION_ID} not found!")
        return

    request_data = json.loads(session['request_json'])

    print("=" * 80)
    print("EXTRACTING JSON SCHEMA FROM REQUEST")
    print("=" * 80)

    # Check if json_schema is in request
    if 'json_schema' in request_data:
        schema = request_data['json_schema']
        print("\n>>> JSON SCHEMA FOUND IN REQUEST:")
        print(json.dumps(schema, indent=2))

        # Extract enum definitions
        print("\n" + "=" * 80)
        print("ENUM DEFINITIONS IN SCHEMA")
        print("=" * 80)

        def find_enums(obj, path=""):
            """Recursively find all enum definitions"""
            if isinstance(obj, dict):
                if 'enum' in obj:
                    print(f"\n>>> {path}")
                    print(f"    enum: {obj['enum']}")
                    if 'description' in obj:
                        print(f"    description: {obj['description']}")

                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    find_enums(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_enums(item, f"{path}[{i}]")

        find_enums(schema)
    else:
        print("\n>>> NO json_schema FIELD IN REQUEST")

    # Get all generator outputs and their validation errors
    print("\n" + "=" * 80)
    print("ANALYZING ALL ITERATIONS")
    print("=" * 80)

    cursor.execute("""
        SELECT event_order, event_type, payload_json
        FROM session_events
        WHERE session_id = ?
        AND event_type IN ('generator_output', 'json_guard_result')
        ORDER BY event_order
    """, (SESSION_ID,))

    events = cursor.fetchall()
    current_iteration = None
    current_output = None

    for event in events:
        payload = json.loads(event['payload_json'])

        if event['event_type'] == 'generator_output':
            iteration = payload['data']['iteration']
            content = payload['data']['content']
            model = payload['data']['model']

            print(f"\n{'='*80}")
            print(f"ITERATION {iteration} - MODEL: {model}")
            print(f"{'='*80}")
            print("\n>>> GENERATED JSON (FULL):")
            print(content)

            # Try to parse and pretty print
            try:
                parsed = json.loads(content)
                print("\n>>> PARSED SUCCESSFULLY (Pretty Print):")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"\n>>> FAILED TO PARSE: {e}")

            current_iteration = iteration
            current_output = content

        elif event['event_type'] == 'json_guard_result':
            iteration = payload['data']['iteration']
            valid = payload['data']['valid']
            errors = payload['data']['errors']

            print(f"\n{'='*80}")
            print(f"VALIDATION RESULT - ITERATION {iteration}")
            print(f"{'='*80}")
            print(f">>> VALID: {valid}")
            print(f">>> ERRORS:")
            for error in errors:
                print(f"    - {error}")

                # If it's a parsing error, analyze it
                if current_output and "JSON parse error" in error:
                    analyze_json_error(current_output, error)

    conn.close()

if __name__ == "__main__":
    main()
