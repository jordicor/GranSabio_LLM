#!/usr/bin/env python3
"""Script to create modular documentation structure."""
import os
import re

BASE_DIR = "S:/01.Coding/BioAI_Unified/templates"
DOCS_DIR = os.path.join(BASE_DIR, "docs")
BACKUP_FILE = os.path.join(BASE_DIR, "api_docs_backup.html")

os.makedirs(DOCS_DIR, exist_ok=True)

# Read the original file
with open(BACKUP_FILE, "r", encoding="utf-8") as f:
    original_content = f.read()

# Extract sections by their IDs
def extract_section(content, start_id, end_ids):
    """Extract a section from start_id until any of end_ids."""
    pattern = rf'(<section id="{start_id}".*?)(?=<section id="(?:{"|".join(end_ids)})"|\s*</main>)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# Define section mappings
section_mappings = {
    "getting-started.html": {
        "start": "overview",
        "end": ["generate"]
    },
    "endpoints.html": {
        "start": "generate", 
        "end": ["analysis-lexical-diversity"]
    },
    "analysis.html": {
        "start": "analysis-lexical-diversity",
        "end": ["project-handshake"]
    },
    "advanced-misc.html": {
        "start": "project-handshake",
        "end": ["qa-layers"]
    },
    "advanced-qa.html": {
        "start": "qa-layers",
        "end": ["word-count-enforcement"]
    },
    "advanced-guards.html": {
        "start": "word-count-enforcement",
        "end": ["json-schema-structured-outputs"]
    },
    "advanced-json.html": {
        "start": "json-schema-structured-outputs",
        "end": ["error-handling"]
    },
    "reference.html": {
        "start": "error-handling",
        "end": ["ENDMARKER"]
    }
}

# Extract and save each section
for filename, mapping in section_mappings.items():
    start_pattern = rf'<section id="{mapping["start"]}"'
    end_patterns = [rf'<section id="{eid}"' for eid in mapping["end"]]
    
    # Find start position
    start_match = re.search(start_pattern, original_content)
    if not start_match:
        print(f"Could not find start section: {mapping['start']}")
        continue
    
    start_pos = start_match.start()
    
    # Find end position
    end_pos = len(original_content)
    for end_pattern in end_patterns:
        end_match = re.search(end_pattern, original_content[start_pos + 1:])
        if end_match:
            end_pos = min(end_pos, start_pos + 1 + end_match.start())
    
    # Also check for </main> as end marker
    main_end = original_content.find("</main>", start_pos)
    if main_end != -1:
        end_pos = min(end_pos, main_end)
    
    section_content = original_content[start_pos:end_pos].strip()
    
    filepath = os.path.join(DOCS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(section_content)
    
    size_kb = len(section_content) / 1024
    print(f"Created {filename}: {size_kb:.1f} KB")

print("\nAll section files created!")
