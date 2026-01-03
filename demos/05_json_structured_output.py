"""
Demo 05: JSON Structured Output
===============================

This demo shows how to use JSON Schema for guaranteed structured outputs.
The API enforces exact format compliance at generation time, eliminating
parsing errors and retry loops.

Features demonstrated:
- json_schema for 100% format guarantee
- Multi-provider support (GPT, Claude, Gemini, Grok)
- Information extraction use cases
- Complex nested schemas

This is ideal for:
- API integrations requiring exact formats
- Data extraction from unstructured text
- Automated pipelines needing reliable JSON
- Building structured knowledge bases

Usage:
    python demos/05_json_structured_output.py

    # With custom text to extract from:
    python demos/05_json_structured_output.py --text "John Doe, 35, software engineer..."
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import AsyncGranSabioClient
from demos.common import run_demo, print_header, print_json_content, colorize, safe_print


# Sample unstructured text for extraction
SAMPLE_RESUME_TEXT = """
John Smith - Senior Software Engineer

Contact: john.smith@email.com | LinkedIn: linkedin.com/in/johnsmith
Location: San Francisco, CA

SUMMARY
Experienced software engineer with 8 years of expertise in building scalable
web applications. Passionate about clean code and mentoring junior developers.

SKILLS
Programming: Python, JavaScript, TypeScript, Go, Rust
Frameworks: React, Node.js, Django, FastAPI
Cloud: AWS (certified), GCP, Docker, Kubernetes
Databases: PostgreSQL, MongoDB, Redis

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2020 - Present
- Led development of microservices architecture serving 10M+ users
- Reduced API latency by 60% through optimization
- Mentored team of 5 junior developers

Software Engineer | StartupXYZ | 2017 - 2020
- Built real-time data pipeline processing 1M events/day
- Implemented CI/CD reducing deployment time by 80%

Junior Developer | WebAgency | 2015 - 2017
- Developed responsive websites for 50+ clients
- Learned modern JavaScript frameworks

EDUCATION
BS Computer Science | Stanford University | 2015
GPA: 3.8/4.0

LANGUAGES
English (native), Spanish (conversational), Mandarin (basic)
"""

# JSON Schema for resume extraction
# Note: OpenAI Structured Outputs requires ALL properties in 'required'
# and optional fields must use ["type", "null"] union
RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "personal_info": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "email": {"type": ["string", "null"]},
                "location": {"type": ["string", "null"]},
                "linkedin": {"type": ["string", "null"]}
            },
            "required": ["full_name", "email", "location", "linkedin"],
            "additionalProperties": False
        },
        "summary": {
            "type": ["string", "null"],
            "description": "Professional summary in 1-2 sentences"
        },
        "total_experience_years": {
            "type": "integer",
            "minimum": 0
        },
        "skills": {
            "type": "object",
            "properties": {
                "programming_languages": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "frameworks": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "cloud_platforms": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "databases": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["programming_languages", "frameworks", "cloud_platforms", "databases"],
            "additionalProperties": False
        },
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "start_year": {"type": ["integer", "null"]},
                    "end_year": {"type": ["integer", "null"]},
                    "is_current": {"type": "boolean"},
                    "highlights": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["title", "company", "start_year", "end_year", "is_current", "highlights"],
                "additionalProperties": False
            }
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string"},
                    "institution": {"type": "string"},
                    "year": {"type": ["integer", "null"]},
                    "gpa": {"type": ["number", "null"]}
                },
                "required": ["degree", "institution", "year", "gpa"],
                "additionalProperties": False
            }
        },
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "proficiency": {
                        "type": "string",
                        "enum": ["native", "fluent", "conversational", "basic"]
                    }
                },
                "required": ["language", "proficiency"],
                "additionalProperties": False
            }
        }
    },
    "required": ["personal_info", "summary", "total_experience_years", "skills", "experience", "education", "languages"],
    "additionalProperties": False
}

# Simple schema for quick demos
SIMPLE_PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": ["integer", "null"], "minimum": 0, "maximum": 150},
        "occupation": {"type": "string"},
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10
        },
        "contact": {
            "type": "object",
            "properties": {
                "email": {"type": ["string", "null"]},
                "phone": {"type": ["string", "null"]}
            },
            "required": ["email", "phone"],
            "additionalProperties": False
        }
    },
    "required": ["name", "age", "occupation", "skills", "contact"],
    "additionalProperties": False
}




async def demo_json_structured_output():
    """Run the JSON structured output demo."""

    parser = argparse.ArgumentParser(description="JSON Structured Output Demo")
    parser.add_argument("--text", help="Custom text to extract from")
    parser.add_argument("--model", default="gpt-5.2",
                        choices=["gpt-5.2", "gpt-5-mini", "claude-sonnet-4-5",
                                 "gemini-3-flash-preview", "grok-4-1-fast-non-reasoning",
                                 "z-ai/glm-4.6", "deepseek/deepseek-v3.2-exp", "qwen/qwen3-max"],
                        help="Model to use for extraction")
    parser.add_argument("--simple", action="store_true",
                        help="Use simpler schema for quick demo")

    args, _ = parser.parse_known_args()

    async with AsyncGranSabioClient() as client:
        info = await client.get_info()
        print(f"Connected to: {info['service']} v{info['version']}")

        # Select text and schema
        if args.text:
            source_text = args.text
            schema = SIMPLE_PERSON_SCHEMA
            schema_name = "Simple Person Schema"
        elif args.simple:
            source_text = "Maria Garcia, 28, data scientist at Google. Skills: Python, ML, SQL. Contact: maria@example.com"
            schema = SIMPLE_PERSON_SCHEMA
            schema_name = "Simple Person Schema"
        else:
            source_text = SAMPLE_RESUME_TEXT
            schema = RESUME_SCHEMA
            schema_name = "Resume Extraction Schema"

        print()
        print(f"Model: {args.model}")
        print(f"Schema: {schema_name}")
        print()
        print("Source Text Preview:")
        print("-" * 40)
        preview = source_text[:300].replace("\n", "\n  ")
        print(f"  {preview}...")

        # Show schema structure
        print()
        print("Expected Output Schema:")
        print("-" * 40)
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})
        for field_name, field_spec in list(properties.items())[:6]:
            field_type = field_spec.get("type", "unknown")
            required = "(required)" if field_name in required_fields else ""
            print(f"  {field_name}: {field_type} {required}")
        if len(properties) > 6:
            print(f"  ... and {len(properties) - 6} more fields")

        # Generate with schema enforcement
        print()
        print_header("Extracting with JSON Schema", "-")

        prompt = f"""
Extract structured information from the following text.
Return a JSON object that matches the provided schema exactly.

TEXT:
{source_text}

Extract all relevant information. For missing fields, use null or empty arrays.
        """.strip()

        result = await client.generate(
            prompt=prompt,
            content_type="json",
            generator_model=args.model,
            temperature=0.3,  # Lower temp for extraction
            max_tokens=2000,
            json_output=True,
            json_schema=schema,
            qa_layers=[],  # No QA - schema validation only
            qa_models=["gpt-5-mini"],
            verbose=True,
            request_name=f"JSON Extraction ({schema_name})",
            wait_for_completion=False  # Return immediately with session_id
        )

        session_id = result["session_id"]
        print(f"Session ID: {session_id}")

        if result.get("status") == "rejected":
            print(f"[REJECTED] {result.get('preflight_feedback', {}).get('user_feedback', 'Unknown')}")
            return

        final = await client.wait_for_completion(
            session_id,
            poll_interval=1.5,
            on_status=lambda s: print(f"  Status: {s['status']}")
        )

        # Parse and display full extracted data
        content = final.get("content", "{}")
        try:
            if isinstance(content, str):
                extracted = json.loads(content)
            else:
                extracted = content

            # Show full JSON output
            print_json_content(extracted, title="Extracted Data (Full JSON)")

            # Validation summary
            print()
            safe_print(colorize("  Schema Validation: PASSED", "green"))
            print(f"  Fields extracted: {len(extracted)}")

            # Show specific extractions for resume
            if "personal_info" in extracted:
                pi = extracted["personal_info"]
                print()
                safe_print(colorize("  Quick Summary:", "cyan"))
                print(f"  Name: {pi.get('full_name', 'N/A')}")
                print(f"  Location: {pi.get('location', 'N/A')}")
                if pi.get("email"):
                    print(f"  Email: {pi.get('email')}")

            if "experience" in extracted:
                exp = extracted['experience']
                print(f"  Experience: {len(exp)} positions")
                for job in exp[:3]:
                    years = ""
                    if job.get("start_year"):
                        end = "Present" if job.get("is_current") else job.get("end_year", "?")
                        years = f" ({job['start_year']}-{end})"
                    safe_print(f"    - {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}{years}")

            if "skills" in extracted:
                skills = extracted.get("skills", {})
                total_skills = sum(len(v) for v in skills.values() if isinstance(v, list))
                print(f"  Total Skills: {total_skills}")
                for category, skill_list in skills.items():
                    if skill_list:
                        print(f"    - {category}: {', '.join(skill_list[:5])}")

            if "education" in extracted:
                print(f"  Education: {len(extracted['education'])} entries")

            if "languages" in extracted:
                langs = [f"{l.get('language')} ({l.get('proficiency')})" for l in extracted['languages']]
                print(f"  Languages: {', '.join(langs)}")

        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON response: {e}")
            print(f"Raw content: {content[:500]}")

        # Compare with flexible JSON (optional)
        print()
        print_header("Schema vs Flexible Mode Comparison", "-")
        print()
        print("WITH json_schema (used above):")
        print("  - 100% guaranteed format compliance")
        print("  - Zero parsing errors")
        print("  - Model validates during generation")
        print("  - Works with GPT-4o, Claude 4, Gemini, Grok")
        print()
        print("WITHOUT json_schema (flexible mode):")
        print("  - Model decides structure")
        print("  - May need retry on format errors")
        print("  - More creative freedom")
        print("  - Better for open-ended generation")


if __name__ == "__main__":
    asyncio.run(run_demo(demo_json_structured_output, "Demo 05: JSON Structured Output"))
