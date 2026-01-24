FAKE AI SERVER - Response Files
================================

This directory contains response files for the Fake AI server.
Each .txt file corresponds to a fake model name.

BASIC MODELS:
- Generator-Dumb.txt: Content generation responses
- QA-Dumb.txt: QA evaluation responses (normal pass)
- GranSabio-Dumb.txt: GranSabio review responses (approves)
- Arbiter-Dumb.txt: Arbiter conflict resolution responses

ACTION VARIANTS (use model:action syntax):
- QA-Dumb_dealbreaker.txt: QA that flags deal-breakers
- QA-Dumb_with-edits.txt: QA that proposes smart-edits
- GranSabio-Dumb_reject.txt: GranSabio that confirms deal-breakers

LAYER-SPECIFIC (auto-detected from prompt):
- QA-Dumb.Accuracy.txt: Response for "Accuracy" layer
- QA-Dumb.Completeness.txt: Response for "Completeness" layer

FILE NAMING CONVENTION:
- Default:        {model}.txt
- With action:    {model}_{action}.txt
- Layer-specific: {model}.{layer}.txt

HOW TO USE:
1. Start the fake server:
   python dev_tests/fake_ai_server.py

2. Set environment variable:
   FAKE_AI_HOST=http://localhost:8989

3. Use model names in your request:
   - model: "Generator-Dumb"
   - qa_models: ["QA-Dumb", "gpt-5.2"]  (mix fake + real)
   - gran_sabio_model: "GranSabio-Dumb"
   - arbiter_model: "Arbiter-Dumb"

4. To use action variants, use model:action syntax:
   - qa_models: ["QA-Dumb:dealbreaker"]   -> reads QA-Dumb_dealbreaker.txt
   - qa_models: ["QA-Dumb:with-edits"]    -> reads QA-Dumb_with-edits.txt
   - gran_sabio_model: "GranSabio-Dumb:reject" -> reads GranSabio-Dumb_reject.txt

HOW IT WORKS:
The fake server reads the .txt file matching the model name and returns
its contents as the AI response. Edit these files to control exactly
what the "AI" returns, enabling deterministic testing.

PRIORITY ORDER:
1. Action variant: QA-Dumb:dealbreaker -> QA-Dumb_dealbreaker.txt
2. Layer-specific: QA-Dumb with "Accuracy" in prompt -> QA-Dumb.Accuracy.txt
3. Default: QA-Dumb -> QA-Dumb.txt

EXAMPLE SCENARIOS:

1. Test minority deal-breaker flow:
   - Use 3 QA models: ["QA-Dumb:dealbreaker", "gpt-5.2", "claude-sonnet-4-5"]
   - The fake model will flag deal-breaker, creating a minority (1/3)
   - GranSabio-Dumb will be called to review

2. Test GranSabio rejection:
   - Use: gran_sabio_model: "GranSabio-Dumb:reject"
   - GranSabio will confirm the deal-breaker and force iteration

3. Test smart-edit flow:
   - Use: qa_models: ["QA-Dumb:with-edits"]
   - QA will propose edits that Arbiter will process

SECURITY:
- Only alphanumeric, underscore, and hyphen allowed in model/action names
- Path traversal attempts (../, etc.) are blocked
