"""
Example requests for Gran Sabio LLM Engine
==========================================

This script demonstrates how to use the Gran Sabio LLM API with different
types of content generation requests.
"""

import asyncio
from typing import Any, Dict, Optional

import aiohttp

API_BASE_URL = "http://localhost:8000"


async def reserve_project_id(session: aiohttp.ClientSession, project_id: Optional[str] = None) -> str:
    """Reserve a project identifier via the /project/new endpoint."""
    payload: Dict[str, Any] = {}
    if project_id:
        payload["project_id"] = project_id

    async with session.post(f"{API_BASE_URL}/project/new", json=payload) as response:
        response.raise_for_status()
        data = await response.json()
        project_id = data["project_id"]
        print(f"📁 Reserved project: {project_id}")
        return project_id


async def test_biography_generation():
    """Test biography generation with comprehensive QA layers"""

    # Biography request with comprehensive QA layers
    request_data = {
        "prompt": "Write a complete biography of Marie Curie of approximately 1500 words, covering her personal life, scientific discoveries, and her impact on modern science.",
        "content_type": "biography",
        "generator_model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 4000,
        "qa_models": ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
        "qa_layers": [
            {
                "name": "Historical Accuracy",
                "description": "Verification of historical facts and dates",
                "criteria": "Verify all dates, historical events, and biographical details. Detect fabricated information or contradictions with known facts.",
                "min_score": 8.5,
                "is_deal_breaker": True,
                "order": 1
            },
            {
                "name": "Literary Quality",
                "description": "Writing style and narrative quality",
                "criteria": "Evaluate the quality of prose, narrative flow, language use, and ability to maintain reader interest.",
                "min_score": 7.5,
                "is_deal_breaker": False,
                "order": 2
            },
            {
                "name": "Structure and Organization",
                "description": "Content organization and logical structure",
                "criteria": "Evaluate chronological organization, clear sections, coherent transitions, and logical progression of the narrative.",
                "min_score": 7.0,
                "is_deal_breaker": False,
                "order": 3
            },
            {
                "name": "Depth and Coverage",
                "description": "Comprehensiveness of biographical content",
                "criteria": "Verify that all important aspects are covered: childhood, education, scientific career, achievements, challenges, and legacy.",
                "min_score": 7.5,
                "is_deal_breaker": False,
                "order": 4
            }
        ],
        "min_global_score": 8.0,
        "max_iterations": 3,
        "gran_sabio_model": "claude-opus",
        "verbose": True
    }

    print("🧬 Testing Biography Generation...")
    print(f"📝 Prompt: {request_data['prompt']}")
    print(f"🎯 Target: {request_data['min_global_score']} global score")
    print(f"🔄 Max iterations: {request_data['max_iterations']}")
    print()

    # Start generation
    async with aiohttp.ClientSession() as session:
        # Reserve a project identifier before the pipeline begins
        project_id = await reserve_project_id(session)
        request_data["project_id"] = project_id

        # Submit generation request
        async with session.post(f"{API_BASE_URL}/generate", json=request_data) as response:
            if response.status == 200:
                result = await response.json()
                session_id = result["session_id"]
                print(f"✅ Generation started. Session ID: {session_id}")
                project_id = result.get("project_id")
                if project_id:
                    print(f"📁 Assigned project: {project_id}")
            else:
                error = await response.text()
                print(f"❌ Failed to start generation: {error}")
                return

        # Monitor progress
        print("\n📊 Monitoring progress...")
        while True:
            async with session.get(f"{API_BASE_URL}/status/{session_id}") as response:
                if response.status == 200:
                    status = await response.json()
                    print(f"Status: {status['status']} | Iteration: {status['current_iteration']}/{status['max_iterations']}")

                    # Show verbose log
                    if status.get('verbose_log'):
                        for log_entry in status['verbose_log'][-3:]:  # Show last 3 entries
                            print(f"  📋 {log_entry}")

                    if status['status'] in ['completed', 'failed']:
                        break

                await asyncio.sleep(2)  # Check every 2 seconds

        # Get final result
        if status['status'] == 'completed':
            async with session.get(f"{API_BASE_URL}/result/{session_id}") as response:
                if response.status == 200:
                    final_result = await response.json()
                    print(f"\n🎉 Generation completed!")
                    print(f"📊 Final score: {final_result['final_score']}")
                    print(f"🔄 Final iteration: {final_result['final_iteration']}")
                    if final_result.get("project_id"):
                        print(f"📁 Project: {final_result['project_id']}")
                    print(f"\n📝 Generated content (first 500 chars):")
                    print(f"{final_result['content'][:500]}...")

                    if 'gran_sabio_reason' in final_result:
                        print(f"\n🧙‍♂️ Gran Sabio intervention: {final_result['gran_sabio_reason']}")
        else:
            print(f"❌ Generation failed with status: {status['status']}")
            if 'error' in status:
                print(f"Error: {status['error']}")


async def test_script_generation():
    """Test script generation with format-specific QA layers"""

    request_data = {
        "prompt": "Escribe el primer acto de un guión cinematográfico de drama sobre un científico que descubre algo que podría cambiar el mundo, pero debe decidir si revelarlo. Incluye diálogos naturales y descripciones de escena.",
        "content_type": "script",
        "generator_model": "claude-sonnet-4-20250514",
        "temperature": 0.8,
        "max_tokens": 3000,
        "qa_models": ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
        "qa_layers": [
            {
                "name": "Formato de Guión",
                "description": "Cumplimiento del formato estándar de guión",
                "criteria": "Verificar formato correcto: encabezados de escena, nombres de personajes, diálogos, descripciones de acción. Seguir estándares industriales.",
                "min_score": 8.0,
                "is_deal_breaker": True,
                "order": 1
            },
            {
                "name": "Calidad del Diálogo",
                "description": "Naturalidad y efectividad de los diálogos",
                "criteria": "Evaluar naturalidad del diálogo, diferenciación de voces de personajes, subtexto, y fluidez conversacional.",
                "min_score": 7.5,
                "is_deal_breaker": False,
                "order": 2
            },
            {
                "name": "Estructura Dramática",
                "description": "Estructura narrativa y desarrollo dramático",
                "criteria": "Evaluar setup, conflicto, desarrollo de personajes, tensión dramática, y progresión de la historia.",
                "min_score": 7.0,
                "is_deal_breaker": False,
                "order": 3
            }
        ],
        "min_global_score": 7.5,
        "max_iterations": 4,
        "gran_sabio_model": "claude-opus",
        "verbose": True
    }

    print("🎬 Testing Script Generation...")
    await _run_generation_test(request_data, "Script")


async def test_simple_article():
    """Test simple article generation with minimal QA"""

    request_data = {
        "prompt": "Escribe un artículo de divulgación científica sobre la inteligencia artificial explicando qué es, cómo funciona, y sus aplicaciones actuales. Hazlo accesible para el público general.",
        "content_type": "article",
        "generator_model": "gpt-4o-mini",
        "temperature": 0.6,
        "max_tokens": 2000,
        "qa_models": ["gpt-4o", "claude-sonnet-4-20250514"],
        "qa_layers": [
            {
                "name": "Precisión Técnica",
                "description": "Exactitud de la información técnica",
                "criteria": "Verificar que la información sobre IA sea técnicamente correcta y actualizada. No debe contener errores conceptuales.",
                "min_score": 8.0,
                "is_deal_breaker": True,
                "order": 1
            },
            {
                "name": "Claridad y Accesibilidad",
                "description": "Claridad para público general",
                "criteria": "Evaluar si el contenido es comprensible para lectores sin conocimiento técnico previo, uso de ejemplos, y explicaciones claras.",
                "min_score": 7.0,
                "is_deal_breaker": False,
                "order": 2
            }
        ],
        "min_global_score": 7.5,
        "max_iterations": 2,
        "verbose": True
    }

    print("📰 Testing Article Generation...")
    await _run_generation_test(request_data, "Article")


async def _run_generation_test(request_data: Dict[str, Any], content_type: str):
    """Helper function to run a generation test"""

    print(f"📝 Prompt: {request_data['prompt'][:100]}...")
    print(f"🎯 Target: {request_data['min_global_score']} global score")
    print(f"🔄 Max iterations: {request_data['max_iterations']}")
    print()

    async with aiohttp.ClientSession() as session:
        # Submit request
        async with session.post(f"{API_BASE_URL}/generate", json=request_data) as response:
            if response.status != 200:
                error = await response.text()
                print(f"❌ Failed to start generation: {error}")
                return

            result = await response.json()
            session_id = result["session_id"]
            print(f"✅ Generation started. Session ID: {session_id}")
            project_id = result.get("project_id")
            if project_id:
                print(f"📁 Project: {project_id}")

        # Monitor progress
        print("\n📊 Monitoring progress...")
        while True:
            async with session.get(f"{API_BASE_URL}/status/{session_id}") as response:
                if response.status == 200:
                    status = await response.json()
                    print(f"Status: {status['status']} | Iteration: {status['current_iteration']}/{status['max_iterations']}")

                    if status['status'] in ['completed', 'failed']:
                        break

                await asyncio.sleep(1)

        # Show results
        if status['status'] == 'completed':
            async with session.get(f"{API_BASE_URL}/result/{session_id}") as response:
                if response.status == 200:
                    final_result = await response.json()
                    print(f"\n🎉 {content_type} completed!")
                    print(f"📊 Final score: {final_result['final_score']}")
                    print(f"Generated: {len(final_result['content'])} characters")
                    if final_result.get("project_id"):
                        print(f"📁 Project: {final_result['project_id']}")
        else:
            print(f"❌ {content_type} failed: {status.get('error', 'Unknown error')}")


async def test_health_check():
    """Test API health and service availability"""
    print("🏥 Testing API Health...")

    async with aiohttp.ClientSession() as session:
        # Test root endpoint
        async with session.get(f"{API_BASE_URL}/") as response:
            if response.status == 200:
                info = await response.json()
                print(f"✅ API is running: {info['service']} v{info['version']}")
            else:
                print(f"❌ API not responding: {response.status}")
                return False

        # Test health endpoint
        async with session.get(f"{API_BASE_URL}/health") as response:
            if response.status == 200:
                health = await response.json()
                print(f"✅ Health check passed: {health['status']}")
                print(f"📊 Active sessions: {health['active_sessions']}")
            else:
                print(f"❌ Health check failed: {response.status}")
                return False

    return True


async def main():
    """Run all tests"""
    print("Gran Sabio LLM - API Testing Suite")
    print("=" * 50)

    # Test API availability first
    if not await test_health_check():
        print("❌ API not available. Please start the server first.")
        print("Run: python -m uvicorn main:app --reload")
        return

    print("\n" + "=" * 50)

    # Run content generation tests
    tests = [
        ("Biography Generation", test_biography_generation),
        ("Script Generation", test_script_generation),
        ("Article Generation", test_simple_article)
    ]

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        print("-" * 30)

        try:
            await test_func()
            print(f"✅ {test_name} completed")
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")

        print("\n" + "=" * 50)
        await asyncio.sleep(1)  # Brief pause between tests

    print("\n🎯 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
