"""
Quick Start Script for Gran Sabio LLM Engine
=============================================

Fast setup and testing script to get the engine running quickly.
"""

import os
import sys
import asyncio
import aiohttp
import json
from pathlib import Path


def check_env_file():
    """Check if .env file exists and has required keys"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("ERROR: .env file not found!")
        print("\nCreating .env from template...")
        
        # Copy template to .env
        template_path = Path(".env.template")
        if template_path.exists():
            env_content = template_path.read_text()
            env_path.write_text(env_content)
            print("SUCCESS: .env file created from template")
            print("\nIMPORTANT: Please edit .env file and add your API keys:")
            print("   - OPENAI_API_KEY")
            print("   - ANTHROPIC_API_KEY") 
            print("   - GOOGLE_API_KEY")
            return False
        else:
            print("ERROR: .env.template not found!")
            return False
    
    # Check if API keys are set
    env_content = env_path.read_text()
    
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if f"{key}=your-" in env_content or f"{key}=" not in env_content:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"WARNING: Missing or placeholder API keys: {', '.join(missing_keys)}")
        print("Please edit .env file and add your real API keys")
        return False
    
    print("SUCCESS: .env file is properly configured")
    return True


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import openai
        import anthropic
        import google.generativeai
        from dotenv import load_dotenv
        print("SUCCESS: All dependencies are installed")
        return True
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False


async def test_basic_request():
    """Test a basic API request"""
    print("\nTesting basic API functionality...")
    
    # Simple test request
    test_request = {
        "prompt": "Escribe un párrafo corto sobre la importancia de la inteligencia artificial en el futuro.",
        "content_type": "article",
        "generator_model": "gpt-4o-mini",  # Use cheaper model for testing
        "temperature": 0.7,
        "max_tokens": 200,
        "qa_models": ["gpt-4o-mini"],  # Single model for faster testing
        "qa_layers": [
            {
                "name": "Calidad Básica",
                "description": "Evaluación básica de calidad",
                "criteria": "Verificar que el contenido sea coherente y bien escrito",
                "min_score": 6.0,
                "is_deal_breaker": False,
                "order": 1
            }
        ],
        "min_global_score": 6.0,
        "max_iterations": 1,  # Just one iteration for testing
        "verbose": True
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test API availability
            async with session.get("http://localhost:8000/health") as response:
                if response.status != 200:
                    print("ERROR: API not running. Please start the server first:")
                    print("   python main.py")
                    return False
                
                health = await response.json()
                print(f"SUCCESS: API is running: {health['status']}")
            
            # Submit test request
            async with session.post("http://localhost:8000/generate", json=test_request) as response:
                if response.status != 200:
                    error = await response.text()
                    print(f"ERROR: Request failed: {error}")
                    return False
                
                result = await response.json()
                session_id = result["session_id"]
                print(f"SUCCESS: Generation started: {session_id}")
            
            # Monitor progress (simple version)
            print("Monitoring progress...")
            for i in range(30):  # Max 30 seconds
                async with session.get(f"http://localhost:8000/status/{session_id}") as response:
                    status = await response.json()
                    print(f"   Status: {status['status']}")
                    
                    if status['status'] in ['completed', 'failed']:
                        break
                    
                    await asyncio.sleep(1)
            
            # Get result if completed
            if status['status'] == 'completed':
                async with session.get(f"http://localhost:8000/result/{session_id}") as response:
                    final_result = await response.json()
                    print("SUCCESS: Test completed successfully!")
                    print(f"Final score: {final_result['final_score']}")
                    print(f"Generated content: {final_result['content'][:100]}...")
                    return True
            else:
                print(f"ERROR: Test failed with status: {status['status']}")
                return False
                
    except Exception as e:
        print(f"ERROR: Test failed with error: {e}")
        return False


def main():
    """Main setup and test function"""
    print("Gran Sabio LLM - Quick Start")
    print("=" * 50)
    
    # Check environment setup
    print("\n1. Checking environment setup...")
    if not check_env_file():
        print("\nERROR: Please configure .env file first, then run this script again")
        return
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        print("\nERROR: Please install dependencies first: pip install -r requirements.txt")
        return
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Import after loading env
    try:
        from config import config
        print("SUCCESS: Configuration loaded")
    except Exception as e:
        print(f"ERROR: Configuration error: {e}")
        return
    
    print("\n3. Starting API test...")
    print("INFO: Make sure the server is running in another terminal:")
    print("   python main.py")
    print("\nPress Enter to continue with API test, or Ctrl+C to exit...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return
    
    # Run API test
    result = asyncio.run(test_basic_request())
    
    if result:
        print("\nSUCCESS: Gran Sabio LLM is working correctly!")
        print("\nNext steps:")
        print("   - Check the full documentation: http://localhost:8000/docs")
        print("   - Run comprehensive tests: python example_request.py")
        print("   - Create your own requests using the API")
    else:
        print("\nERROR: Something went wrong. Check the server logs for details.")


if __name__ == "__main__":
    main()