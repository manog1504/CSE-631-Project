#!/usr/bin/env python3
"""Test TAMU API setup and project dependencies"""

import sys


def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    try:
        import openai
        print("  ✓ openai")
    except ImportError:
        print("  ✗ openai - install with: pip install openai")
        return False

    try:
        from sentence_transformers import SentenceTransformer
        print("  ✓ sentence_transformers")
    except ImportError:
        print("  ✗ sentence_transformers - install with: pip install sentence-transformers")
        return False

    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError:
        print("  ✗ scikit-learn - install with: pip install scikit-learn")
        return False

    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - install with: pip install pandas")
        return False

    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib - install with: pip install matplotlib")
        return False

    return True


def test_api_connection():
    """Test TAMU API connection"""
    print("\nTesting TAMU API connection...")
    from protocols.client import call_llm

    test_messages = [
        {"role": "user", "content": "What is a Nash equilibrium? Answer in one sentence."}
    ]

    print("  Calling API with test prompt...", end="", flush=True)
    try:
        response, usage = call_llm(test_messages, temperature=0.7, max_tokens=100)

        if response:
            print(" ✓")
            print(f"  Response: {response[:80]}...")
            print(f"  Tokens: {usage['total_tokens']}")
            return True
        else:
            print(" ✗")
            print("  API call returned empty response. Check CF_Authorization cookie.")
            return False

    except Exception as e:
        print(f" ✗\n  Error: {e}")
        print("\n  Troubleshooting:")
        print("  1. Verify CF_Authorization cookie in config.py is fresh (not expired)")
        print("  2. Verify TAMU_API_KEY in config.py is correct")
        print("  3. Check internet connection")
        return False


def test_embedding_model():
    """Test sentence embedding model download"""
    print("\nTesting embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        print("  Downloading all-MiniLM-L6-v2...", end="", flush=True)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(" ✓")

        # Test encoding
        embeddings = model.encode(["Hello world", "Test embedding"])
        print(f"  Embedding shape: {embeddings.shape}")
        return True

    except Exception as e:
        print(f" ✗\n  Error: {e}")
        return False


def test_directories():
    """Test that output directories exist"""
    print("\nTesting output directories...")
    from pathlib import Path

    dirs = ["transcripts", "results", "results/figures", "report"]
    for d in dirs:
        p = Path(d)
        if p.exists():
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ - creating...")
            p.mkdir(parents=True, exist_ok=True)

    return True


def main():
    print("=" * 60)
    print("CSCE 631 LLM Debate Project — Setup Verification")
    print("=" * 60)

    all_good = True

    all_good &= test_imports()
    all_good &= test_directories()
    all_good &= test_embedding_model()
    all_good &= test_api_connection()

    print("\n" + "=" * 60)
    if all_good:
        print("✓ All checks passed! You're ready to run experiments.")
        print("\n  Next steps:")
        print("  1. (Optional) Verify a single experiment:")
        print("     python -c \"from protocols.protocol_a import run_simultaneous; import json; topics = json.load(open('topics.json')); run_simultaneous(topics[0]['topic'], topics[0]['id'])\"")
        print("  2. Run all experiments:")
        print("     python run_experiments.py")
        print("  3. Analyze results:")
        print("     python analyze_results.py")
        return 0
    else:
        print("✗ Some checks failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
