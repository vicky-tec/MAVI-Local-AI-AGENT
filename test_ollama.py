#!/usr/bin/env python
"""Quick Test Script for Ollama Integration"""
import sys
from openai import OpenAI

# Initialize client with Ollama endpoint
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Required but not used by Ollama
)

print("=" * 60)
print("Testing Ollama Integration")
print("=" * 60)

# Test 1: Embedding Generation
print("\n1. Testing Embedding Generation (qwen3-embedding:0.6b)...")
try:
    response = client.embeddings.create(
        model="qwen3-embedding:0.6b",
        input=["This is a test document for embedding generation."]
    )
    embedding = response.data[0].embedding
    print(f"✓ Success! Generated embedding with {len(embedding)} dimensions")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Chat Completion
print("\n2. Testing Chat Completion (llama3.2:3b)...")
try:
    response = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Say 'Hello! I'm running on Ollama.' in exactly those words."}
        ],
        max_tokens=50
    )
    answer = response.choices[0].message.content
    print(f"✓ Response: {answer}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: Streaming Chat
print("\n3. Testing Streaming Chat...")
try:
    stream = client.chat.completions.create(
        model="llama3.2:3b",
        messages=[
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        stream=True,
        max_tokens=50
    )
    print("✓ Stream output: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("=" * 60)
print("All Tests Passed! ✓")
print("=" * 60)
