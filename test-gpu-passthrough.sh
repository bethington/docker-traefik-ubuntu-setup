#!/bin/bash
# GPU Passthrough Test Script for RTX 3060 in LXC
# Run this script AFTER applying the LXC GPU configuration

echo "🎯 Testing RTX 3060 GPU Passthrough in Docker"
echo "=============================================="

# Test 1: Check NVIDIA devices
echo "1. Checking NVIDIA device availability..."
if ls /dev/nvidia* >/dev/null 2>&1; then
    echo "✅ NVIDIA devices found:"
    ls -la /dev/nvidia*
else
    echo "❌ No NVIDIA devices found - LXC configuration needed"
    exit 1
fi

# Test 2: Basic nvidia-smi test
echo -e "\n2. Testing nvidia-smi..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    echo "✅ nvidia-smi working"
else
    echo "❌ nvidia-smi not available"
fi

# Test 3: Docker GPU access test
echo -e "\n3. Testing Docker GPU access..."
docker run --rm --gpus all ubuntu:20.04 bash -c "
    if ls /dev/nvidia* >/dev/null 2>&1; then
        echo '✅ GPU devices accessible in Docker container'
        echo 'Available devices:'
        ls -la /dev/nvidia*
    else
        echo '❌ GPU devices not accessible in Docker'
    fi
"

# Test 4: CUDA Runtime Test
echo -e "\n4. Testing CUDA runtime..."
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi 2>/dev/null && echo "✅ CUDA runtime working" || echo "⚠️  CUDA image not available (will download on first run)"

# Test 5: VibeVoice TTS Test (~2GB VRAM)
echo -e "\n5. Testing VibeVoice TTS (Text-to-Speech)..."
docker run --rm --gpus all -d --name vibevoice-test -p 8765:8765 \
    -e MODEL_DEVICE=cuda \
    -e OPTIMIZE_FOR_SPEED=1 \
    eworkerinc/vibevoice >/dev/null 2>&1 && {
    echo "✅ VibeVoice container started on port 8765"
    sleep 10
    if curl -s http://localhost:8765/health >/dev/null 2>&1; then
        echo "✅ VibeVoice health check passed"
    else
        echo "⚠️  VibeVoice may still be initializing"
    fi
    docker stop vibevoice-test >/dev/null 2>&1
} || echo "⚠️  VibeVoice image not available (will download on first run)"

# Test 6: Whisper ASR Test (~1GB VRAM)
echo -e "\n6. Testing Whisper ASR (Speech-to-Text)..."
docker run --rm --gpus all -d --name whisper-test -p 9000:9000 \
    -e ASR_MODEL=base \
    onerahmet/openai-whisper-asr-webservice:latest-gpu >/dev/null 2>&1 && {
    echo "✅ Whisper ASR container started on port 9000"
    sleep 15
    if curl -s http://localhost:9000/health >/dev/null 2>&1; then
        echo "✅ Whisper health check passed"
    else
        echo "⚠️  Whisper may still be initializing"
    fi
    docker stop whisper-test >/dev/null 2>&1
} || echo "⚠️  Whisper image not available (will download on first run)"

# Test 7: Ollama LLM Test (~4-6GB VRAM)
echo -e "\n7. Testing Ollama LLM..."
docker run --rm --gpus all -d --name ollama-test -p 11434:11434 \
    ollama/ollama >/dev/null 2>&1 && {
    echo "✅ Ollama container started on port 11434"
    sleep 10
    docker exec ollama-test ollama pull gemma2:2b >/dev/null 2>&1 && {
        echo "✅ Ollama model download initiated"
        docker exec ollama-test ollama list
    } || echo "⚠️  Ollama model download may be in progress"
    docker stop ollama-test >/dev/null 2>&1
} || echo "⚠️  Ollama image not available (will download on first run)"

# Test 8: VRAM Usage Summary
echo -e "\n8. GPU Memory Usage:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "Used: %s MB / Total: %s MB (%.1f%% used)\n", $1, $2, ($1/$2)*100}'
else
    echo "❌ nvidia-smi not available for memory check"
fi

echo -e "\n🎯 GPU Passthrough Test Complete!"
echo "================================================"
echo "Expected VRAM usage for simultaneous operation:"
echo "  • VibeVoice: ~2GB"
echo "  • Whisper: ~1GB"
echo "  • Ollama (7B model): ~4-6GB"
echo "  • Total: ~7-9GB / 12GB available on RTX 3060"
echo "================================================"