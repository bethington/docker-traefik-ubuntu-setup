docker build -t claude-proxy .
docker run -d \
  -p 5000:5000 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  --name claude-proxy \
  claude-proxy