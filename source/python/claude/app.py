from flask import Flask, request, Response
import requests
import os
import json

app = Flask(__name__)

# Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_BASE_URL = 'https://api.anthropic.com'
FORWARD_HEADERS = ['content-type', 'accept', 'user-agent']

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy(path):
    try:
        # Build target URL
        target_url = f"{ANTHROPIC_BASE_URL}/{path}"
         
        # Prepare headers
        headers = {
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
            'Host': 'api.anthropic.com'  # Override the Host header explicitly
        }
        
        # Add select headers from original request (excluding Host)
        for header in FORWARD_HEADERS:
            if header in request.headers:
                headers[header] = request.headers[header]

        # Log incoming request
        print("\n=== Incoming Request ===")
        print(f"Method: {request.method}")
        print(f"Path: {request.path}")
        print(f"Headers: {json.dumps(dict(request.headers), indent=2)}")
        print(f"Body: {request.get_data().decode('utf-8')}\n")

        # Forward the request
        response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=request.args,
            data=request.get_data(),
            stream=True,
            timeout=30
        )

        # Log outgoing request and response
        print("\n=== Outgoing Request ===")
        print(f"URL: {target_url}")
        print(f"Headers: {json.dumps(headers, indent=2)}")
        print(f"Body: {request.get_data().decode('utf-8')}")

        print("\n=== Response ===")
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"Body: {response.text[:500]}...\n")  # Limit body size for readability

        # Build response
        proxy_response = Response(
            response.iter_content(chunk_size=8192),
            status=response.status_code
        )
        # Include response headers (excluding certain headers to avoid conflicts)
        excluded_headers = [
            'transfer-encoding', 'content-encoding', 'content-length', 'connection',
            'date', 'server', 'via', 'cf-ray', 'cf-cache-status', 'x-robots-tag'
        ]
        for key, value in response.headers.items():
            if key.lower() not in excluded_headers:
                proxy_response.headers[key] = value

        return proxy_response

    except requests.exceptions.RequestException as e:
        print(f"\n=== Error ===\n{e}\n")
        return Response(str(e), status=500)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
