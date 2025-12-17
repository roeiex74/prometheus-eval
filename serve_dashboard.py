import http.server
import socketserver
import webbrowser
import os
import sys

import subprocess
import time

PORT = 8000

def kill_existing_server():
    """Kills any process running on PORT."""
    try:
        # Find PID using lsof
        # -t: terse output (only PID)
        # -i: Internet address
        result = subprocess.run(
            ["lsof", "-t", f"-i:{PORT}"], 
            capture_output=True, 
            text=True
        )
        pids = result.stdout.strip().split('\n')
        
        for pid in pids:
            if pid:
                print(f"Stopping existing server (PID: {pid})...")
                subprocess.run(["kill", "-9", pid], check=False)
                time.sleep(1) # Wait for port to clear
                
    except Exception as e:
        print(f"Warning: Failed to cleanup port {PORT}: {e}")

def serve():
    kill_existing_server()

    # Ensure we are serving from the project root
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)
    
    # Custom handler to suppress default log noise if needed, 
    # but standard SimpleHTTPRequestHandler is fine.
    Handler = http.server.SimpleHTTPRequestHandler

    print(f"Starting dashboard server...")
    print(f"Root Directory: {root_dir}")
    print(f"Dashboard URL: http://localhost:{PORT}/dashboard/")
    
    # Open the browser automatically
    webbrowser.open(f"http://localhost:{PORT}/dashboard/")

    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving HTTP on 0.0.0.0 port {PORT} ...")
            print("Press Ctrl+C to stop.")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:
             # Should be handled by kill_existing_server, but just in case
            print(f"Error: Port {PORT} is still in use.")
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        print("\nStopping server...")

if __name__ == "__main__":
    serve()
