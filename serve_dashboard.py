import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000

def serve():
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
    # We use a timer or just call it before serve_forever (it's non-blocking usually if it just spawns a process, but safe to do before)
    webbrowser.open(f"http://localhost:{PORT}/dashboard/")

    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving HTTP on 0.0.0.0 port {PORT} ...")
            print("Press Ctrl+C to stop.")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:
            print(f"Error: Port {PORT} is already in use.")
            print("Try stopping the other server or wait a moment.")
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        print("\nStopping server...")

if __name__ == "__main__":
    serve()
