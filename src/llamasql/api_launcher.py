#!/usr/bin/env python
"""
LlamaDB API Server Management Script
This script helps manage the LlamaDB API server process.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from typing import List, Optional, Tuple

DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"


def is_port_in_use(port: int, host: str = DEFAULT_HOST) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_process_using_port(port: int) -> Optional[int]:
    """Find the process ID using the specified port."""
    if sys.platform.startswith("darwin") or sys.platform.startswith("linux"):
        # macOS or Linux
        try:
            # Use lsof to find the process
            cmd = f"lsof -i :{port} -t"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                return int(result.stdout.strip())
        except Exception as e:
            print(f"Error finding process: {e}")
    elif sys.platform.startswith("win"):
        # Windows
        try:
            # Use netstat to find the process
            cmd = f"netstat -ano | findstr :{port}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.strip().split()
                        if len(parts) > 4:
                            return int(parts[-1])
        except Exception as e:
            print(f"Error finding process: {e}")

    return None


def kill_process(pid: int) -> bool:
    """Kill a process by its PID."""
    try:
        os.kill(pid, signal.SIGTERM)
        # Give it a moment to terminate gracefully
        time.sleep(1)
        # Check if it's still running
        try:
            os.kill(
                pid, 0
            )  # Doesn't actually send a signal, just checks if process exists
            # Process still exists, force kill
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
        except OSError:
            # Process already terminated
            pass
        return True
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False


def start_server(
    port: int = DEFAULT_PORT, host: str = DEFAULT_HOST, restart: bool = False
) -> None:
    """Start the LlamaDB API server."""
    if is_port_in_use(port, host):
        print(f"Port {port} is already in use.")
        if restart:
            pid = find_process_using_port(port)
            if pid:
                print(f"Killing process {pid} using port {port}...")
                if kill_process(pid):
                    print(f"Successfully killed process {pid}.")
                else:
                    print(f"Failed to kill process {pid}. Please kill it manually.")
                    return
            else:
                print(
                    f"Could not find process using port {port}. Please free the port manually."
                )
                return
        else:
            print("Use --restart to kill the existing process.")
            return

    # Start the server
    print(f"Starting LlamaDB API server on {host}:{port}...")

    try:
        cmd = [
            sys.executable,
            "-m",
            "llamadb.cli.main",
            "serve",
            "--host",
            host,
            "--port",
            str(port),
        ]
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Wait for server to start
        for _ in range(10):
            if is_port_in_use(port, host):
                print(f"LlamaDB API server is running at http://{host}:{port}")
                break
            time.sleep(0.5)

        # Print a few lines of output
        print("\nServer output:")
        try:
            for _ in range(5):
                line = server_process.stdout.readline()
                if not line:
                    break
                print(f"  {line.strip()}")
        except:
            pass

        print("\nServer is running in the background. To stop it, use:")
        print(f"  python {os.path.basename(__file__)} --stop --port {port}")
    except Exception as e:
        print(f"Error starting server: {e}")


def stop_server(port: int = DEFAULT_PORT) -> None:
    """Stop the LlamaDB API server."""
    if not is_port_in_use(port):
        print(f"No server running on port {port}.")
        return

    pid = find_process_using_port(port)
    if pid:
        print(f"Stopping LlamaDB API server (PID: {pid})...")
        if kill_process(pid):
            print("Server stopped successfully.")
        else:
            print("Failed to stop server. Please stop it manually.")
    else:
        print(f"Could not find process using port {port}.")


def main() -> None:
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Manage LlamaDB API server")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to use (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host to bind to (default: {DEFAULT_HOST})",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="Start the server")
    group.add_argument("--stop", action="store_true", help="Stop the server")
    group.add_argument("--restart", action="store_true", help="Restart the server")
    group.add_argument(
        "--status", action="store_true", help="Check if server is running"
    )

    args = parser.parse_args()

    if args.status:
        if is_port_in_use(args.port, args.host):
            pid = find_process_using_port(args.port)
            print(
                f"LlamaDB API server is running on {args.host}:{args.port}"
                + (f" (PID: {pid})" if pid else "")
            )
        else:
            print(f"No LlamaDB API server running on {args.host}:{args.port}")
    elif args.start:
        start_server(args.port, args.host)
    elif args.stop:
        stop_server(args.port)
    elif args.restart:
        start_server(args.port, args.host, restart=True)


if __name__ == "__main__":
    main()
