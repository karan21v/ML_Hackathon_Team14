"""
PathSnap Server Launcher - Laptop 1 (Model and API).
Detects local network IP, displays setup instructions for Laptop 2,
and runs Uvicorn server bound to 0.0.0.0:8000.
"""

import os
import sys
import socket
import uvicorn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

def get_lan_ips():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        if local_ip and not local_ip.startswith("127."):
            ips.append(local_ip)
        s.close()
    except Exception:
        pass

    if not ips:
        ips = ["127.0.0.1"]
    return ips

def main():
    lan_ips = get_lan_ips()
    primary_ip = lan_ips[0]

    print("=" * 65)
    print("      PATHSNAP HACKATHON - LAPTOP 1 API & MODEL SERVER")
    print("=" * 65)
    print(f"Hostname:             {socket.gethostname()}")
    print(f"Laptop 1 Local IP:    {primary_ip}")
    print(f"All Network IPs:      {', '.join(lan_ips)}")
    print("-" * 65)
    print("DEMO INSTRUCTIONS FOR EVALUATOR (LAPTOP 2):")
    print(" 1. Ensure Laptop 2 is connected to the same Wi-Fi / Mobile Hotspot.")
    print(" 2. On Laptop 2, open any browser (Chrome, Edge, Firefox).")
    print(f" 3. Navigate to:  http://{primary_ip}:8000/ui")
    print(f" 4. Interactive Docs: http://{primary_ip}:8000/docs")
    print("-" * 65)
    print("NOTE ON WINDOWS FIREWALL:")
    print(" If Laptop 2 cannot connect, ensure Windows Firewall allows port 8000:")
    print("   PowerShell (Admin): New-NetFirewallRule -DisplayName 'PathSnap' -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow")
    print("=" * 65)
    print("[Starting FastAPI server on 0.0.0.0:8000 ...]\n")

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
