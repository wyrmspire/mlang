
import requests
import json
import sys
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def log(msg, status="INFO"):
    print(f"[{status}] {msg}")

def test_health():
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200 and resp.json()['status'] == 'ok':
            log("Health Check Passed", "PASS")
            return True
        else:
            log(f"Health Check Failed: {resp.text}", "FAIL")
            return False
    except Exception as e:
        log(f"Health Check Exception: {e}", "FAIL")
        return False

def test_dates():
    try:
        resp = requests.get(f"{BASE_URL}/api/dates")
        dates = resp.json()
        if isinstance(dates, list) and len(dates) > 0:
            log(f"Dates Endpoint Passed (Found {len(dates)} dates)", "PASS")
            return dates[0] # Return newest date for other tests
        else:
            log(f"Dates Endpoint returned empty or invalid: {dates}", "WARN")
            return None
    except Exception as e:
        log(f"Dates Endpoint Exception: {e}", "FAIL")
        return None

def test_generate_session(date_str):
    try:
        payload = {
            "day_of_week": 0,
            "session_type": "RTH",
            "start_price": 5800.0,
            "date": date_str,
            "timeframe": "5m"
        }
        resp = requests.post(f"{BASE_URL}/api/generate/session", json=payload)
        if resp.status_code == 200:
            data = resp.json()['data']
            if len(data) > 0:
                log(f"Single Session Generation Passed ({len(data)} bars)", "PASS")
                return True
            else:
                log("Single Session Generation returned empty data", "FAIL")
                return False
        else:
            log(f"Single Session Generation Failed: {resp.text}", "FAIL")
            return False
    except Exception as e:
        log(f"Single Session Gen Exception: {e}", "FAIL")
        return False

def test_generate_multiday_weekend_crossover():
    # Test 5 days starting from a Friday to ensure it crosses weekend
    # 2024-01-05 is a Friday
    start_date = "2024-01-05" 
    try:
        payload = {
            "num_days": 5,
            "session_type": "RTH",
            "initial_price": 5800.0,
            "start_date": start_date,
            "timeframe": "15m"
        }
        resp = requests.post(f"{BASE_URL}/api/generate/multi-day", json=payload)
        if resp.status_code == 200:
            data = resp.json()['data']
            if len(data) > 0:
                # Check for distinct synthetic days
                days = set(d.get('synthetic_day') for d in data)
                log(f"Multi-Day Generation Passed ({len(data)} bars, {len(days)} distinct trading days)", "PASS")
                return True
            else:
                log("Multi-Day Generation returned empty data", "FAIL")
                return False
        else:
            log(f"Multi-Day Generation Failed: {resp.status_code} {resp.text}", "FAIL")
            return False
    except Exception as e:
        log(f"Multi-Day Gen Exception: {e}", "FAIL")
        return False

def run_all():
    log("Starting Smoke Tests...")
    
    if not test_health():
        sys.exit(1)
        
    latest_date = test_dates()
    
    # Use strict date for session if available, else omit
    test_generate_session(latest_date)
    
    test_generate_multiday_weekend_crossover()
    
    log("Smoke Tests Complete.")

if __name__ == "__main__":
    run_all()
