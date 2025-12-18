"""
MLang2 API Server
FastAPI backend serving viz data and proxying agent chat to Gemini.

Run:
    uvicorn src.server.main:app --reload --port 8000
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from src.config import RESULTS_DIR


app = FastAPI(title="MLang2 API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Viz data directory
VIZ_DIR = RESULTS_DIR / "viz"


# =============================================================================
# MODELS
# =============================================================================

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatContext(BaseModel):
    runId: str
    currentIndex: int
    currentMode: str  # 'DECISION' or 'TRADE'


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: ChatContext


class UIAction(BaseModel):
    type: str
    payload: Any


class AgentResponse(BaseModel):
    reply: str
    ui_action: Optional[UIAction] = None


# =============================================================================
# ENDPOINTS: Runs
# =============================================================================

def find_run_dir(run_id: str) -> Optional[Path]:
    """Find the directory for a run ID, searching all results subdirs."""
    # Check viz directory first
    viz_path = RESULTS_DIR / "viz" / run_id
    if viz_path.exists():
        return viz_path
    # Check direct in results
    direct_path = RESULTS_DIR / run_id
    if direct_path.exists():
        return direct_path
    # Search all subdirs
    for subdir in RESULTS_DIR.iterdir():
        if subdir.is_dir():
            candidate = subdir / run_id if subdir.name != run_id else subdir
            if candidate.exists() and candidate.is_dir():
                return candidate
            # Also check if the subdir itself is the run
            if subdir.name == run_id:
                return subdir
    return None


def find_jsonl_file(run_dir: Path, names: List[str]) -> Optional[Path]:
    """Find a JSONL file by trying multiple names."""
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    # Also try any .jsonl file
    jsonl_files = list(run_dir.glob("*.jsonl"))
    if jsonl_files:
        return jsonl_files[0]
    return None


@app.get("/runs")
async def list_runs() -> List[str]:
    """List available runs from all results subdirectories."""
    runs = set()
    
    if not RESULTS_DIR.exists():
        return []
    
    # Check direct subdirs of results
    for subdir in RESULTS_DIR.iterdir():
        if subdir.is_dir():
            # If it contains data files, it's a run
            if any(subdir.glob("*.jsonl")) or any(subdir.glob("*.json")):
                runs.add(subdir.name)
            # Also check nested dirs (like results/viz/run_name)
            for nested in subdir.iterdir():
                if nested.is_dir() and (any(nested.glob("*.jsonl")) or any(nested.glob("*.json"))):
                    runs.add(nested.name)
    
    return sorted(runs)


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    """Get run metadata."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    # Try run.json, then summary.json
    for name in ["run.json", "summary.json"]:
        path = run_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    
    # Return basic info if no metadata file
    return {"run_id": run_id, "path": str(run_dir)}


@app.get("/runs/{run_id}/manifest")
async def get_manifest(run_id: str) -> Dict[str, Any]:
    """Get run manifest."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    
    # Return basic manifest if not found
    return {"run_id": run_id, "start_time": "", "config": {}}


@app.get("/runs/{run_id}/decisions")
async def get_decisions(run_id: str) -> List[Dict[str, Any]]:
    """Get all decisions for a run."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    # Try multiple file names
    decisions_file = find_jsonl_file(run_dir, [
        "decisions.jsonl",
        "or_multi_oco_records.jsonl",  # OR strategy output
        "records.jsonl"
    ])
    
    if not decisions_file:
        return []
    
    decisions = []
    with open(decisions_file) as f:
        for i, line in enumerate(f):
            if line.strip():
                record = json.loads(line)
                # Normalize to VizDecision format
                if "decision_id" not in record:
                    record["decision_id"] = record.get("id", f"rec_{i}")
                if "index" not in record:
                    record["index"] = i
                decisions.append(record)
    return decisions


@app.get("/runs/{run_id}/trades")
async def get_trades(run_id: str) -> List[Dict[str, Any]]:
    """Get all trades for a run."""
    run_dir = find_run_dir(run_id)
    if not run_dir:
        raise HTTPException(404, f"Run {run_id} not found")
    
    trades_file = find_jsonl_file(run_dir, ["trades.jsonl"])
    
    if not trades_file:
        return []  # OR data doesn't have separate trades file
    
    trades = []
    with open(trades_file) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    return trades


# =============================================================================
# ENDPOINTS: Agent Chat
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash-exp"


def build_agent_prompt(context: ChatContext, decisions: List[Dict], trades: List[Dict]) -> str:
    """Build system prompt for the trade viz agent."""
    # Find current item
    if context.currentMode == "DECISION":
        current = next((d for d in decisions if d.get("index") == context.currentIndex), None)
        item_type = "decision"
    else:
        current = next((t for t in trades if t.get("index") == context.currentIndex), None)
        item_type = "trade"
    
    current_json = json.dumps(current, indent=2) if current else "None selected"
    
    return f"""You are a trade analysis assistant for the MLang2 trading research platform.
You can BOTH analyze existing data AND run new strategies to create data.

CURRENT CONTEXT:
- Run ID: {context.runId}
- Viewing: {item_type} index {context.currentIndex}
- Mode: {context.currentMode}

CURRENT {item_type.upper()} DATA:
{current_json}

AVAILABLE ACTIONS:
1. Navigate: ACTION: {{"type": "SET_INDEX", "payload": <number>}}
2. Switch mode: ACTION: {{"type": "SET_MODE", "payload": "DECISION" or "TRADE"}}
3. Load run: ACTION: {{"type": "LOAD_RUN", "payload": "<run_id>"}}
4. RUN STRATEGY: ACTION: {{"type": "RUN_STRATEGY", "payload": {{"strategy": "opening_range", "start_date": "2025-03-17", "weeks": 3}}}}

AVAILABLE STRATEGIES: "opening_range" (or "or")

When user asks to run/create/generate data, use RUN_STRATEGY.
Include action at END in format: ACTION: {{"type": "...", "payload": ...}}
Be concise."""


@app.post("/agent/chat")
async def agent_chat(request: ChatRequest) -> AgentResponse:
    """Proxy chat to Gemini agent with trade context."""
    
    if not GEMINI_API_KEY:
        return AgentResponse(
            reply="Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
        )
    
    # Load run data for context
    try:
        decisions = await get_decisions(request.context.runId)
        trades = await get_trades(request.context.runId)
    except HTTPException:
        decisions = []
        trades = []
    
    # Build prompt
    system_prompt = build_agent_prompt(request.context, decisions, trades)
    
    # Build messages for Gemini
    gemini_messages = [{"role": "user", "parts": [{"text": system_prompt}]}]
    
    for msg in request.messages:
        role = "user" if msg.role == "user" else "model"
        gemini_messages.append({"role": role, "parts": [{"text": msg.content}]})
    
    # Call Gemini API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json={"contents": gemini_messages},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract text from response
            reply_text = ""
            if "candidates" in data and data["candidates"]:
                parts = data["candidates"][0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        reply_text += part["text"]
            
            if not reply_text:
                reply_text = "I didn't receive a valid response from the model."
            
            # Parse for ACTION
            ui_action = None
            if "ACTION:" in reply_text:
                action_str = reply_text.split("ACTION:")[-1].strip()
                try:
                    action_data = json.loads(action_str)
                    ui_action = UIAction(**action_data)
                    # Remove action from reply text
                    reply_text = reply_text.split("ACTION:")[0].strip()
                except (json.JSONDecodeError, ValueError):
                    pass
            
            return AgentResponse(reply=reply_text, ui_action=ui_action)
            
        except httpx.HTTPError as e:
            return AgentResponse(reply=f"Error calling Gemini: {str(e)}")


# =============================================================================
# ENDPOINTS: Strategy Runner (Agent Tool)
# =============================================================================

class RunStrategyRequest(BaseModel):
    strategy: str = "opening_range"  # Strategy name
    start_date: str = "2025-03-17"
    weeks: int = 3
    run_name: Optional[str] = None


@app.post("/agent/run-strategy")
async def run_strategy(request: RunStrategyRequest) -> Dict[str, Any]:
    """
    Run a strategy and create a new dataset.
    This allows the agent to create data directly from the chat.
    """
    import subprocess
    from datetime import datetime
    
    # Generate run name if not provided
    run_name = request.run_name or f"{request.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Map strategy to script
    scripts = {
        "opening_range": "scripts/run_or_multi_oco.py",
        "or": "scripts/run_or_multi_oco.py",
    }
    
    script = scripts.get(request.strategy)
    if not script:
        return {"success": False, "error": f"Unknown strategy: {request.strategy}"}
    
    # Build command
    out_dir = RESULTS_DIR / run_name
    cmd = [
        "python", script,
        "--start-date", request.start_date,
        "--weeks", str(request.weeks),
        "--out", str(out_dir)
    ]
    
    try:
        # Run strategy (blocking for now, could make async)
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent.parent),  # mlang2 root
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "run_id": run_name,
                "message": f"Strategy '{request.strategy}' completed. Load run '{run_name}' to view results.",
                "output": result.stdout[-500:] if result.stdout else ""
            }
        else:
            return {
                "success": False,
                "error": result.stderr[-500:] if result.stderr else "Unknown error"
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Strategy timed out (>120s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health():
    runs = await list_runs()
    return {
        "status": "ok", 
        "results_dir": str(RESULTS_DIR),
        "available_runs": runs
    }
