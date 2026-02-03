import json
import os
import re
import tempfile
from typing import Any, Dict, Optional, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from faster_whisper import WhisperModel

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct").strip()
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small").strip()
CORS_ORIGINS = [
  x.strip()
  for x in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173").split(",")
  if x.strip()
]

app = FastAPI(title="FluxBill Backend", version="1.0.0")
app.add_middleware(
  CORSMiddleware,
  allow_origins=CORS_ORIGINS,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")

ACTIONS = [
  "click",
  "type",
  "none",

  # CRUD and filters
  "create_invoice",
  "delete_invoice",
  "update_invoice",
  "filter_invoices",

  "create_customer",
  "delete_customer",
  "update_customer",

  "create_subscription",
  "delete_subscription",
  "update_subscription",
]

TARGETS = [
  # navigation
  "nav.dashboard",
  "nav.invoices",
  "nav.subscriptions",
  "nav.customers",
  "nav.reports",
  "nav.settings",

  # ui fields/actions
  "field.search",
  "action.createInvoice",
  "action.collectPayment",
]

class AssistantTextRequest(BaseModel):
  text: str
  active_tab: str = "dashboard"


class Command(BaseModel):
  action: str = Field(default="none")
  target: Optional[str] = None
  args: Dict[str, Any] = Field(default_factory=dict)
  reply: str = "ok"


def _system_prompt() -> str:
  return f"""
You are an intent + entity extractor for a React billing dashboard.

Return ONLY valid JSON with this schema:
{{
  "action": "one of: {", ".join(ACTIONS)}",
  "target": "one of: {", ".join(TARGETS)} or null",
  "args": {{ "any": "key-values" }},
  "reply": "short human message"
}}

Rules:
- Use action="click" when the user wants to navigate tabs or press a UI button.
- Use action="type" only for search input, set target="field.search" and args={{"text":"..."}}.
- For create/update/delete/filter, put extracted fields inside args and set target=null unless you need a UI click too.
- If user intent is unclear, output action="none" and ask a short question in reply.
- Always respond in English.

Extraction formats:
- create_customer args: {{ "name": "...", "tier": "SMB|Mid-market|Enterprise", "status": "active|new|at_risk" }}
- delete_customer args: {{ "customer_id": "CUST-0901" }} OR {{ "name": "Apex Retail Pvt Ltd" }}
- create_invoice args: {{ "customer_id": "...", "customer_name": "...", "amount": 25000, "currency": "INR", "status": "draft|sent|paid|overdue" }}
- delete_invoice args: {{ "invoice_id": "INV-10431" }}
- filter_invoices args: {{ "amount_min": 10000, "amount_max": 50000, "status": "paid|sent|overdue|draft" }}

Examples:
User: open invoices
{{"action":"click","target":"nav.invoices","args":{{}},"reply":"opening invoices"}}

User: search apex
{{"action":"type","target":"field.search","args":{{"text":"apex"}},"reply":"searching apex"}}

User: create customer acme retail tier mid-market status at risk
{{"action":"create_customer","target":null,"args":{{"name":"Acme Retail","tier":"Mid-market","status":"at_risk"}},"reply":"added customer Acme Retail"}}

User: delete invoice INV-10431
{{"action":"delete_invoice","target":null,"args":{{"invoice_id":"INV-10431"}},"reply":"deleting invoice INV-10431"}}

User: filter invoices above 20000
{{"action":"filter_invoices","target":null,"args":{{"amount_min":20000}},"reply":"filtering invoices"}}
""".strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
  try:
    obj = json.loads(text)
    if isinstance(obj, dict):
      return obj
  except Exception:
    pass

  m = re.search(r"\{[\s\S]*\}", text)
  if not m:
    return None
  try:
    obj = json.loads(m.group(0))
    return obj if isinstance(obj, dict) else None
  except Exception:
    return None


async def plan_command(user_text: str, active_tab: str) -> Command:
  if not OPENROUTER_API_KEY:
    raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

  url = "https://openrouter.ai/api/v1/chat/completions"
  headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
  }

  payload = {
    "model": OPENROUTER_MODEL,
    "messages": [
      {"role": "system", "content": _system_prompt()},
      {"role": "user", "content": f"Active tab: {active_tab}\nUser: {user_text}"},
    ],
    "temperature": 0.1,
  }

  async with httpx.AsyncClient(timeout=60) as client:
    r = await client.post(url, headers=headers, json=payload)
    if r.status_code >= 400:
      raise HTTPException(status_code=500, detail=f"OpenRouter error: {r.status_code} {r.text}")

  content = r.json()["choices"][0]["message"]["content"]
  obj = _extract_json(content)
  if not obj:
    return Command(action="none", reply='I could not understand. Try: "open invoices", "search apex".')

  try:
    cmd = Command(**obj)
  except ValidationError:
    return Command(action="none", reply="I got an invalid command format. Try again.")

  # light guardrails
  if cmd.action not in ACTIONS:
    return Command(action="none", reply="That action is not supported.")
  if cmd.target is not None and cmd.target not in TARGETS:
    return Command(action="none", reply="That UI target is not supported.")

  # if typing, enforce args.text
  if cmd.action == "type":
    text = (cmd.args.get("text") or "").strip()
    if cmd.target != "field.search" or not text:
      return Command(action="none", reply="Tell me what to search for.")
    cmd.args["text"] = text

  return cmd


def transcribe_audio(file_path: str) -> str:
  segments, _info = whisper_model.transcribe(
    file_path,
    language="en",
    task="transcribe",
    vad_filter=True,
    initial_prompt="This is English. Transcribe only English words.",
  )
  parts = []
  for s in segments:
    if s.text:
      parts.append(s.text.strip())
  return " ".join([p for p in parts if p]).strip()


@app.get("/health")
def health():
  return {"ok": True, "whisper_model": WHISPER_MODEL_NAME, "openrouter_model": OPENROUTER_MODEL}


@app.post("/assistant/text")
async def assistant_text(req: AssistantTextRequest):
  cmd = await plan_command(req.text, req.active_tab)
  return {"transcript": req.text, "command": cmd.model_dump()}


@app.post("/assistant/voice")
async def assistant_voice(file: UploadFile = File(...), active_tab: str = Form("dashboard")):
  filename = file.filename or "voice"
  _, ext = os.path.splitext(filename)
  if not ext:
    ext = ".webm"

  with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    tmp_path = tmp.name
    tmp.write(await file.read())

  try:
    transcript = transcribe_audio(tmp_path)
    if not transcript:
      return {"transcript": "", "command": Command(action="none", reply="I could not hear anything. Try again.").model_dump()}

    cmd = await plan_command(transcript, active_tab)
    return {"transcript": transcript, "command": cmd.model_dump()}
  finally:
    try:
      os.remove(tmp_path)
    except Exception:
      pass



