# billing_routes.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from db import get_session
from models import Invoice, Customer, Subscription, Payment

router = APIRouter(prefix="/api", tags=["billing"])

def _match(q: str, *values: str) -> bool:
  ql = q.strip().lower()
  return any(ql in (v or "").lower() for v in values)

@router.get("/invoices", response_model=List[Invoice])
def list_invoices(q: Optional[str] = None, session: Session = Depends(get_session)):
  rows = session.exec(select(Invoice)).all()
  if not q:
    return rows
  return [r for r in rows if _match(q, r.id, r.customer, r.status, r.method)]

@router.post("/invoices", response_model=Invoice)
def create_invoice(inv: Invoice, session: Session = Depends(get_session)):
  exists = session.get(Invoice, inv.id)
  if exists:
    raise HTTPException(status_code=409, detail="Invoice id already exists")
  session.add(inv)
  session.commit()
  session.refresh(inv)
  return inv

@router.post("/invoices/{invoice_id}/pay")
def pay_invoice(invoice_id: str, method: str = "UPI", session: Session = Depends(get_session)):
  inv = session.get(Invoice, invoice_id)
  if not inv:
    raise HTTPException(status_code=404, detail="Invoice not found")

  inv.status = "paid"
  inv.method = method

  payment = Payment(invoice_id=invoice_id, amount=inv.amount, method=method)
  session.add(payment)
  session.add(inv)
  session.commit()

  return {"ok": True, "invoice_id": invoice_id}

@router.get("/customers", response_model=List[Customer])
def list_customers(q: Optional[str] = None, session: Session = Depends(get_session)):
  rows = session.exec(select(Customer)).all()
  if not q:
    return rows
  return [r for r in rows if _match(q, r.id, r.name, r.tier, r.status)]

@router.post("/customers", response_model=Customer)
def create_customer(c: Customer, session: Session = Depends(get_session)):
  exists = session.get(Customer, c.id)
  if exists:
    raise HTTPException(status_code=409, detail="Customer id already exists")
  session.add(c)
  session.commit()
  session.refresh(c)
  return c

@router.get("/subscriptions", response_model=List[Subscription])
def list_subscriptions(q: Optional[str] = None, session: Session = Depends(get_session)):
  rows = session.exec(select(Subscription)).all()
  if not q:
    return rows
  return [r for r in rows if _match(q, r.id, r.plan, r.customer, r.status)]

@router.post("/seed")
def seed_if_empty(session: Session = Depends(get_session)):
  # Seed only if DB is empty
  any_invoice = session.exec(select(Invoice)).first()
  if any_invoice:
    return {"ok": True, "seeded": False}

  from datetime import date

  session.add_all([
    Invoice(id="INV-10428", customer="Apex Retail Pvt Ltd", amount=48900, currency="INR", status="paid", due=date(2025,12,2), created=date(2025,11,25), method="UPI"),
    Invoice(id="INV-10429", customer="BlueSky Logistics", amount=125000, currency="INR", status="overdue", due=date(2025,12,8), created=date(2025,11,28), method="Card"),
    Invoice(id="INV-10430", customer="Nimbus Clinics", amount=76000, currency="INR", status="sent", due=date(2025,12,20), created=date(2025,12,1), method="NetBanking"),
  ])

  session.add_all([
    Subscription(id="SUB-2201", plan="Growth", customer="Nimbus Clinics", mrr=6999, status="active"),
    Subscription(id="SUB-2202", plan="Starter", customer="Orchid Education", mrr=1999, status="active"),
  ])

  session.add_all([
    Customer(id="CUST-901", name="Apex Retail Pvt Ltd", tier="Mid-market", invoices=12, status="healthy"),
    Customer(id="CUST-902", name="BlueSky Logistics", tier="Enterprise", invoices=21, status="at_risk"),
  ])

  session.commit()
  return {"ok": True, "seeded": True}
