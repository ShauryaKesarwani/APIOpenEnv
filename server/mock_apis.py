"""
Mock APIs for the API Workflow Environment.

These APIs simulate a real-world backend system with users, orders, products, tickets, etc.
The agent must call these APIs in the correct sequence to complete tasks.
"""

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

MOCK_DB_PATH = Path(__file__).with_name("mock_db.json")


def _load_mock_db_template() -> Dict[str, Any]:
    """Load static mock records from JSON on disk."""
    with MOCK_DB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _hydrate_relative_dates(db_template: Dict[str, Any]) -> Dict[str, Any]:
    """Convert relative day offsets into ISO timestamps for runtime use."""
    db = copy.deepcopy(db_template)
    now = datetime.now()

    for order in db.get("orders", {}).values():
        days_ago = int(order.pop("days_ago", 0))
        order["date"] = (now - timedelta(days=days_ago)).isoformat()

    for ticket in db.get("tickets", {}).values():
        days_ago = int(ticket.pop("days_ago", 0))
        ticket["created"] = (now - timedelta(days=days_ago)).isoformat()

    # Runtime collections are always resettable mutable containers.
    db["invoices"] = {}
    db["emails_sent"] = []
    db["refunds_processed"] = []

    return db


def _load_runtime_mock_db() -> Dict[str, Any]:
    return _hydrate_relative_dates(_load_mock_db_template())


# Runtime mock database (re-hydrated on every environment reset).
MOCK_DB = _load_runtime_mock_db()


def get_user(user_id: str) -> Dict[str, Any]:
    """
    Retrieve user information by user ID.

    Args:
        user_id: The user ID to look up

    Returns:
        User information dict or error
    """
    if user_id in MOCK_DB["users"]:
        return {"success": True, "data": MOCK_DB["users"][user_id]}
    return {"success": False, "error": f"User {user_id} not found"}


def get_orders(user_id: str) -> Dict[str, Any]:
    """
    Retrieve all orders for a given user.

    Args:
        user_id: The user ID to look up orders for

    Returns:
        List of orders or error
    """
    orders = [order for order in MOCK_DB["orders"].values() if order["user_id"] == user_id]
    if orders:
        return {"success": True, "data": orders}
    return {"success": False, "error": f"No orders found for user {user_id}"}


def get_product(product_id: str) -> Dict[str, Any]:
    """
    Retrieve product information by product ID.

    Args:
        product_id: The product ID to look up

    Returns:
        Product information dict or error
    """
    if product_id in MOCK_DB["products"]:
        return {"success": True, "data": MOCK_DB["products"][product_id]}
    return {"success": False, "error": f"Product {product_id} not found"}


def create_invoice(user_id: str, order_id: str) -> Dict[str, Any]:
    """
    Create an invoice for a user's order.

    Args:
        user_id: The user ID
        order_id: The order ID

    Returns:
        Invoice information or error
    """
    if user_id not in MOCK_DB["users"]:
        return {"success": False, "error": f"User {user_id} not found"}

    if order_id not in MOCK_DB["orders"]:
        return {"success": False, "error": f"Order {order_id} not found"}

    order = MOCK_DB["orders"][order_id]
    if order["user_id"] != user_id:
        return {"success": False, "error": f"Order {order_id} does not belong to user {user_id}"}

    invoice_id = f"INV-{order_id}"
    invoice = {
        "invoice_id": invoice_id,
        "user_id": user_id,
        "order_id": order_id,
        "amount": order["amount"],
        "created": datetime.now().isoformat(),
        "status": "generated",
    }

    MOCK_DB["invoices"][invoice_id] = invoice
    return {"success": True, "data": invoice}


def send_email(email: str, subject: Optional[str] = None, body: Optional[str] = None) -> Dict[str, Any]:
    """
    Send an email to the specified address.

    Args:
        email: The recipient email address
        subject: Email subject (optional)
        body: Email body (optional)

    Returns:
        Confirmation of email sent
    """
    if not email or "@" not in email:
        return {"success": False, "error": "Invalid email address"}

    email_record = {
        "to": email,
        "subject": subject or "Notification",
        "body": body or "This is a notification email",
        "sent_at": datetime.now().isoformat(),
    }

    MOCK_DB["emails_sent"].append(email_record)
    return {"success": True, "data": {"message": f"Email sent to {email}", "record": email_record}}


def get_ticket(ticket_id: str) -> Dict[str, Any]:
    """
    Retrieve support ticket information.

    Args:
        ticket_id: The ticket ID to look up

    Returns:
        Ticket information dict or error
    """
    if ticket_id in MOCK_DB["tickets"]:
        return {"success": True, "data": MOCK_DB["tickets"][ticket_id]}
    return {"success": False, "error": f"Ticket {ticket_id} not found"}


def process_refund(user_id: str, order_id: str) -> Dict[str, Any]:
    """
    Process a refund for a user's order.
    Refunds are only valid if the order is within 30 days.

    Args:
        user_id: The user ID
        order_id: The order ID

    Returns:
        Refund confirmation or error
    """
    if user_id not in MOCK_DB["users"]:
        return {"success": False, "error": f"User {user_id} not found"}

    if order_id not in MOCK_DB["orders"]:
        return {"success": False, "error": f"Order {order_id} not found"}

    order = MOCK_DB["orders"][order_id]
    if order["user_id"] != user_id:
        return {"success": False, "error": f"Order {order_id} does not belong to user {user_id}"}

    # Check if order is within 30 days
    order_date = datetime.fromisoformat(order["date"])
    days_ago = (datetime.now() - order_date).days

    if days_ago > 30:
        return {
            "success": False,
            "error": f"Refund not allowed. Order is {days_ago} days old (policy: max 30 days)",
        }

    refund_id = f"REF-{order_id}"
    refund = {
        "refund_id": refund_id,
        "user_id": user_id,
        "order_id": order_id,
        "amount": order["amount"],
        "processed_at": datetime.now().isoformat(),
        "status": "completed",
    }

    MOCK_DB["refunds_processed"].append(refund)
    return {"success": True, "data": refund}


# API Registry - maps API names to functions
API_REGISTRY = {
    "get_user": get_user,
    "get_orders": get_orders,
    "get_product": get_product,
    "create_invoice": create_invoice,
    "send_email": send_email,
    "get_ticket": get_ticket,
    "process_refund": process_refund,
}


def call_api(api_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route API calls to the appropriate function.

    Args:
        api_name: Name of the API to call
        args: Arguments to pass to the API

    Returns:
        API result or error
    """
    if api_name not in API_REGISTRY:
        return {"success": False, "error": f"Unknown API: {api_name}"}

    try:
        func = API_REGISTRY[api_name]
        result = func(**args)
        return result
    except TypeError as e:
        return {"success": False, "error": f"Invalid arguments for {api_name}: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"API error: {str(e)}"}


def reset_mock_db():
    """Reset the mock database to initial state (useful for testing)."""
    global MOCK_DB
    MOCK_DB = _load_runtime_mock_db()
