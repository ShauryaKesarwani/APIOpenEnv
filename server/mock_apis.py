"""
Mock APIs for the API Workflow Environment.

These APIs simulate a real-world backend system with users, orders, products, tickets, etc.
The agent must call these APIs in the correct sequence to complete tasks.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Mock Database
MOCK_DB = {
    "users": {
        "U101": {"user_id": "U101", "name": "Alice Smith", "email": "alice@example.com"},
        "U102": {"user_id": "U102", "name": "Bob Johnson", "email": "bob@example.com"},
        "U103": {"user_id": "U103", "name": "Charlie Brown", "email": "charlie@example.com"},
        "U104": {"user_id": "U104", "name": "Diana Prince", "email": "diana@example.com"},
        "U105": {"user_id": "U105", "name": "Eve Davis", "email": "eve@example.com"},
    },
    "orders": {
        "O501": {
            "order_id": "O501",
            "user_id": "U101",
            "product_id": "P701",
            "amount": 49.99,
            "date": (datetime.now() - timedelta(days=5)).isoformat(),
            "status": "completed",
        },
        "O502": {
            "order_id": "O502",
            "user_id": "U102",
            "product_id": "P702",
            "amount": 99.99,
            "date": (datetime.now() - timedelta(days=10)).isoformat(),
            "status": "completed",
        },
        "O503": {
            "order_id": "O503",
            "user_id": "U103",
            "product_id": "P703",
            "amount": 29.99,
            "date": (datetime.now() - timedelta(days=15)).isoformat(),
            "status": "completed",
        },
        "O504": {
            "order_id": "O504",
            "user_id": "U104",
            "product_id": "P704",
            "amount": 149.99,
            "date": (datetime.now() - timedelta(days=20)).isoformat(),
            "status": "completed",
        },
        "O505": {
            "order_id": "O505",
            "user_id": "U105",
            "product_id": "P705",
            "amount": 79.99,
            "date": (datetime.now() - timedelta(days=35)).isoformat(),
            "status": "completed",
        },
    },
    "products": {
        "P701": {"product_id": "P701", "name": "Laptop Pro", "price": 49.99, "category": "Electronics"},
        "P702": {"product_id": "P702", "name": "Wireless Mouse", "price": 99.99, "category": "Accessories"},
        "P703": {"product_id": "P703", "name": "USB-C Cable", "price": 29.99, "category": "Accessories"},
        "P704": {"product_id": "P704", "name": "Monitor 27\"", "price": 149.99, "category": "Electronics"},
        "P705": {"product_id": "P705", "name": "Keyboard Mechanical", "price": 79.99, "category": "Accessories"},
    },
    "tickets": {
        "T301": {
            "ticket_id": "T301",
            "user_id": "U103",
            "order_id": "O503",
            "issue": "Product arrived damaged",
            "status": "open",
            "created": (datetime.now() - timedelta(days=2)).isoformat(),
        },
        "T302": {
            "ticket_id": "T302",
            "user_id": "U105",
            "order_id": "O505",
            "issue": "Requesting refund - changed mind",
            "status": "open",
            "created": (datetime.now() - timedelta(days=40)).isoformat(),
        },
    },
    "invoices": {},
    "emails_sent": [],
    "refunds_processed": [],
}


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
    MOCK_DB["invoices"].clear()
    MOCK_DB["emails_sent"].clear()
    MOCK_DB["refunds_processed"].clear()
