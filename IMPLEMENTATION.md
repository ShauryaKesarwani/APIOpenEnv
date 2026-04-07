# API Workflow Environment - Implementation Guide 🚀

**Complete implementation for the Scaler x Meta x PyTorch OpenEnv Hackathon**

---

## ✅ Implementation Complete

All core components have been successfully implemented:

1. ✅ **Mock APIs** (`server/mock_apis.py`)
2. ✅ **Pydantic Models** (`models.py`)
3. ✅ **Environment Logic** (`server/api_open_env_environment.py`)
4. ✅ **Grader** (`server/grader.py`)
5. ✅ **Test Suite** (`test_environment.py`)

---

## 🎯 What Was Built

### 1. Mock API System (`server/mock_apis.py`)

**7 fully functional APIs**:
- `get_user(user_id)` - Fetch user information
- `get_orders(user_id)` - Retrieve user's orders
- `get_product(product_id)` - Get product details
- `create_invoice(user_id, order_id)` - Generate invoice
- `send_email(email, subject, body)` - Send notifications
- `get_ticket(ticket_id)` - Get support ticket details
- `process_refund(user_id, order_id)` - Process refund (30-day constraint)

**Mock Database**:
- 5 users (U101-U105)
- 5 orders (O501-O505) with varying dates
- 5 products (P701-P705)
- 2 support tickets (T301: <30 days, T302: >30 days)

---

### 2. OpenEnv Models (`models.py`)

**ApiOpenAction**:
```python
{
    "api_name": "get_user",
    "args": {"user_id": "U101"}
}
```

**ApiOpenObservation**:
```python
{
    "task_description": "Generate invoice for user U101",
    "available_apis": ["get_user", "get_orders", ...],
    "last_api_result": {"success": True, "data": {...}},
    "api_call_history": [...],
    "step_count": 3,
    "max_steps": 8,
    "task_complete": False,
    "done": False,
    "reward": 0.5
}
```

---

### 3. Environment Logic (`server/api_open_env_environment.py`)

**Key Features**:
- `reset(task_difficulty)` - Initialize environment with task
- `step(action)` - Execute API call and update state
- Automatic task completion detection
- Reward calculation based on API success
- Built-in `grade()` method (0.0-1.0 score)

**Task Success Conditions**:
- **Easy**: Successfully call `get_user`
- **Medium**: Successfully call `create_invoice`
- **Hard**: Successfully call `process_refund` AND `send_email`

**Reward System**:
- +0.2: Successful API call
- +0.3: Critical API (invoice/refund)
- +1.0: Task completion bonus
- -0.1: Failed API call
- -0.3: Invalid API for task

---

### 4. Grading System (`server/grader.py`)

**Scoring Formula** (0.0 - 1.0):
- 50% - Task completion
- 30% - Efficiency (fewer unnecessary calls)
- 20% - API sequence correctness

**Example Grades**:
- 1.0: Perfect - optimal path, no mistakes
- 0.9: Excellent - 1 extra call
- 0.8: Good - 2-3 extra calls
- 0.6: Poor - many wasted calls
- 0.0: Failed - task incomplete

---

## 🧪 Testing the Implementation

### Run the Test Suite
```bash
python test_environment.py
```

**Expected Output**:
```
🚀 API Workflow Environment Test Suite
============================================================

Testing Task 1 (Easy): Fetch user email
Task: Fetch user email for user U101
✓ PASSED

Testing Task 2 (Medium): Generate invoice
Task: Generate invoice for user U102
✓ PASSED

Testing Task 3 (Hard): Resolve support ticket
Task: Resolve support ticket T301
✓ PASSED

Summary
============================================================
Task 1 (Easy):   ✓ PASSED
Task 2 (Medium): ✓ PASSED
Task 3 (Hard):   ✓ PASSED

✓ All tests passed!
```

### Quick Import Verification
```bash
python check_imports.py
```

---

## 📊 Task Details

### Task 1 (Easy): Fetch User Email

**Scenario**: Agent needs to retrieve user's email address

**Perfect Solution**:
```python
env = ApiOpenEnvironment()
obs = env.reset(task_difficulty="easy")

# Extract user_id from task description (e.g., "Fetch user email for user U101")
user_id = obs.task_description.split()[-1]

# Make API call
action = ApiOpenAction(api_name="get_user", args={"user_id": user_id})
obs = env.step(action)

# Task complete!
assert obs.task_complete == True
assert env.grade() == 1.0
```

**Key Learning**: Single API call, straightforward

---

### Task 2 (Medium): Generate Invoice

**Scenario**: Agent must create an invoice for a user's order

**Perfect Solution**:
```python
env = ApiOpenEnvironment()
obs = env.reset(task_difficulty="medium")
user_id = obs.task_description.split()[-1]

# Step 1: Get user info
obs = env.step(ApiOpenAction(api_name="get_user", args={"user_id": user_id}))

# Step 2: Get user's orders
obs = env.step(ApiOpenAction(api_name="get_orders", args={"user_id": user_id}))
order_id = obs.last_api_result["data"][0]["order_id"]
product_id = obs.last_api_result["data"][0]["product_id"]

# Step 3: Get product details
obs = env.step(ApiOpenAction(api_name="get_product", args={"product_id": product_id}))

# Step 4: Create invoice
obs = env.step(ApiOpenAction(
    api_name="create_invoice",
    args={"user_id": user_id, "order_id": order_id}
))

# Task complete!
assert obs.task_complete == True
assert env.grade() >= 0.9  # Optimal path
```

**Key Learning**: Multi-step workflow, must chain API results

---

### Task 3 (Hard): Resolve Support Ticket

**Scenario**: Customer opened support ticket, agent must process refund and notify

**Constraint**: Refunds only allowed if order is ≤30 days old

**Perfect Solution**:
```python
env = ApiOpenEnvironment()
obs = env.reset(task_difficulty="hard")

# Extract ticket_id from task description
words = obs.task_description.split()
ticket_id = words[words.index("ticket") + 1]

# Step 1: Get ticket details
obs = env.step(ApiOpenAction(api_name="get_ticket", args={"ticket_id": ticket_id}))
ticket = obs.last_api_result["data"]
user_id = ticket["user_id"]
order_id = ticket["order_id"]

# Step 2: Get user info
obs = env.step(ApiOpenAction(api_name="get_user", args={"user_id": user_id}))
email = obs.last_api_result["data"]["email"]

# Step 3: Get orders (for verification)
obs = env.step(ApiOpenAction(api_name="get_orders", args={"user_id": user_id}))

# Step 4: Attempt refund
obs = env.step(ApiOpenAction(
    api_name="process_refund",
    args={"user_id": user_id, "order_id": order_id}
))

# Step 5: Send confirmation email
obs = env.step(ApiOpenAction(
    api_name="send_email",
    args={
        "email": email,
        "subject": "Refund Status",
        "body": "Your refund request has been processed."
    }
))

# Task complete!
assert obs.task_complete == True
assert env.grade() >= 0.9
```

**Key Learning**: Complex workflow with constraints, must handle both success and failure cases

---

## 🎮 Running the Environment Server

### Start the server
```bash
# Using OpenEnv CLI
openenv server start

# OR using uvicorn directly
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Access the environment
- **API Docs**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000/web
- **Health Check**: http://localhost:8000/health

---

## 🧠 Why This Environment is Interesting

### 1. Real-World Simulation
Not a toy problem - mirrors actual:
- Customer support systems
- Billing pipelines
- Backend automation workflows

### 2. Multi-Step Reasoning
Agent must:
- Plan ahead (what info do I need?)
- Chain results (use output of API A in API B)
- Handle constraints (30-day refund policy)

### 3. Partial Observability
- No direct database access
- Must query APIs to gather information
- Forces sequential exploration

### 4. Clear Success Metrics
- Deterministic grading (0.0-1.0)
- Multiple difficulty levels
- Measurable progress

---

## 📈 Expected Agent Behavior

### Novice Agent (Random)
- Random API calls
- Ignores task description
- Score: ~0.0 (never completes)

### Learning Agent
- Tries different sequences
- Learns which APIs are useful
- Score: 0.3-0.6 (sometimes completes)

### Proficient Agent
- Parses task description
- Extracts relevant IDs
- Chains APIs correctly
- Score: 0.8-1.0 (optimal paths)

---

## 🔍 File-by-File Summary

### `server/mock_apis.py` (270 lines)
- Mock database with users, orders, products, tickets
- 7 API functions with error handling
- `call_api()` routing function
- `reset_mock_db()` for testing

### `models.py` (60 lines)
- `ApiOpenAction`: API name + arguments
- `ApiOpenObservation`: Full state observation

### `server/api_open_env_environment.py` (240 lines)
- `ApiOpenEnvironment` class
- Task definitions (easy, medium, hard)
- `reset()` and `step()` methods
- Built-in `grade()` scoring

### `server/grader.py` (200 lines)
- `grade_episode()` - comprehensive scoring
- Task-specific graders
- Detailed feedback and metrics

### `test_environment.py` (180 lines)
- End-to-end tests for all 3 tasks
- Demonstrates perfect solutions
- Validates grading system

---

## ✨ Next Steps for Hackathon

1. **Test thoroughly**: Run `python test_environment.py`
2. **Start server**: `openenv server start`
3. **Train an agent**: Use OpenEnv SDK
4. **Document results**: Show agent learning curves
5. **Package for submission**: Include README and examples

---

## 🏆 Evaluation Criteria Met

✅ **Real-world environment** - Simulates actual API workflows
✅ **Multi-step tasks** - Requires planning and chaining
✅ **OpenEnv interface** - Proper reset(), step(), state()
✅ **Pydantic models** - Action and Observation
✅ **3 difficulty levels** - Easy, Medium, Hard
✅ **Grader included** - Deterministic 0.0-1.0 scoring
✅ **Clean code** - Well-documented and testable
✅ **Working tests** - Validates all functionality

---

**Implementation Status: COMPLETE ✅**

Built with clean, minimal, working code focused on correctness and real-world mapping.
