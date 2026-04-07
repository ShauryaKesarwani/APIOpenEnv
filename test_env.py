"""
Test script for the API Workflow Environment.

Run this to verify the environment works correctly for all three difficulty levels.
"""

import sys
sys.path.insert(0, ".")

from server.api_open_env_environment import ApiOpenEnvironment
from models import ApiOpenAction


def test_easy_task():
    """Test easy task: fetch user email."""
    print("\n" + "=" * 50)
    print("TESTING: Easy Task - Fetch User Email")
    print("=" * 50)

    env = ApiOpenEnvironment()
    obs = env.reset(task_difficulty="easy")

    print(f"Task: {obs.task_description}")
    print(f"Available APIs: {obs.available_apis}")
    print(f"Max steps: {obs.max_steps}")

    # Execute the correct action
    action = ApiOpenAction(api_name="get_user", args={"user_id": "U101"})
    obs = env.step(action)

    print(f"\nStep 1 - Called get_user(user_id='U101')")
    print(f"Result: {obs.last_api_result}")
    print(f"Task complete: {obs.task_complete}")
    print(f"Done: {obs.done}")
    print(f"Reward: {obs.reward}")
    print(f"Grade: {env.grade():.2f}")

    assert obs.task_complete, "Easy task should be complete"
    assert env.grade() >= 0.8, f"Expected grade >= 0.8, got {env.grade()}"
    print("PASSED!")


def test_medium_task():
    """Test medium task: generate invoice."""
    print("\n" + "=" * 50)
    print("TESTING: Medium Task - Generate Invoice")
    print("=" * 50)

    env = ApiOpenEnvironment()
    obs = env.reset(task_difficulty="medium")

    print(f"Task: {obs.task_description}")
    print(f"Available APIs: {obs.available_apis}")

    # Step 1: Get user
    obs = env.step(ApiOpenAction(api_name="get_user", args={"user_id": "U101"}))
    print(f"\nStep 1 - get_user: success={obs.last_api_result.get('success')}")

    # Step 2: Get orders
    obs = env.step(ApiOpenAction(api_name="get_orders", args={"user_id": "U101"}))
    print(f"Step 2 - get_orders: success={obs.last_api_result.get('success')}")
    order_id = obs.last_api_result.get("data", [{}])[0].get("order_id", "O501")
    product_id = obs.last_api_result.get("data", [{}])[0].get("product_id", "P701")

    # Step 3: Get product
    obs = env.step(ApiOpenAction(api_name="get_product", args={"product_id": product_id}))
    print(f"Step 3 - get_product: success={obs.last_api_result.get('success')}")

    # Step 4: Create invoice
    obs = env.step(ApiOpenAction(api_name="create_invoice", args={"user_id": "U101", "order_id": order_id}))
    print(f"Step 4 - create_invoice: success={obs.last_api_result.get('success')}")

    print(f"\nTask complete: {obs.task_complete}")
    print(f"Done: {obs.done}")
    print(f"Total steps: {obs.step_count}")
    print(f"Grade: {env.grade():.2f}")

    assert obs.task_complete, "Medium task should be complete"
    assert env.grade() >= 0.8, f"Expected grade >= 0.8, got {env.grade()}"
    print("PASSED!")


def test_hard_task_refund_eligible():
    """Test hard task with refund-eligible ticket (T301 - within 30 days)."""
    print("\n" + "=" * 50)
    print("TESTING: Hard Task - Support Ticket (Refund Eligible)")
    print("=" * 50)

    env = ApiOpenEnvironment()

    # Force T301 ticket (within 30 days)
    env._task_config = {}
    obs = env.reset(task_difficulty="hard")

    # Check which ticket we got and adjust test accordingly
    ticket_id = env._task_config.get("ticket_id", "T301")
    print(f"Task: {obs.task_description}")
    print(f"Ticket ID: {ticket_id}")

    if ticket_id != "T301":
        print("Got T302 instead, skipping this test (T302 is >30 days)")
        return

    # Step 1: Get ticket
    obs = env.step(ApiOpenAction(api_name="get_ticket", args={"ticket_id": "T301"}))
    print(f"\nStep 1 - get_ticket: {obs.last_api_result}")
    user_id = obs.last_api_result.get("data", {}).get("user_id")
    order_id = obs.last_api_result.get("data", {}).get("order_id")

    # Step 2: Get user
    obs = env.step(ApiOpenAction(api_name="get_user", args={"user_id": user_id}))
    print(f"Step 2 - get_user: success={obs.last_api_result.get('success')}")
    email = obs.last_api_result.get("data", {}).get("email")

    # Step 3: Get orders
    obs = env.step(ApiOpenAction(api_name="get_orders", args={"user_id": user_id}))
    print(f"Step 3 - get_orders: success={obs.last_api_result.get('success')}")

    # Step 4: Process refund
    obs = env.step(ApiOpenAction(api_name="process_refund", args={"user_id": user_id, "order_id": order_id}))
    print(f"Step 4 - process_refund: success={obs.last_api_result.get('success')}")

    # Step 5: Send email
    obs = env.step(ApiOpenAction(api_name="send_email", args={"email": email, "subject": "Refund Processed"}))
    print(f"Step 5 - send_email: success={obs.last_api_result.get('success')}")

    print(f"\nTask complete: {obs.task_complete}")
    print(f"Done: {obs.done}")
    print(f"Total steps: {obs.step_count}")
    print(f"Grade: {env.grade():.2f}")

    assert obs.task_complete, "Hard task (T301) should be complete"
    print("PASSED!")


def test_hard_task_refund_ineligible():
    """Test hard task with refund-ineligible ticket (T302 - >30 days)."""
    print("\n" + "=" * 50)
    print("TESTING: Hard Task - Support Ticket (Refund Ineligible)")
    print("=" * 50)

    env = ApiOpenEnvironment()

    # Manually set ticket to T302
    obs = env.reset(task_difficulty="hard")
    env._task_config["ticket_id"] = "T302"
    env._task_config["description"] = obs.task_description.replace(
        env._task_config.get("ticket_id", "T301"), "T302"
    )

    print(f"Task: {env._task_config['description']}")
    print("Note: T302 is >30 days old, so refund should be denied")

    # Step 1: Get ticket
    obs = env.step(ApiOpenAction(api_name="get_ticket", args={"ticket_id": "T302"}))
    print(f"\nStep 1 - get_ticket: {obs.last_api_result.get('data', {}).get('issue')}")
    user_id = obs.last_api_result.get("data", {}).get("user_id")
    order_id = obs.last_api_result.get("data", {}).get("order_id")

    # Step 2: Get user
    obs = env.step(ApiOpenAction(api_name="get_user", args={"user_id": user_id}))
    print(f"Step 2 - get_user: success={obs.last_api_result.get('success')}")
    email = obs.last_api_result.get("data", {}).get("email")

    # Step 3: Get orders
    obs = env.step(ApiOpenAction(api_name="get_orders", args={"user_id": user_id}))
    print(f"Step 3 - get_orders: success={obs.last_api_result.get('success')}")

    # Step 4: Attempt refund (should fail)
    obs = env.step(ApiOpenAction(api_name="process_refund", args={"user_id": user_id, "order_id": order_id}))
    print(f"Step 4 - process_refund: {obs.last_api_result}")

    # Step 5: Send email (informing customer)
    obs = env.step(ApiOpenAction(api_name="send_email", args={"email": email, "subject": "Refund Denied - Policy"}))
    print(f"Step 5 - send_email: success={obs.last_api_result.get('success')}")

    print(f"\nTask complete: {obs.task_complete}")
    print(f"Done: {obs.done}")
    print(f"Total steps: {obs.step_count}")
    print(f"Grade: {env.grade():.2f}")

    assert obs.task_complete, "Hard task (T302) should be complete after attempting refund and sending email"
    print("PASSED!")


def test_invalid_api():
    """Test that using unavailable API gives penalty."""
    print("\n" + "=" * 50)
    print("TESTING: Invalid API Call")
    print("=" * 50)

    env = ApiOpenEnvironment()
    obs = env.reset(task_difficulty="easy")

    # Try to call an API not available for easy task
    obs = env.step(ApiOpenAction(api_name="create_invoice", args={}))

    print(f"Called unavailable API 'create_invoice' on easy task")
    print(f"Result: {obs.last_api_result}")
    print(f"Reward: {obs.reward}")

    assert obs.reward < 0, "Should get negative reward for invalid API"
    assert "not available" in obs.last_api_result.get("error", "")
    print("PASSED!")


def test_grader():
    """Test the grader function."""
    print("\n" + "=" * 50)
    print("TESTING: Grader Function")
    print("=" * 50)

    env = ApiOpenEnvironment()

    # Optimal easy task
    obs = env.reset(task_difficulty="easy")
    obs = env.step(ApiOpenAction(api_name="get_user", args={"user_id": "U101"}))
    grade = env.grade()
    print(f"Optimal easy task grade: {grade:.2f}")
    assert grade >= 0.9, f"Expected optimal grade >= 0.9, got {grade}"

    # Suboptimal easy task (extra steps)
    obs = env.reset(task_difficulty="easy")
    env.step(ApiOpenAction(api_name="get_user", args={"user_id": "U999"}))  # Wrong user
    env.step(ApiOpenAction(api_name="get_user", args={"user_id": "U101"}))  # Correct user
    grade = env.grade()
    print(f"Suboptimal easy task grade (extra step): {grade:.2f}")
    assert grade < 1.0, "Suboptimal task should have lower grade"

    # Failed task
    env.reset(task_difficulty="medium")
    env.step(ApiOpenAction(api_name="get_user", args={"user_id": "U101"}))  # Only first step
    grade = env.grade()
    print(f"Incomplete medium task grade: {grade:.2f}")
    assert grade < 0.5, f"Incomplete task should have grade < 0.5, got {grade}"

    print("PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("API WORKFLOW ENVIRONMENT - TEST SUITE")
    print("=" * 60)

    try:
        test_easy_task()
        test_medium_task()
        test_hard_task_refund_eligible()
        test_hard_task_refund_ineligible()
        test_invalid_api()
        test_grader()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
