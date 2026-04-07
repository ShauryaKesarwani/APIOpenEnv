"""
Test script for the API Workflow Environment.

Demonstrates all three tasks: easy, medium, and hard.
"""

from server.api_open_env_environment import ApiOpenEnvironment
from models import ApiOpenAction
from server.grader import grade_episode


def test_task_easy():
    """Test Task 1 (Easy): Fetch user email."""
    print("\n" + "=" * 60)
    print("Testing Task 1 (Easy): Fetch user email")
    print("=" * 60)

    env = ApiOpenEnvironment()
    obs = env.reset(task_difficulty="easy")

    print(f"\nTask: {obs.task_description}")
    print(f"Available APIs: {obs.available_apis}")
    print(f"Max steps: {obs.max_steps}")

    # Extract user_id from task description
    user_id = obs.task_description.split()[-1]
    print(f"\nAgent action: Calling get_user(user_id='{user_id}')")

    # Agent takes action
    action = ApiOpenAction(api_name="get_user", args={"user_id": user_id})
    obs = env.step(action)

    print(f"API Result: {obs.last_api_result}")
    print(f"Task Complete: {obs.task_complete}")
    print(f"Reward: {obs.reward}")

    # Grade the episode
    grade = grade_episode(
        task_config=env._task_config,
        api_call_history=obs.api_call_history,
        task_complete=obs.task_complete,
    )
    print(f"\nGrade: {grade['score']:.3f}")
    print(f"Feedback: {grade['feedback']}")

    return obs.task_complete


def test_task_medium():
    """Test Task 2 (Medium): Generate invoice."""
    print("\n" + "=" * 60)
    print("Testing Task 2 (Medium): Generate invoice")
    print("=" * 60)

    env = ApiOpenEnvironment()
    obs = env.reset(task_difficulty="medium")

    print(f"\nTask: {obs.task_description}")
    print(f"Available APIs: {obs.available_apis}")
    print(f"Max steps: {obs.max_steps}")

    # Extract user_id from task description
    user_id = obs.task_description.split()[-1]

    # Step 1: Get user
    print(f"\nStep 1: get_user(user_id='{user_id}')")
    action = ApiOpenAction(api_name="get_user", args={"user_id": user_id})
    obs = env.step(action)
    print(f"Result: {obs.last_api_result}")

    # Step 2: Get orders
    print(f"\nStep 2: get_orders(user_id='{user_id}')")
    action = ApiOpenAction(api_name="get_orders", args={"user_id": user_id})
    obs = env.step(action)
    print(f"Result: {obs.last_api_result}")

    # Extract order and product info
    if obs.last_api_result and obs.last_api_result.get("success"):
        order_data = obs.last_api_result["data"][0]
        order_id = order_data["order_id"]
        product_id = order_data["product_id"]

        # Step 3: Get product
        print(f"\nStep 3: get_product(product_id='{product_id}')")
        action = ApiOpenAction(api_name="get_product", args={"product_id": product_id})
        obs = env.step(action)
        print(f"Result: {obs.last_api_result}")

        # Step 4: Create invoice
        print(f"\nStep 4: create_invoice(user_id='{user_id}', order_id='{order_id}')")
        action = ApiOpenAction(api_name="create_invoice", args={"user_id": user_id, "order_id": order_id})
        obs = env.step(action)
        print(f"Result: {obs.last_api_result}")

    print(f"\nTask Complete: {obs.task_complete}")
    print(f"Total Reward: {obs.reward}")

    # Grade the episode
    grade = grade_episode(
        task_config=env._task_config,
        api_call_history=obs.api_call_history,
        task_complete=obs.task_complete,
    )
    print(f"\nGrade: {grade['score']:.3f}")
    print(f"Feedback: {grade['feedback']}")

    return obs.task_complete


def test_task_hard():
    """Test Task 3 (Hard): Resolve support ticket with refund."""
    print("\n" + "=" * 60)
    print("Testing Task 3 (Hard): Resolve support ticket with refund")
    print("=" * 60)

    env = ApiOpenEnvironment()
    obs = env.reset(task_difficulty="hard")

    print(f"\nTask: {obs.task_description}")
    print(f"Available APIs: {obs.available_apis}")
    print(f"Max steps: {obs.max_steps}")

    # Extract ticket_id from task description
    words = obs.task_description.split()
    ticket_id = words[words.index("ticket") + 1]

    # Step 1: Get ticket
    print(f"\nStep 1: get_ticket(ticket_id='{ticket_id}')")
    action = ApiOpenAction(api_name="get_ticket", args={"ticket_id": ticket_id})
    obs = env.step(action)
    print(f"Result: {obs.last_api_result}")

    if obs.last_api_result and obs.last_api_result.get("success"):
        ticket_data = obs.last_api_result["data"]
        user_id = ticket_data["user_id"]
        order_id = ticket_data["order_id"]

        # Step 2: Get user
        print(f"\nStep 2: get_user(user_id='{user_id}')")
        action = ApiOpenAction(api_name="get_user", args={"user_id": user_id})
        obs = env.step(action)
        print(f"Result: {obs.last_api_result}")
        user_email = obs.last_api_result["data"]["email"]

        # Step 3: Get orders (for verification)
        print(f"\nStep 3: get_orders(user_id='{user_id}')")
        action = ApiOpenAction(api_name="get_orders", args={"user_id": user_id})
        obs = env.step(action)
        print(f"Result: {obs.last_api_result}")

        # Step 4: Process refund
        print(f"\nStep 4: process_refund(user_id='{user_id}', order_id='{order_id}')")
        action = ApiOpenAction(api_name="process_refund", args={"user_id": user_id, "order_id": order_id})
        obs = env.step(action)
        print(f"Result: {obs.last_api_result}")

        # Step 5: Send email
        print(f"\nStep 5: send_email(email='{user_email}')")
        action = ApiOpenAction(
            api_name="send_email",
            args={"email": user_email, "subject": "Refund Processed", "body": "Your refund has been processed."},
        )
        obs = env.step(action)
        print(f"Result: {obs.last_api_result}")

    print(f"\nTask Complete: {obs.task_complete}")
    print(f"Total Reward: {obs.reward}")

    # Grade the episode
    grade = grade_episode(
        task_config=env._task_config,
        api_call_history=obs.api_call_history,
        task_complete=obs.task_complete,
    )
    print(f"\nGrade: {grade['score']:.3f}")
    print(f"Feedback: {grade['feedback']}")
    print(f"Metrics: {grade['metrics']}")

    return obs.task_complete


def main():
    """Run all tests."""
    print("\n🚀 API Workflow Environment Test Suite")
    print("=" * 60)

    results = {}
    results["easy"] = test_task_easy()
    results["medium"] = test_task_medium()
    results["hard"] = test_task_hard()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Task 1 (Easy):   {'✓ PASSED' if results['easy'] else '✗ FAILED'}")
    print(f"Task 2 (Medium): {'✓ PASSED' if results['medium'] else '✗ FAILED'}")
    print(f"Task 3 (Hard):   {'✓ PASSED' if results['hard'] else '✗ FAILED'}")

    all_passed = all(results.values())
    print(f"\n{'✓ All tests passed!' if all_passed else '✗ Some tests failed.'}")


if __name__ == "__main__":
    main()
