import unittest
from main_agent.interface import Interface
from main_agent.router import Router
from main_agent.supervisor import Supervisor


class TestMainAgent(unittest.TestCase):
    def setUp(self):
        self.interface = Interface()
        self.router = Router()
        self.supervisor = Supervisor()

    # -------- Interface tests --------
    def test_get_user_input_stores_and_returns_value(self):
        input_text = "Test input"
        result = self.interface.get_user_input(input_text)

        # It should return the same text
        self.assertEqual(result, input_text)
        # It should also store it internally
        self.assertEqual(self.interface.input_data, input_text)

    def test_get_user_input_with_empty_string(self):
        input_text = ""
        result = self.interface.get_user_input(input_text)

        self.assertEqual(result, "")
        self.assertEqual(self.interface.input_data, "")

    def test_get_user_input_with_whitespace(self):
        input_text = "   spaced   "
        result = self.interface.get_user_input(input_text)

        self.assertEqual(result, input_text)
        self.assertEqual(self.interface.input_data, input_text)

    # -------- Router tests --------
    def test_route_to_text_generator(self):
        input_data = "Some marketing brief"
        result = self.router.route_to_subagent(input_data, "text_generator")

        expected = f"Routing to text_generator with {input_data}"
        self.assertEqual(result, expected)

    def test_route_to_unknown_subagent(self):
        input_data = "Some marketing brief"
        result = self.router.route_to_subagent(input_data, "image_generator")

        self.assertEqual(result, "No route found")

    def test_router_starts_with_empty_routes_dict(self):
        self.assertIsInstance(self.router.routes, dict)
        self.assertEqual(self.router.routes, {})

    # -------- Supervisor tests --------
    def test_supervisor_initial_status_is_idle(self):
        self.assertEqual(self.supervisor.status, "idle")

    def test_coordinate_workflow_sets_status_and_returns_message(self):
        input_data = "Workflow data"
        result = self.supervisor.coordinate_workflow(input_data)

        # Status should change from 'idle' to 'running'
        self.assertEqual(self.supervisor.status, "running")

        expected = f"Workflow started with {input_data}"
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
