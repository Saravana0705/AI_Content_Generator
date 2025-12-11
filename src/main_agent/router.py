class Router:
    def __init__(self):
        self.routes = {}

    def route_to_subagent(self, input_data, subagent_type):
        if subagent_type == "text_generator":
            return f"Routing to text_generator with {input_data}"
        return "No route found"