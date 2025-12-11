class Supervisor:
    def __init__(self):
        self.status = "idle"

    def coordinate_workflow(self, input_data):
        self.status = "running"
        return f"Workflow started with {input_data}"