from main_agent.interface import Interface
from main_agent.router import Router
from main_agent.supervisor import Supervisor
from sub_agents.text_generator.modules.generator.content_generator import Generator

def main():
    interface = Interface()
    router = Router()
    supervisor = Supervisor()
    generator = Generator()

    input_text = interface.get_user_input("Generate a blog post about digital marketing")
    routed_data = router.route_to_subagent(input_text, "text_generator")
    supervisor.coordinate_workflow(routed_data)
    content = generator.generate(input_text)
    print(f"Generated content: {content}")

if __name__ == "__main__":
    main()