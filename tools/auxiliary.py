import logging

from env.action import PentestAction, Result
from metasploit.msf_rpc import MsfRpcController
from tools.base_tool import BaseTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class AuxiliaryRunner(BaseTool):
    def __init__(self):
        super().__init__(name="AuxiliaryRunner")
        self.msf = MsfRpcController.get_instance("truongquangchu", "localhost", 55553, True)
        # Optionally, list available auxiliary modules
        self.aux_modules = self.msf.list_auxiliary_modules()

    def run(self, action: PentestAction, actions: list[PentestAction], state):
        self.logger.info(f"Running auxiliary module for action: {action.description}")
        host = action.target
        # Determine which auxiliary module to run based on the target or discovered service.
        module_name = action.module
        port = action.port

        # Run the auxiliary module through your RPC controller (you might need to implement run_auxiliary)
        result = self.msf.run_auxiliary(module_name, host, port)
        self.logger.debug(f"Auxiliary module result: {result}")

        # {'error': 'Missing required options: rhost', 'output': '', 'results': None, 'success': False}
        # 'dos/http/slowloris'
        # Process the results as needed and generate new actions or rewards
        reward = 0.5  # Example reward value based on output
        future_actions = []  # You can chain more actions if necessary

        return Result(
            reward=reward,
            status="success",
            details={"module": module_name, "target": host},
            future_actions=future_actions,
        )
