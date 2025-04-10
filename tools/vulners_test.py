import logging

from env.action import PentestAction
from env.state import State
from tools.exploit import Exploiter
from tools.exploit_db import ExploitDbRunner
from tools.scanners import Scanner
from tools.vulners import VulnerabilityScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# targets = ["sv.dut.udn.vn", "thpt-hungvuong.edu.vn", "thptnguyenthaibinh.edu.vn"]
targets = ["14.236.17.231","27.71.235.203", "222.255.167.54"]

for target in targets:

    state = State()

    # Run port scanner
    portScanner = Scanner()
    portScanAction = PentestAction(action_type="scan", target=target)
    result = portScanner.run(portScanAction,
                             [portScanAction], state)

    state.update(result)
    # Run the vulnerability scanner

    vulScanner = VulnerabilityScanner()
    # vulScanner = VulnerabilityScanner()
    vulScanAction = PentestAction(action_type="vuln_scan", target=target)
    result = vulScanner.run(vulScanAction, [vulScanAction], state)
    state.update(result)

    for action in result.future_actions:
        if isinstance(action.tool, Exploiter):
            result = action.tool.run(action, result.future_actions, state)
            state.update(result)

            logging.info("state after exploit %s", state)
        elif isinstance(action.tool, ExploitDbRunner):
            # action.tool.run(action, result.future_actions, state)
            logging.info("ignore")