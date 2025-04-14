from enum import Enum

class RewardType(Enum):
    VULNERABILITY = "vulns"
    CREDS = "creds"
    SHELL = "shell"
    METERPRETER = "meterpreter"
    FAIL_EXPLOIT = "fail_exploit"
    DISCOVERED_PORT = "discovered_port"

REWARDS = {
    RewardType.VULNERABILITY: 5,
    RewardType.CREDS: 20,
    RewardType.SHELL: 50,
    RewardType.METERPRETER: 80,
    RewardType.FAIL_EXPLOIT: -1,
    RewardType.DISCOVERED_PORT: 0.1,
}