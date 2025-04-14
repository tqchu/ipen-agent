# Initialize state with maximum 3 hosts and 4 ports per host
from env.v1.action import Result
from env.v1.state import PenTestState, ShellType
from model.enum import Privilege

MH, MP = 10, 10
state = PenTestState(MH, MP)

# Populate Host 0
state.add_host("192.168.100.168", privilege_level=0)
state.add_open_port("192.168.100.168", 22)
state.add_vulnerability("192.168.100.168", 22, "CVE-2021-1111")
state.add_open_port("192.168.100.168", 80)
state.add_vulnerability("192.168.100.168", 80, "CVE-2021-2222")
state.add_username("192.168.100.168", "admin")
state.add_credential("192.168.100.168", "admin", "password123")

print("Is done", state.is_done())

# Populate Host 1
state.add_host("192.168.100.169", privilege_level=1)
state.add_open_port("192.168.100.169", 443)
state.mark_exploited("192.168.100.169", ShellType.METERPRETER)
state.add_vulnerability("192.168.100.169", 443, "CVE-2022-3333")
state.add_vulnerability("192.168.100.169", 443, "CVE-2023-4444")
state.add_token("192.168.100.169", "session-token-abc")

print("Is done", state.is_done())
# Host 2 remains with no data (undiscovered)

# Encode the state to a vector
vec = state.to_vector()
print("Encoded vector:", vec.tolist())

# Decode the vector back to a state object
decoded_state = PenTestState.from_vector(vec, MH, MP)
print("Reconstructed state from vector:", decoded_state.get_readable_state())

# Verify that encoding the reconstructed state yields the same vector
re_vec = decoded_state.to_vector()
print("Re-encoded vector matches original:", (re_vec == vec).all())

result = Result(
    reward=0,
    discovered_hosts=["192.168.100.200", "192.168.100.210"],
    open_ports={
        "192.168.100.200": [(22, "ssh"), (80, "http")],
    },
    vulners={
        "192.168.100.200": [(22, ["CVE-2023-5555", "CVE-2023-6666"])],
    },
    usernames={
        "192.168.100.200": ["admin", "user"],
    },
    privileges={
        "192.168.100.200": Privilege.USER,
    },
    credentials={
        "192.168.100.200": [("msf", "msf")],
    },
)

state.update(result)
print("State after update:", state.get_readable_state())

result = Result(
    reward=0,
    open_ports={
        "192.168.100.210": [(80, "http")],
    },
    hosts_shell={
        "192.168.100.200": ShellType.SHELL,
    },
    hosts_meterpreter={
        "192.168.100.200": ShellType.METERPRETER,
    },
    tokens={
        "192.168.100.200": ["token1", "token2"],
    },
)
state.update(result)

print("State after update:", state.get_readable_state())

vec = state.to_vector()
print("Encoded vector:", vec.tolist())

# Decode the vector back to a state object
decoded_state = PenTestState.from_vector(vec, MH, MP)
print("Reconstructed state from vector:", decoded_state.get_readable_state())

# Verify that encoding the reconstructed state yields the same vector
re_vec = decoded_state.to_vector()
print("Re-encoded vector matches original:", (re_vec == vec).all())
