#!/usr/bin/env python3
import logging
import sys
import ipaddress

# Import the pentest agent module
from agents import pentest_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PenTest-AI")


def get_user_input() -> dict:
    """Interactively get target information from the user."""
    print("\n==== AI-Driven Penetration Testing Tool ====\n")

    # Get target information
    while True:
        target = input("\nEnter target hostname, IP address, or network range (CIDR): ").strip()
        if not target:
            print("Target cannot be empty. Please try again.")
            continue

        # Basic validation
        try:
            # Check if it's an IP address or CIDR notation
            if '/' in target:
                ipaddress.ip_network(target)
            elif not any(c.isalpha() for c in target):  # Likely an IP address
                ipaddress.ip_address(target)
            # If it contains letters, assume it's a hostname (no validation here)
            break
        except ValueError:
            print("Invalid IP address or CIDR notation. Please try again.")

    # Confirm before proceeding
    print("\n=== Configuration Summary ===")
    print(f"Target: {target}")

    while True:
        confirm = input("\nStart penetration test with these settings? (y/n): ").strip().lower()
        if confirm == 'y':
            break
        elif confirm == 'n':
            print("Operation cancelled.")
            sys.exit(0)
        else:
            print("Please enter 'y' or 'n'.")

    return {
        "target": target,
    }


def main():
    """Main function to interactively get user input and start the penetration test."""
    try:
        # Get user input
        args = get_user_input()

        # Set logging level
        logging.getLogger().setLevel(getattr(logging, "INFO"))

        # Log the target information
        logger.info(f"Starting penetration test against {args['target']}")

        # Start the penetration testing agent with the provided arguments
        pentest_agent.start(**args)

    except KeyboardInterrupt:
        print("\nPenetration test interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()