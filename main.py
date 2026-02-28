import asyncio
import logging
import sys

from agent import run_agent


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).error("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
