"""Print the installed Ray and RLlib versions."""

import importlib.metadata as md


def get_version(package: str) -> str:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "not installed"


def main() -> None:
    import ray

    print(f"ray (import):  {ray.__version__}")
    print(f"ray (dist):    {get_version('ray')}")

    # RLlib ships inside the ray distribution, so it shares ray's version.
    try:
        import ray.rllib as rllib

        print(f"rllib module:  {getattr(rllib, '__version__', ray.__version__)}")
    except ImportError:
        print("rllib module:  not installed")


if __name__ == "__main__":
    main()
