class HunchError(Exception):
    """User-facing library error."""


class NoSessionError(HunchError):
    def __init__(self) -> None:
        super().__init__(
            "ask(), draft(), and feels() need an active hunch session. Call classify.run(jev)."
        )
