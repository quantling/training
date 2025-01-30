class NotEnoughDiskSpaceError(Exception):
    """Custom exception for insufficient disk space."""
    def __init__(self, message="There is likely not enough disk space left. Dumped the current state of the counters."):
        self.message = message
        super().__init__(self.message)