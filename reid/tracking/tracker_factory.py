from typing import Any, Union

from ultralytics.trackers.byte_tracker import BYTETracker
from .enhanced_bytetrack import EnhancedByteTracker


class TrackerFactory:
    """Factory responsible for instantiating tracker implementations based on configuration settings."""

    @staticmethod
    def create(config: Any = None, args: Any = None) -> Union[BYTETracker, EnhancedByteTracker]:
        """Create a tracker implementation instance from the provided configuration.

        Supports positional 'config' or keyword 'args' arguments to match both manual creation
        and standard Ultralytics callbacks invocation.

        Args:
            config (Any, optional): Configuration object (dict, namespace, etc.) containing 'tracker_type'.
            args (Any, optional): Alternative keyword config object containing 'tracker_type'.

        Returns:
            Union[BYTETracker, EnhancedByteTracker]: Instantiated tracker backend object.

        Raises:
            ValueError: If configuration doesn't specify a valid 'tracker_type' or is unsupported.
        """
        cfg = config if config is not None else args

        if cfg is None:
            raise ValueError("Configuration settings must be provided to create a tracker.")

        if isinstance(cfg, dict):
            tracker_type = cfg.get("tracker_type")
        else:
            tracker_type = getattr(cfg, "tracker_type", None)

        if not tracker_type:
            raise ValueError("Configuration must contain a valid 'tracker_type' field.")

        tracker_type = tracker_type.lower()

        if tracker_type == "bytetrack":
            return BYTETracker(cfg)
        elif tracker_type in ("enhanced_bytetrack", "bytetrackx"):
            return EnhancedByteTracker(cfg)
        else:
            raise ValueError(f"Unsupported tracker type: '{tracker_type}'")
