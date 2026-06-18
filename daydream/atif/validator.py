"""Validator for Agent Trajectory Interchange Format (ATIF) trajectories.

This module provides validation functionality for trajectory files following
the ATIF specification (RFC 0001).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import ValidationError

from daydream.atif.models import Trajectory


class TrajectoryValidator:
    """Validator for ATIF trajectory format.

    Validates that trajectory JSON follows the schema defined in RFC 0001,
    using Pydantic models for validation.

    Always collects all validation errors before returning.
    """

    def __init__(self):
        """Initialize the validator."""
        self.errors: List[str] = []
        self._trajectory_dir: Path | None = None

    def _add_error(self, error: str) -> None:
        """Add an error to the error list.

        Args:
            error: Error message to add.
        """
        self.errors.append(error)

    def _is_url(self, path: str) -> bool:
        """Check if a path is a URL rather than a local file path.

        Args:
            path: The path to check.

        Returns:
            True if the path appears to be a URL, False otherwise.
        """
        return "://" in path  # scheme:// pattern (https://, s3://, gs://)

    def _validate_image_paths(self, trajectory_data: dict) -> None:
        """Validate that all referenced local image paths exist.

        URLs are skipped since they cannot be validated locally.

        Args:
            trajectory_data: The parsed trajectory dictionary.
        """
        if self._trajectory_dir is None:
            return

        def check_content_for_images(content: Any, location: str) -> None:
            """Check content field for image references."""
            if not isinstance(content, list):
                return
            for idx, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image":
                    source = part.get("source", {})
                    if isinstance(source, dict):
                        image_path = source.get("path")
                        if image_path:
                            if self._is_url(image_path):  # URLs can't be validated locally
                                continue
                            path_obj = Path(image_path)
                            if path_obj.is_absolute():
                                full_path = path_obj
                            else:
                                full_path = self._trajectory_dir / image_path
                            if not full_path.exists():
                                self._add_error(
                                    f"{location}[{idx}].source.path: "
                                    f"referenced image file does not exist: {image_path}"
                                )

        for step_idx, step in enumerate(trajectory_data.get("steps", [])):
            step_loc = f"trajectory.steps[{step_idx}]"

            message = step.get("message")
            if isinstance(message, list):
                check_content_for_images(message, f"{step_loc}.message")

            observation = step.get("observation")
            if observation:
                for res_idx, result in enumerate(observation.get("results", [])):
                    content = result.get("content")
                    if isinstance(content, list):
                        check_content_for_images(
                            content,
                            f"{step_loc}.observation.results[{res_idx}].content",
                        )

    def validate(
        self, trajectory: Union[Dict[str, Any], str, Path], validate_images: bool = True
    ) -> bool:
        """Validate a complete trajectory.

        Args:
            trajectory: Trajectory to validate. Can be a dict, JSON string,
                       or path to a JSON file.
            validate_images: Whether to validate that referenced image paths exist.
                           Only applicable when trajectory is a file path.

        Returns:
            True if valid, False otherwise. All errors are collected in self.errors.
        """
        self.errors = []
        self._trajectory_dir = None

        if isinstance(trajectory, (str, Path)):
            path = Path(trajectory)
            if path.exists():
                self._trajectory_dir = path.parent
                with open(path, "r") as f:
                    try:
                        trajectory = json.load(f)
                    except json.JSONDecodeError as e:
                        self._add_error(f"Invalid JSON: {e}")
                        return False
            else:
                try:
                    trajectory = json.loads(str(trajectory))
                except json.JSONDecodeError as e:
                    if isinstance(trajectory, Path):
                        self._add_error(f"File not found: {trajectory}")
                    else:
                        self._add_error(
                            f"Input string is not a valid file path and not valid JSON: {e}"
                        )
                    return False

        if not isinstance(trajectory, dict):
            self._add_error("Trajectory must be a JSON object/dict")
            return False

        try:
            Trajectory(**trajectory)
        except ValidationError as e:
            for error in e.errors():
                loc_str = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                error_type = error["type"]
                error_input = error.get("input")

                if error_type == "missing":
                    self._add_error(f"trajectory.{loc_str}: required field is missing")
                elif error_type == "extra_forbidden":
                    self._add_error(
                        f"trajectory.{loc_str}: unexpected field (not part of ATIF schema)"
                    )
                elif error_type.startswith("value_error"):
                    self._add_error(f"trajectory.{loc_str}: {msg}")
                elif error_type.startswith("type_error") or error_type in [
                    "string_type",
                    "int_type",
                    "float_type",
                    "dict_type",
                    "list_type",
                ]:
                    # Include the actual value for better debugging
                    if error_input is not None:
                        self._add_error(
                            f"trajectory.{loc_str}: expected {error_type.replace('_', ' ')}, got {type(error_input).__name__}"
                        )
                    else:
                        self._add_error(f"trajectory.{loc_str}: {msg}")
                elif error_type == "literal_error":
                    # include the actual invalid value
                    if error_input is not None:
                        self._add_error(
                            f"trajectory.{loc_str}: {msg}, got '{error_input}'"
                        )
                    else:
                        self._add_error(f"trajectory.{loc_str}: {msg}")
                else:
                    self._add_error(f"trajectory.{loc_str}: {msg}")

        if validate_images and self._trajectory_dir is not None:
            self._validate_image_paths(trajectory)

        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        """Get all validation errors.

        Returns:
            List of error messages.
        """
        return self.errors


def validate_trajectory(trajectory: Union[Dict[str, Any], str, Path]) -> bool:
    """Validate a trajectory against the ATIF schema.

    Args:
        trajectory: Trajectory to validate (dict, JSON string, or file path).

    Returns:
        True if valid, False otherwise.
    """
    validator = TrajectoryValidator()
    return validator.validate(trajectory)

