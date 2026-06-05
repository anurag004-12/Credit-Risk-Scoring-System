class PipelineError(Exception):
    """Generic pipeline error."""
    pass


class DataValidationError(PipelineError):
    """Raised when input data is invalid for the pipeline."""
    pass
