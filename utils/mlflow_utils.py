"""MLflow helpers — call patch_artifact_logging() once at entry-point startup."""

import logging

log = logging.getLogger(__name__)


def patch_artifact_logging() -> None:
    """Wrap mlflow.log_artifact / log_artifacts to warn-and-continue on server errors.

    The remote artifact backend occasionally returns HTTP 500 (storage misconfigured,
    disk full, etc.). Without this patch a single failed upload aborts the entire
    training run. Params and metrics are unaffected — only file uploads are guarded.
    """
    try:
        import mlflow
    except ImportError:
        return

    if getattr(mlflow, "_artifact_logging_patched", False):
        return

    _orig_log_artifact  = mlflow.log_artifact
    _orig_log_artifacts = mlflow.log_artifacts

    def _safe_log_artifact(local_path, artifact_path=None):
        try:
            _orig_log_artifact(local_path, artifact_path)
        except Exception as exc:
            log.warning("mlflow.log_artifact skipped (%s): %s", local_path, exc)

    def _safe_log_artifacts(local_dir, artifact_path=None):
        try:
            _orig_log_artifacts(local_dir, artifact_path)
        except Exception as exc:
            log.warning("mlflow.log_artifacts skipped (%s): %s", local_dir, exc)

    mlflow.log_artifact  = _safe_log_artifact
    mlflow.log_artifacts = _safe_log_artifacts
    mlflow._artifact_logging_patched = True
    log.info("mlflow artifact logging patched (errors → warnings)")
