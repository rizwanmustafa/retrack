from .interface import AIModel, AnalysisResult, BackendUnavailable, CheckpointInfo, ModelError, Observation


def create_ai_model(config):
    """The configured AIModel backend. Tests replace app.extensions['retrack_ai'] with a fake."""
    backend = config['AI_BACKEND']
    if backend == 'tracktacular':
        from .tracktacular import TrackTacularModel
        return TrackTacularModel(config['WORLDTRACK_DIR'], config['INFERENCE_DEVICE'])
    raise ValueError(f'Unknown AI backend {backend!r}')
