import torch, logging
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline

# Spezifische Warnung unterdrücken: "You have disabled the safety checker ..."
_diffusers_logger = logging.getLogger('diffusers')

class _SuppressSafetyWarning(logging.Filter):
    _needle = 'You have disabled the safety checker'
    def filter(self, record: logging.LogRecord) -> bool:  # True => weiterreichen
        msg = record.getMessage()
        if self._needle in msg:
            return False
        return True

if not any(isinstance(f, _SuppressSafetyWarning) for f in _diffusers_logger.filters):  # idempotent
    _diffusers_logger.addFilter(_SuppressSafetyWarning())

def infer_defaults(model_id: str, steps, guidance):
    if steps is None:
        steps = 2 if 'turbo' in model_id.lower() else 25
    if guidance is None:
        guidance = 0.0 if 'turbo' in model_id.lower() else 7.5
    return steps, guidance

def build_pipeline(model_id: str, half: bool):
    torch_dtype = torch.float16 if half and torch.cuda.is_available() else None
    if 'turbo' in model_id.lower():
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant='fp16' if torch_dtype else None
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None
        )
    if torch.cuda.is_available():
        pipe = pipe.to('cuda')
    return pipe
