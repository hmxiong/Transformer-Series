from .detr import build
from .deformable_detr import build_deformable_detr

def build_model(args):
    if args.model_type == 'deformable':
        return build_deformable_detr(args)
    else:
        return build(args)
