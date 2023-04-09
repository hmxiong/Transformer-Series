from .detr import build
from .deformable_detr import build_deformable_detr
from .dino import build_dino

def build_model(args):
    if args.model_type in ['deformable']:
        return build_deformable_detr(args)
    elif args.model_type in ['dino']:
        return build_dino(args)
    else:
        return build(args)
