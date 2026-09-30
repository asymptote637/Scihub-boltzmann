"""Select the actual numerical engine; retain the archived cavity path."""
from research_options import generic_engine, is_extended


def build_config(params):
    if is_extended(params):
        from research_options import make_config
        return make_config(params)
    from reference import make_config
    return make_config(**{k:v for k,v in params.items() if k!="backend"})


def solver_class(params):
    if generic_engine(params):
        if params["backend"]=="cpu":
            from research_solver import ResearchSolver
            return ResearchSolver
        if params["backend"]=="array":
            from research_solver import ResearchArraySolver
            return ResearchArraySolver
        from research_gpu import ResearchFusedSolver
        return ResearchFusedSolver
    if params["backend"]=="cpu":
        from cavity_models import CavityCPUSolver
        return CavityCPUSolver
    if params["backend"]=="array":
        from gpu_solver import ArrayGPUSolver
        return ArrayGPUSolver
    from gpu_fused import FusedGPUSolver
    return FusedGPUSolver
