def get_simulator(simulator_name: str, **kwargs):
    if simulator_name == "expert_user":
        from .expert_user import ExpertUser
        return ExpertUser(**kwargs)
    elif simulator_name == "copref_user":
        from .copref_user import CoPrefUser
        return CoPrefUser(**kwargs)
    elif simulator_name == "full_spec_user":
        from .full_spec_user import FullSpecificationUser
        return FullSpecificationUser(**kwargs)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")
