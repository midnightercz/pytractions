def _init_traction(traction_cls, inputs, resources, args, index, uid="0", observer=None):
    """After tractor is created, all tractions needs to be copied.

    This way it's possible to use same Tractor class multiple times
    """
    init_fields = {}
    for ft, field in traction_cls.__dataclass_fields__.items():
        # set all inputs for the traction to outputs of traction copy
        # created bellow

        if ft.startswith("r_"):
            init_fields[ft] = resources[ft]

        elif ft.startswith("a_"):
            init_fields[ft] = args[ft]

        elif ft.startswith("i_") and ft in inputs:
            init_fields[ft] = inputs[ft]

    init_fields["uid"] = uid
    # create copy of existing traction
    ret = traction_cls(**init_fields)
    ret._observer._observers[id(observer)] = (observer, str(f"tractions.[{index}]"))
    return ret
