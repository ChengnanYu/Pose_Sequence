
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'posenet':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    elif opt.model == 'poselstm':
        from .poselstm_model import PoseLSTMModel
        model = PoseLSTMModel()
    elif opt.model == 'posenetnobeta':
        from .posenet_nobeta_model import PoseNetNoBetaModel
        model = PoseNetNoBetaModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
