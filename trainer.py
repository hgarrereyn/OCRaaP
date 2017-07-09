import model

model = model.Model(train=True)

model.run_training(5000, checkpointFolder='checkpoints', ident=2)
