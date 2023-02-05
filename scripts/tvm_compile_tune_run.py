from tvm.driver import tvmc


model = tvmc.load("../assets/resnet50-v2-7.onnx")
model.summary()
