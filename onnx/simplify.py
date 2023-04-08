import onnx
import onnx_graphsurgeon as gs
import numpy as np


@gs.Graph.register()
def replace_ConstantOfShape(self, node):
    assert node.op == "ConstantOfShape"
    in_tensor = node.inputs[0]
    in_tensor.outputs.clear()
    out_tensor = node.outputs[0]
    out_tensor.inputs.clear()

    zeros_like_in_tensor = self.layer(
        op="Mul", inputs=[in_tensor, np.zeros(1, np.int64)], outputs=["zeros_like"]
    )[0]
    ones_like_in_tensor = self.layer(
        op="Add",
        inputs=[zeros_like_in_tensor, np.ones(1, np.int64)],
        outputs=["ones_like"],
    )[0]
    value = self.layer(
        op="Reshape",
        inputs=[node.attrs["value"].values, ones_like_in_tensor],
        outputs=["value"],
    )[0]
    return self.layer(op="Tile", inputs=[value, in_tensor], outputs=[out_tensor])


graph = gs.import_onnx(onnx.load("preprocessor.onnx"))
for node in graph.nodes:
    if node.op == "ConstantOfShape":
        graph.replace_ConstantOfShape(node)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "preprocessor32.onnx")

graph = gs.import_onnx(onnx.load("encoder.onnx"))
for node in graph.nodes:
    if node.op == "ConstantOfShape":
        graph.replace_ConstantOfShape(node)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "encoder32.onnx")

graph = gs.import_onnx(onnx.load("decoder.onnx"))
for node in graph.nodes:
    if node.op == "ConstantOfShape":
        graph.replace_ConstantOfShape(node)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "decoder32.onnx")
