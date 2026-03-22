import torch
import numpy as np
import cv2
import base64

def generate_gradcam(model, input_tensor, device):
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    input_tensor = input_tensor.to(device)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = np.maximum(cam.cpu().numpy(), 0)
    cam = cv2.resize(cam, (128, 128))

    if cam.max() != 0:
        cam = cam / cam.max()

    # Convert to heatmap image
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    _, buffer = cv2.imencode('.png', heatmap)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    handle_f.remove()
    handle_b.remove()

    return {
        "heatmap": heatmap_base64,
        "activations": acts
    }