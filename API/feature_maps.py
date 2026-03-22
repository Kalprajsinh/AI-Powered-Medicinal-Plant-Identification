import cv2
import numpy as np
import base64

def feature_maps_to_images(acts, max_maps=6):
    feature_images = []

    acts = acts.squeeze(0)  

    for i in range(min(max_maps, acts.shape[0])):
        fmap = acts[i].detach().cpu().numpy()

        fmap -= fmap.min()
        if fmap.max() != 0:
            fmap /= fmap.max()

        fmap = np.uint8(255 * fmap)

        fmap = cv2.resize(fmap, (128, 128))

        fmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)

        _, buffer = cv2.imencode('.png', fmap)
        fmap_base64 = base64.b64encode(buffer).decode("utf-8")

        feature_images.append(fmap_base64)

    return feature_images