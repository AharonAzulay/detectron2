import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import copy
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
# torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
# net = torch.hub.load('zhanghang1989/ResNeSt', 'resnest269', pretrained=True)

OUTPUTS_DIR = './translations/'
MODEL_DIR = './models/'
def addMask(X,M):
    XwMasks = copy.deepcopy(X).astype('float64')
    M = np.mean(M.numpy(),0)
    M[M>0] = 255
    XwMasks[:,:,2] += M.astype('float64')
    XwMasks[XwMasks>255] = 255
    X = (0.6 * X + 0.4 * XwMasks).astype('uint8')
    return X

def visualize_translated(images,masks):
    # initialize water image
    height = images[0].shape[0]
    width = images[0].shape[1]
    water_depth = np.zeros((height, width,3), dtype=float)

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = 5
    video_filename = 'output.avi'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width-50, height-50),True)

    for i, (image) in enumerate(images):
        mask = masks[i]
        image = addMask(image,mask)

        plt.close('all')
        plt.imshow(image)
        plt.tight_layout()
        plt.axis('off')
        out.write(image[25:-25,25:-25,:])
        plt.savefig(OUTPUTS_DIR + str(i+100) + '.png')

    out.release()


im = cv2.imread("./car1.png")


cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file((MODEL_DIR + "mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x.yml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_DIR + "mask_cascade_rcnn_ResNeSt_200_FPN_dcn_syncBN_all_tricks_3x.yml")
predictor = DefaultPredictor(cfg)


Masks = []
Ims = []
for i in range(32):
    translatedIm = im[i:-32+i,i:-32+i,:]
    outputs = predictor(translatedIm)
    pr_mask = outputs["instances"].pred_masks
    Masks.append(pr_mask)
    Ims.append(translatedIm)


visualize_translated(Ims,Masks)

outputs["instances"].pred_masks
# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)


# We can use `Visualizer` to draw the predictions on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite('test2.png',out.get_image()[:, :, ::-1])

