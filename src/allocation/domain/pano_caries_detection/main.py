import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.utils.benchmark as benchmark

from src.allocation.domain.pano_caries_detection.utils import draw_objs
from src.allocation.domain.pano_caries_detection.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from src.allocation.domain.pano_caries_detection.network.faster_rcnn_framework import FasterRCNN

def create_model(num_classes):

    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, trainable_layers=3)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes+1, rpn_score_thresh=0.5)

    return model

def pano_caries_detecion(model, weights_path, pil_img, return_type='image_array'):

    # read class_indict
    class_dict={'Decay': 1}
    category_index = {str(v): str(k) for k, v in class_dict.items()}
    
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("using {} device.".format(device))

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(pil_img)

    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    torch.backends.cudnn.benchmark = True  
    model.eval()
    error_messages = ""  
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        predictions = model(img.to(device))[0]

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        results_dict={
            'boxes': predict_boxes,
            'labels': predict_classes,
            'scores': predict_scores,
        }


        # 保存预测的图片结果
        #plot_img.save("test_result.jpg")

    if return_type == 'image_array':
        if len(predict_boxes) == 0:
            return None, "No caries found"
        plot_img = draw_objs(pil_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='./conf/arial.ttf',
                             font_size=20)        
        return plot_img, error_messages
    else:
        return results_dict


if __name__ == '__main__':
    # # create model
    model = create_model(num_classes=1)
    weights_path = "./models/dentistry_pano-caries-detection-resNetFpn_5.12.pth"
    pil_img = Image.open("./tests/files/027107.jpg")
    plot_img = pano_caries_detecion(model, weights_path, pil_img, return_type='image')
    plt.imshow(plot_img)
    plt.show()