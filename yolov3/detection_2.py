
from .models import *
from .utils.utils import *
from .utils.parse_config import parse_data_cfg


class Yolo():
    def __init__(self, weights='yolov3/weights/yolov3.pt',
                 #data_cfg='yolov3/data/camera.data',
                 cfg='yolov3/cfg/yolov3.cfg', img_size=416, conf_thres=0.5,
                 nms_thres=0.4):
        self.device = torch_utils.select_device()
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.model = Darknet(cfg, img_size)
        self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        self.model.fuse()
        self.model.to(self.device).eval()

    def letterbox(self, img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
        # Resize a rectangular image to a 32 pixel multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            ratio = float(new_shape) / max(shape)
        else:
            ratio = max(new_shape) / max(shape)  # ratio  = new / old
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if mode is 'auto':  # minimum rectangle
            dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
            dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
        elif mode is 'square':  # square
            dw = (new_shape - new_unpad[0]) / 2  # width padding
            dh = (new_shape - new_unpad[1]) / 2  # height padding
        elif mode is 'rect':  # square
            dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
            dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
        return img, ratio, dw, dh

    def transform(self, img, img_size):
        # Padded resize
        trs, _, _, _ = self.letterbox(img, new_shape=img_size)

        # Normalize RGB
        trs = trs[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        trs = np.ascontiguousarray(trs, dtype=np.float32)  # uint8 to float32
        trs /= 255.0  # 0 - 255 to 0.0 - 1.0

        return trs

    def detect_image(self, img):

        # classes = load_classes(parse_data_cfg(data_cfg)['names'])
        classes = 'person'

        img_trs = self.transform(img, self.img_size)

        img_trs = torch.from_numpy(img_trs).unsqueeze(0).to(self.device)
        pred, _ = self.model(img_trs)
        det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]

        crds = []

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img_trs.shape[2:], det[:, :4], img.shape).round()

            det = det.detach().numpy()



            for bbox in det:
                if bbox[6] == 0 and bbox[4] > 0.80:
                    tl = bbox[:2]
                    br = bbox[2:4]
                    # w = int(bbox[2] - bbox[0])
                    # h = int(bbox[3] - bbox[1])
                    # x = int(w//2 + bbox[0])
                    # y = int(h//2 + bbox[1])

                    # crds.append([tl[0], tl[1], br[0], br[1], det[0][4]])
                    crds.append([tl[0], tl[1], br[0], br[1]])

            crds = np.array(crds)

        return crds