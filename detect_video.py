import cv2
from tool import darknet2pytorch
import torch
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect

model_pt = darknet2pytorch.Darknet('yolov4-obj.cfg', inference=True)
model_pt.load_state_dict(torch.load('yolov4-pytorch.pth'))

try:
    video = cv2.VideoCapture(input("Enter video path: "))
except:
    print("Error opening video")
    exit()
class_names = load_class_names('obj.names')

frame_width = int(video.get(3))
frame_height = int(video.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), int(video.get(cv2.CAP_PROP_FPS)), (frame_width,frame_height))

while True:
    return_value, frame = video.read()
    if return_value:
        original_image = frame
        frame = cv2.resize(frame, (416, 416))
        boxes = do_detect(model_pt, frame, 0.5, 0.4, use_cuda=False)
        print(f"how many boxes: {len(boxes[0])}")
        result = plot_boxes_cv2(original_image, boxes[0], class_names=["furry"])
        out.write(result)
        cv2.imshow('result', result)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            break
    else:
        break
out.release()