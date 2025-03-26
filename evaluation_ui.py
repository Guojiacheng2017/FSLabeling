import gradio as gr
from PIL import Image
from collections import defaultdict
import cv2, os, uuid
import json
import numpy as np

# LABEL = {'0': 'background','1': 'POS', '2': 'phone'}
LABEL = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush"
}

class IMGLABEL:
    def __init__(self, imgpath=[], labelpath=[]):
        self.IMG = [
            ".bmp", ".dib",       # Windows 位图
            ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",  # 便携式像素图 (PBM/PGM/PPM)
            ".sr", ".ras",        # Sun 光栅图像
            ".hdr", ".pic",        # Radiance HDR 格式
            ".jpeg", ".jpg", ".jpe",  # JPEG 格式 (注意: 不支持 12-bit/CMYK 等变体)
            ".jp2",                   # JPEG 2000
            ".png",                   # Portable Network Graphics
            ".webp",                  # WebP 图像 (需 OpenCV 3.0+)
            ".tiff", ".tif",          # TIFF 格式 (支持多数常见压缩类型)
            ".exr"
        ]

        if type(imgpath) != list or type(labelpath) != list:
            raise ValueError("imgpath / labelpath should be in list type")
        elif imgpath is None:
            raise ValueError("imgpath should not be empty")
        self.imglst, self.lbllst = imgpath, labelpath
        self.imgdict = self.__mutation__() # len(labelpath)
        
    def __basename__(self, path):
        return os.path.splitext(os.path.basename(path))[0]
    
    def __prefix__(self, path):
        if os.path.isdir(path):
            raise ValueError("Wrong Input type")
        return os.path.splitext(path)[-1].casefold()

    def __mutation__(self):
        imgdict = {}
        for img in self.imglst:
            if self.__prefix__(img) not in self.IMG:
                continue
            imgdict[self.__basename__(img)] = {'img': img,
                                               'lbl': False}
        self.imglst = [item['img'] for item in imgdict.values()]
        for lbl in self.lbllst:
            if self.__basename__(lbl) in imgdict:
                imgdict[self.__basename__(lbl)]['lbl'] = lbl
        print(imgdict)
        return imgdict
    
    def __len__(self):
        return len(self.imglst)

    def __renew__(self, labels=[]):
        if labels is None:
            raise ValueError("Empty label files")
        elif type(labels) != list:
            labels = [labels]
        self.lbllst = labels
        self.imgdict = self.__mutation__()


    def __getitem__(self, filename):
        basename = self.__basename__(filename)
        if basename in self.imgdict:
            rst = {
                'img': filename,
                'lbl': self.imgdict[basename]['lbl'] if self.imgdict[basename]['lbl'] else None
            }
            print(rst)
            return rst
        return f"No file named '{filename}'"
        


class EvaluationUI:
    def __init__(self):
        # self.image_path = ""
        self.annotation_path = ""
        self.image = None
        self.annotations = [] # annotation within one file
        self.file_index = 0
        self.result = {}  # Store evaluation results
        self.task = None
        # self.idx = 0
    
    def load_images(self, images_path):
        print(images_path)
        self.task = IMGLABEL(imgpath=images_path)
        self.single_idx = 0 # index within one annotation file
        return self.image_anno(self.file_index)

    def load_annotations(self, annotations_path):
        self.task.__renew__(labels=annotations_path)
        self.single_idx = 0
        img, patch, label, bbox, message = self.image_anno(self.file_index)
        return img, patch, label, bbox, message # self.labels, self.bboxes

    # ================================================================== #

    def image_anno(self, idx):
        # rst = self.task.__getitem__(self.task.imglst[idx]).items()
        # self.image_path, self.annotation_path = rst['img'], rst['lbl']
        self.image_path, self.annotation_path = self.task.__getitem__(self.task.imglst[idx]).values()
        try:
            self.image = cv2.imread(self.image_path)
            self.labels, self.bboxes = self.get_annotation(self.annotation_path)
            # print(self.image.shape)
            self.w, self.h, _ = self.image.shape
            self.labels = [int(i) + 1 for i in self.labels]
            if len(self.labels) > 0:
                # self.bboxes = [[int((bbox[0] - bbox[2]/2) * h),
                #                 int((bbox[1] - bbox[3]/2) * w),
                #                 int((bbox[0] + bbox[2]/2) * h),
                #                 int((bbox[1] + bbox[3]/2) * w)] for bbox in self.bboxes]
                image, patch, label, bbox = self.show_image(label=self.labels[self.single_idx], bbox=self.bboxes[self.single_idx])
                return image, patch, label, bbox, f"Loaded image: {self.image_path}"
            image, patch, label, bbox = self.show_image()
            return image, patch, None, None, f"Loaded image: {self.image_path}"
        except FileNotFoundError:
            return None, None, None, None, "Error: Image file not found."
        except Exception as e:
            return None, None, None, None, f"Error loading image: {e}"
    
    # ================================================================== #

    def show_image(self, label=None, bbox=None):
        im = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if label is None or bbox is None:
            return im, None, None, None
        crop_bbox = [int((bbox[0] - bbox[2]/2) * self.h),
                     int((bbox[1] - bbox[3]/2) * self.w),
                     int((bbox[0] + bbox[2]/2) * self.h),
                     int((bbox[1] + bbox[3]/2) * self.w)]
        return self.show_anno(im, label, crop_bbox), self.show_patch(self.image, label, crop_bbox), label, " ".join(map(str, bbox))
    
    
    def show_anno(self, im, label, bbox):
        # print(im.shape)
        # print(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        im = cv2.rectangle(im, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(255,0,0), thickness=2)
        org = (bbox[0] + 5, bbox[1] - 5)
        im = cv2.putText(im, f"{LABEL[str(label)]}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(255,0,0), thickness=2)
        # print(label, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        return im
    
    
    def show_patch(self, im, label, bbox):
        return im[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    
    # ================================================================== #

    def get_annotation(self, annopath):
        try:
            with open(annopath, 'r') as f:
                anno = f.readlines()
            anno = [an.split() for an in anno]
            anno = self.__2deval__(anno)
            # print(anno[:, 0], anno[:, 1:]) # .dtype(int)
            return anno[:, 0], anno[:, 1:]
        except:
            return np.array([]), np.array([])
        # labels & bboxes
    
    # ================================================================== #

    def __2deval__(self, array2d):
        for ridx in range(len(array2d)):
            for cidx in range(len(array2d[0])):
                array2d[ridx][cidx] = eval(array2d[ridx][cidx])
        return np.array(array2d)
    
    # ================================================================== #

    def prev_image(self):
        self.file_index -= 1
        if self.file_index < 0:
            result_str = f"Very first Image."
            # print(result_str)  # Optionally print to console
            self.file_index += 1
        self.single_idx = 0
        return self.image_anno(self.file_index)

    def next_image(self):
        self.file_index += 1
        if self.file_index >= self.task.__len__():
            result_str = f"Very last Image."
            # print(result_str)  # Optionally print to console
            self.file_index -= 1
        self.single_idx = 0
        return self.image_anno(self.file_index)

    def prev_annotation(self):
        self.single_idx -= 1
        if self.single_idx < 0:
            result_str = f"Very fisrt annotation ..."
            # print(result_str)  # Optionally print to console
            self.single_idx += 1
        try:
            return self.show_image(self.labels[self.single_idx], self.bboxes[self.single_idx])
        except:
            return self.show_image()

    def next_annotation(self):
        self.single_idx += 1
        result_str = "Next annotation ..."
        if self.single_idx >= len(self.labels):
            result_str = f"End of last annotation ..."
            # print(result_str)  # Optionally print to console
            self.single_idx -= 1
        # print(self.single_idx)
        try:
            image, patch, label, bbox = self.show_image(self.labels[self.single_idx], self.bboxes[self.single_idx])
            return image, patch, label, bbox, result_str
        except Exception as e:
            print(e)
            image, patch, label, bbox = self.show_image()
            return image, patch, label, bbox, result_str
    
    # def mark_match(self):
    #     image, patch, label, bbox = self.next_annotation()
    #     return image, patch, label, bbox, "no save annotation..."
        
    # def mark_mismatch(self):
    #     image, patch, label, bbox = self.next_annotation()
    #     return image, patch, label, bbox, "save annotation..."

    def discard(self):
        # image, patch, label, bbox = self.next_annotation()
        # return image, patch, label, bbox, "discard annotation..."
        return self.next_annotation()
    
    def save(self, patch, label, bbox, saved_path='./tmp_patch_label'):
        os.makedirs(os.path.join(saved_path, f'patches/{LABEL[label]}'), exist_ok=True)
        os.makedirs(os.path.join(saved_path, f'labels/{LABEL[label]}'), exist_ok=True)
        
        name = uuid.uuid4().hex[:8]
        cv2.imwrite(os.path.join(saved_path, f'patches/{LABEL[label]}/{name}.jpg'), patch)
        with open(os.path.join(saved_path, f'labels/{LABEL[label]}/{name}.txt'), 'w') as f:
            f.write(f"{label} {bbox}")


def button_interactive(patch):
    return gr.update(interactive=(patch is not None))

def colorfitting(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def radio_interactive(choice):
    for key, val in LABEL.items():
        if val == choice:
            return gr.Textbox(value=key, label="Label Index"), gr.Textbox(value=choice, label="Label", visible=True)
    return gr.Textbox(value="0", label="Label Index"), gr.Textbox(value=choice, label="Label", visible=True)

def update(labelint):
    # "0" if labelint is None else labelint
    try:
        return gr.Text(value=labelint, label="Label", visible=True), gr.Radio(label="Change Label", choices=list(LABEL.values()), value=LABEL[labelint], interactive=True)
    except:
        print(labelint)
        return gr.Text(value=labelint, label="Label", visible=True), gr.Radio(label="Change Label", choices=list(LABEL.values()), value=LABEL["0"], interactive=True)

ui = EvaluationUI()
# GRADIO_TEMP_DIR = "./gradio_uploads"
# os.environ["GRADIO_TEMP_DIR"] = "./gradio_uploads"
# os.makedirs(GRADIO_TEMP, exist_ok=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_show = gr.Image(type="pil", height=600, width=600)
            with gr.Row():
                # file_types=[".txt", ".jpg"]
                image_upload = gr.UploadButton(label="image upload", file_count="directory")#, file_types=['image']) #file_types=['image', 'text'])
                anno_upload = gr.UploadButton(label="Annotation File (YOLO)", file_count="directory")
            # upload_type = gr.Radio(choices=["Single", "List"], value="Single")
        with gr.Column():
            with gr.Blocks():
                patch_show = gr.Image(type="numpy", height=300, interactive=False)
                # gr.Image(type='numpy')
            with gr.Row():
                labelint = gr.Textbox(value="0", label="Label Index")
                labeltxt = gr.Textbox(value=LABEL[labelint.value], label="Label", visible=True)
            bboxtxt = gr.Textbox(label="Bounding Boxes", visible=False)
            status_display = gr.Textbox(label="Status")
            with gr.Row():
                modi_button = gr.Radio(label="Change Label", choices=list(LABEL.values()), value=LABEL[labelint.value], interactive=True)
            with gr.Row(variant='compact', equal_height=True):
                with gr.Column(min_width=150):
                    prev_img = gr.Button("Prev Img")
                    next_img = gr.Button("Next Img")
                # prev_bbox = gr.Button("Prev Label")
                with gr.Column(min_width=150):
                    discard_button = gr.Button("Discard")
                    save_button = gr.Button("Save Patch", interactive=(patch_show == None))
                # with gr.Column(min_width=150):
                #     true_button = gr.Button("True")
                #     false_button = gr.Button("False")
                # next_bbox = gr.Button("Next Label")
                

    image_upload.upload(ui.load_images, image_upload, [image_show, patch_show, labelint, bboxtxt, status_display])
    anno_upload.upload(ui.load_annotations, anno_upload, [image_show, patch_show, labelint, bboxtxt, status_display])
    prev_img.click(ui.prev_image, None, [image_show, patch_show, labelint, bboxtxt, status_display])
    next_img.click(ui.next_image, None, [image_show, patch_show, labelint, bboxtxt, status_display])
    # prev_bbox.click(ui.prev_annotation, None, status_display)
    # next_bbox.click(ui.next_annotation, None, status_display)

    # modi_button.click(lambda x: x, None, None)
    modi_button.change(
        fn=radio_interactive,
        inputs=modi_button,
        outputs=[labelint, labeltxt],
    )
    labelint.change(
        fn=update,
        inputs=labelint,
        outputs=[labeltxt, modi_button]
    )
    

    save_button.click(ui.save, [patch_show, labelint, bboxtxt], None)
    patch_show.change(
        fn=button_interactive,
        inputs=patch_show,
        outputs=save_button,
    )
    

    discard_button.click(ui.discard, None, [image_show, patch_show, labelint, bboxtxt, status_display])
    
    # true_button.click(ui.mark_match, None, [image_show, patch_show, labelint, bboxtxt])
    # false_button.click(ui.mark_mismatch, None, [image_show, patch_show, labelint, bboxtxt])
    
demo.launch()