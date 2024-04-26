from typing import Tuple, List
import argparse
import dataclasses
import numpy as np

import cv2
import openvino as ov

from utils import load_label_dict


@dataclasses.dataclass
class DetectionResult:
    id: int
    label: str
    score: float
    bbox: Tuple[int, int, int, int]


class OvDetrResnet50:
    def __init__(
        self,
        arch: str,
        wght: str,
        device: str,
        in_height: str,
        in_width: str,
        threshold: float,
    ) -> None:

        self.arch = arch
        self.wght = wght
        self.device = device
        self.in_width = in_width
        self.in_height = in_height
        self.threshold = threshold

    def model_init(self) -> None:
        """
        Read the network and weights from file, load the model on device and
        get the input and output of the nodes.

        Returns:
            input_key: Input node network
            output_keys: Output nodes network
            exec_net: Encoder model network
            net: Model network
        """
        model_path = self.arch
        device = self.device

        core = ov.Core()

        model = core.read_model(model=model_path)
        compiled_model = core.compile_model(model=model, device_name=device)

        input_key = compiled_model.input(0)
        output_keys = compiled_model.output(0)
        return input_key, output_keys, compiled_model

    def predict(self, model_init_object, raw_image_bytes: bytes, label_dict: dict):
        """
        Run the inference on the input image.

        Args:
            model_init_object: Model object
            raw_image_bytes: Image in bytes
        Returns:
            result: Inference result
        """
        results = []
        nparr = np.frombuffer(raw_image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        input_size = (self.in_width, self.in_height)

        resized_image_re = cv2.resize(frame, input_size)
        input_image_re = np.expand_dims(resized_image_re.transpose(2, 0, 1), 0)

        _, _, compiled_model_re = model_init_object

        prediction_result = compiled_model_re([input_image_re])[
            compiled_model_re.output(1)
        ]
        scores_result = compiled_model_re([input_image_re])[compiled_model_re.output(0)]
        height, width = frame.shape[:2]

        detection_id = 0
        for i in range(prediction_result.shape[1]):

            box = prediction_result[0, i, :]
            scores = scores_result[0, i, :91]

            max_score = -1
            max_class_id = -1

            for class_id in range(scores.shape[0]):
                if scores[class_id] > max_score:
                    max_score = scores[class_id]
                    max_class_id = class_id

            if max_class_id != -1 and max_score > self.threshold:
                detection_id += 1
                label = label_dict[max_class_id - 1]

                x_center = int(box[0] * width)
                y_center = int(box[1] * height)
                w = int(box[2] * width)
                h = int(box[3] * height)

                x_min = x_center - w // 2
                y_min = y_center - h // 2
                x_max = x_min + w
                y_max = y_min + h

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

                results.append(
                    DetectionResult(
                        id=detection_id,
                        label=label,
                        score=max_score,
                        bbox=(x_min, y_min, x_max, y_max),
                    )
                )
            else:
                continue

        cv2.imwrite("output_image.jpg", frame)
        return results


def main():

    parser = argparse.ArgumentParser(
        description="Simple OpenVINO DETR Resnet50 example"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="demo/sample.png",
        help="Path to the image file",
    )
    parser.add_argument(
        "--threshold", type=int, default=10, help="Confidence score threhold"
    )

    arg = parser.parse_args()

    model_instance = OvDetrResnet50(
        arch="model/public/detr-resnet50/FP32/detr-resnet50.xml",
        wght="model/public/detr-resnet50/FP32/detr-resnet50.bin",
        device="CPU",
        in_height=800,
        in_width=1137,
        threshold=arg.threshold,
    )
    model_init_object = model_instance.model_init()

    with open(arg.image_path, mode="rb") as img_file:
        image_binary = img_file.read()

    # coco label obtained from ref: https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt
    label_dict = load_label_dict()

    results = model_instance.predict(model_init_object, image_binary, label_dict)
    print(f"{results=}")


if __name__ == "__main__":
    main()
