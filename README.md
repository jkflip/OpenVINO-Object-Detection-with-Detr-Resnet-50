# OpenVINO Detection with Detr-ResNet50

This is a simple example of utilizing OpenVINO to perform object detection onan image using the detr-resnet50 model.


1.  Setup environment
    ```bash
    .script/local-setup-venv.sh
    ```

2.  Download the model from OpenVINO's model zoo
    ```bash
    .script/download-model.sh
    ```

3.  Convert model to OpenVINO compatible format
    ```bash
    .script/convert-pth-model.sh
    ```

4.  Run the inference
    ```bash
    source venv/bin/activate
    python3 openvino_detr_resnet50.py
    ```
    The inference results will be shown in the terminal and the output image will be saved as `output_image.jpg`.

    By default, the script is set to inference a sample image from `demo` folder.

    To see options, enter:
    ```bash
    python3 openvino_detr_resnet50.py --help
    ```


## Reference

1.  [OpenVINO detr-resnet50](https://docs.openvino.ai/2024/omz_models_model_detr_resnet50.html)
2.  [OpenVINO convert model](https://docs.openvino.ai/2024/notebooks/121-convert-to-openvino-with-output.html)
3.  [OpenVINO Object Detection Demo](https://docs.openvino.ai/2022.3/omz_demos_object_detection_demo_python.html#doxid-omz-demos-object-detection-demo-python)
4.  [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
5.  [Coco dataset label](https://github.com/amikelive/coco-labels/blob/master/coco-labels-paper.txt)
