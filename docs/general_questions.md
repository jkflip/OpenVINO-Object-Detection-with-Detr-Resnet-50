# OpenVINO questions

1.  What is OpenVINO?
2.  Explain the proces of training AI model from data collection to having a working model.
3.  Given a model `detr-resnet-50`, create an implementation of running this model with OpenVINO.


## 1. What is OpenVINO?

OpenVINO (shorthand for `Open Visual Inference and Neural Network Optimization`) is cross-platform software that enables the development of computer vision and deep learning applications.

It mainly focuses on optimizing deep learning models for inference on Intel hardware platforms, including CPUs, GPUs, FPGAs, and VPUs. The toolkit provides a comprehensive set of tools, libraries, and pre-trained models to help developers build and deploy computer vision applications.

It also includes a high-level C++ or Python inference engine API that simplifies the integration of deep learning models with application logic. OpenVINO supports popular deep learning frameworks such as TensorFlow, Caffe, and ONNX, which allows developers to leverage existing models and frameworks in their applications. Overall, OpenVINO aims to accelerate the development and deployment of computer vision applications by providing optimized tools and libraries for deep learning inference on Intel hardware.

In the context of my work, I mainly utilized OpenVINO models to perform real-time object detection and classification tasks on video  or images from surveillance cameras.

## 2. Explain the process of training an AI model from data collection to having a working model.

The process of training an AI model can be summarized into the following:

1. Data collection or preparation: The first step in training an AI model is to collect a large and diverse dataset that represents the problem domain. The dataset should include labeled examples that the model will learn from. For example, in an image classification task, the dataset would consist of images with corresponding class labels (i.e. if task is vehicle classification, take images of different vehicles and label them accordingly).

2. Data preprocessing: Once the dataset is collected, it needs to be preprocessed to prepare it for training. This may involve tasks such as resizing images, normalizing pixel values, and augmenting the data to increase its diversity.

3. Model selection: Next, you'll need to choose an appropriate model architecture that is well-suited depending on the task required to be solved. This may involve selecting a pre-trained model or designing a custom architecture.

4. Model training: With the dataset and model architecture in place, model training can begin feeding it with collected data. During training, the model learns to map input data to output labels by adjusting its internal parameters.

5. Model evaluation: Once the model is trained, it needs to be evaluated on a separate validation dataset to assess its performance. This involves measuring metrics such as accuracy, precision, recall, and F1 score.

6. Hyperparameter tuning: To improve the model's performance, some hyperparameters such as learning rate, batch size, and optimizer settings can be tuned. This process involves running multiple training experiments with different hyperparameter configurations.

7. Model deployment: If the model's performance is sufficiently satisfactory, you can deploy it to a production environment. This may involve converting the model to a format compatible with the deployment platform and integrating it with the application logic.

## 3.  Given a model `detr-resnet-50`, create an implementation of running this model with OpenVINO.

Refer to `README.md`.