# Project-FaceMaskDetection

In this project, an image classifier has been desgined based on the ResNet graph on Pytorch.
The dataset contains 9000 images of three classes: with face mask, without face mask and not a person.
please follow the instruction below so as to run the project.

## How to train the model?

- Find the [train.py](https://github.com/pedramfekri/Project-FaceMaskDetection/blob/master/Train/train.py) in the [Train](https://github.com/pedramfekri/Project-FaceMaskDetection/tree/master/Train) folder.
- Run [train.py](https://github.com/pedramfekri/Project-FaceMaskDetection/blob/master/Train/train.py)

After running the aforementioned module, the model will be trained and the accuracy of the model will be ploted for both training and validation data. In addition, the loss for each iteration will be plotted. Then, the model will be saved and the evaulation phase will be started on the test data. Finally, the performance of the model on the test data will be investigated using different metrics.

## How to test the model?

The saved model in the section above will be used in order to evaluate the model on the entire dataset. Becuase the dataset was already splitted in the training phase randomly, it is not possible to assess the performance of the model on the test subset splitted in the training phase. To this end, this module evaluates the model on the entire dataset and the performance might be slightly better than the reported result in the project document. Also, here is a [pretrained model](https://github.com/pedramfekri/Project-FaceMaskDetection/tree/master/Train) you can use, if you do not intend to train the model from scratch.
Please go through the following step for testing the model:

- Find the [inference.py](https://github.com/pedramfekri/Project-FaceMaskDetection/blob/master/Evaluation/inference.py) in the [Evaluation](https://github.com/pedramfekri/Project-FaceMaskDetection/tree/master/Evaluation) folder.
- Run [inference.py](https://github.com/pedramfekri/Project-FaceMaskDetection/blob/master/Evaluation/inference.py)

## Inference engine with live predictions on the video stream of a camera.
The saved model is also utilized in an inference engine with the aim of making predictions on the input data received from a camera stream. Having a camera installed, you can run the inference engine through the following steps:

- Find the [live_inference.py](https://github.com/pedramfekri/Project-FaceMaskDetection/blob/master/Inference/live_inference.py) in the [Evaluation](https://github.com/pedramfekri/Project-FaceMaskDetection/tree/master/Inference) folder.
- Run [live_inference.py](https://github.com/pedramfekri/Project-FaceMaskDetection/blob/master/Inference/live_inference.py).

You will see the a frame showing your camera stream as well as the predicted class labels in the terminal. 
