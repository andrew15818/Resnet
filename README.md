# ResNet Image Classifier
The goal of this project is to feel more comfortable with PyTorch by practicing image classification on a simple resnet.

My main reference is the [original paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).
In the paper, the authors develop residual networks with different amounts of residual blocks,
of which I chose ResNet18 because it would be good enough for this project.

For the dataset, originally I planned on using ImageNet since that is used in
many computer vision papers, but due to hardware constraints I went with 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) instead.
The dataset contains 50,000 32x32 training images belonging to 10 classes.

By default the model is trained for 90 epochs using the SGD optimizer and a cosine annealing learning rate scheduler.
The performance is as follows

|Model|Epochs|Dataset|Accuracy|
|:---:|:---:|:---:|:---:|
resnet18|90|CIFAR10||

### ToDo
- Remove the hard-coded block sizes to allow for larger models (e.g. R50).

