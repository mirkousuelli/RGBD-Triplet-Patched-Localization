# RGB-D Semantic Sampling
Image Analysis and Computer Vision course project 2021-2022, Politecnico di Milano.

This work analyses the Localization task of RGB-D images using a Deep Neural Network (DNN) tuned to improve the baseline performance of RANSAC by exploiting visual semantic information. We propose a DNN able to extract a semantic sampling distribution from paired key points to improve the Mean Average Accuracy (mAA), chosen as a reference metric. Next, the importance of the depth channel is shown by comparing the same DNN trained on RGB or RGB-D. Finally, Point Clouds are generated from paired images of the same scene, and Registration is performed to visualize the results in 3D space through the depth information.
