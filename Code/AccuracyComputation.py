from MeanAverageAccuracy import meanAverageAccuracy


dnn_acc, cv_acc = meanAverageAccuracy(threshold=10,
									  last_image=30,
									  merge_features=5000,
									  detect_method="ORB",
									  match_method="FLANN",
									  network_path="neuralnetwork/model/rgbd_triplet_patch_encoder_model_euclidean.pt",
									  ransac_iterations=5000,
									  patch_side=8)
print("The accuracy of the DNN ransac is: %s" % dnn_acc)
print("The accuracy of the classical CV ransac is: %s" % cv_acc)