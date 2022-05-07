from MeanAverageAccuracy import MeanAverageAccuracy

ACCURACY_PATH = "accuracy_computation/maa1.json"
ALSO_COMPUTE = True

# Create the mAA computation object
mAA = MeanAverageAccuracy(threshold=10,
						  frames_distance=20,
						  last_image=832,
						  merge_features=5000,
						  detect_method="ORB",
						  match_method="FLANN",
						  network_path="neuralnetwork/model/rgbd_triplet_patch_encoder_model_euclidean.pt",
						  ransac_iterations=5000,
						  patch_side=8)

# If the accuracy must be computed, we compute it
if ALSO_COMPUTE:
	dnn_acc, cv_acc = mAA.compute_metric()
	mAA.save_to_file(ACCURACY_PATH, dnn_acc, cv_acc)

# Prints the accuracy
mAA.print_accuracy_computation(ACCURACY_PATH)