from camera.Action import Action
from camera.Frame import Frame
from tools.Localizer import Localizer
from tools.Merger import Merger
from utils.utils import get_str

verbose = True
where_to_write = "localization/01.matches"
matches_str = "1 0 0 0 0 0 0"
for i in range(1, 834, 1):
	action = Action(Frame("../Dataset/Colors/00000-color.png",
						  "../Dataset/Depths/00000-depth.png",
						  0),
					Frame("../Dataset/Colors/00" + get_str(i) + "-color.png",
						  "../Dataset/Depths/00" + get_str(i) + "-depth.png",
						  i))
	
	if verbose:
		print("Executing step: ", i)
	
	merger = Merger(num_features=5000,
					detector_method="ORB",
					matcher_method="FLANN")
	merge_image = merger.merge_action(action)
	
	Localizer.roto_translation(action, normalize_em=False)
	q = Localizer.from_rot_to_quat(action, normalize_em=False)
	matches_str += "\n" + str(q[0]) + str(q[1]) + str(q[2]) + str(q[3])
	matches_str += str(action.t[0]) + str(action.t[1]) + str(action.t[2])
	
f = open(where_to_write)
f.write(matches_str)
f.close()
