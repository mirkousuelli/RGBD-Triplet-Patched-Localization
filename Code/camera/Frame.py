# Python imports

# External imports

# In-project imports
from camera.CameraObject import CameraObject


# TODO: wow, implement me please
class Frame(CameraObject):

    def __init__(self):
        super().__init__()
        # TODO: hey Marco, mi servirebbero questi due attributes per fare detection + matching
        self.key_points = None #
        self.descriptors = None

    # non sono troppo pratico nell'utilizzare le classi astratte e il compilatore mi dà degli alert di sintassi...
        # evito di toccare quello che stai facendo, te lo lascio solo come memo... poi quando hai finito questo
        # package lo integro nel Detector + Matcher perchè per ora l'ho fatto funzionare solo con image pure
	def setFeatures(self, _key_points, _descriptors):
		self.key_points = _key_points
		self.descriptors = _descriptors
