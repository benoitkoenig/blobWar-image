from boxes import get_boxes
from models.classifier import Classifier
from models.feature_mapper import FeatureMapper
from models.regr import Regr
from models.roi_pooling import RoiPooling
from models.rpn import Rpn

feature_mapper = FeatureMapper()
rpn = Rpn()
roi_pooling = RoiPooling()
classifier = Classifier()
regr = Regr()

feature_mapper.load_weights("./weights/feature_mapper")
rpn.load_weights("./weights/rpn")
classifier.load_weights("./weights/classifier")
regr.load_weights("./weights/regr")

def get_prediction(img):
    features = feature_mapper(img)
    rpn_map = rpn(features)

    boxes, probs = get_boxes(rpn_map)
    feature_areas = roi_pooling(features, boxes)

    classification_logits = classifier(feature_areas)
    regression_values = regr(feature_areas)

    return boxes, probs, classification_logits.numpy(), regression_values.numpy()
