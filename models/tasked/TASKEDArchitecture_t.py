import tensorflow as tf
from tensorflow.keras import layers, models

from .FeatureExtractor import FeatureExtractor
from .ActivityClassifier import ActivityClassifier
from .SubjectDiscriminator import SubjectDiscriminator

class TASKEDArchitecture(models.Model):
    def __init__(self, Cs, W, S, na, N):
        super(TASKEDArchitecture, self).__init__()

        self.feature_extractor = FeatureExtractor(Cs, W, S)

        self.activity_classifier = ActivityClassifier(W, na)

        self.subject_discriminator = SubjectDiscriminator(N, W)

        self.info = f"Cs_{Cs}_W_{W}_S_{S}_na_{na}_N_{N}"
        self.model_name = "tasked"

    def call(self, x, training=False):
        # Data is input in b,W,c shape typical to TF, so we transpose to align with the model as desribed in the article
        x = tf.transpose(x, (0, 2, 1))

        feature_output = self.feature_extractor(x, training=training)

        activity_output = self.activity_classifier(feature_output, training=training)
        subject_output = self.subject_discriminator(feature_output, training=training)

        return activity_output, subject_output, feature_output




class TASKED(models.Model):
    def __init__(self, Cs, W, S, na, N):
        super(TASKED, self).__init__()

        self.feature_extractor = FeatureExtractor(Cs, W, S)
        self.activity_classifier = ActivityClassifier(W, na)

        self.info = f"Cs_{Cs}_W_{W}_S_{S}_na_{na}_N_{N}"
        self.model_name = "tasked"

    def call(self, x, training=False):
        # Data is input in b,W,c shape typical to TF, so we transpose to align with the model as desribed in the article
        x = tf.transpose(x, (0, 2, 1))

        x = self.feature_extractor(x, training=training)
        x = self.activity_classifier(x, training=training)

        return x
