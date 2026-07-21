"""Custom layer needed to load the exported .keras models.

Import this module before calling keras.models.load_model() so the
@register_keras_serializable decorator runs and Keras can find MatchedCrop.
"""
import keras
from keras import layers


@keras.saving.register_keras_serializable(package="final_project")
class MatchedCrop(layers.Layer):
    """Random crop while training, deterministic centre crop while evaluating."""

    def __init__(self, height, width, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.seed = seed
        self.random_crop = layers.RandomCrop(height, width, seed=seed)
        self.center_crop = layers.CenterCrop(height, width)

    def build(self, input_shape):
        self.random_crop.build(input_shape)
        self.center_crop.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        if training:
            return self.random_crop(inputs, training=True)
        return self.center_crop(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.height, self.width, input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"height": self.height, "width": self.width, "seed": self.seed})
        return config
