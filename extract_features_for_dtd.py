import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from lib.data_loader import load_dtd
from lib.feature_extractor import NASNetLargeExtractor

devices = tf.config.list_physical_devices('GPU')
if len(devices) < 1:
    raise AttributeError("No GPU found!")
else:
    print(devices)
    print()

batch_size = 128

# load the dataset
x_train, y_train, x_valid, y_valid, x_test, y_test = load_dtd()
print("There are {} training samples and {} validation samples".format(x_train.shape[0], x_valid.shape[0]))
print("There are {} test samples.".format(x_test.shape[0]))

y_train = to_categorical(y_train, num_classes=47)
y_valid = to_categorical(y_valid, num_classes=47)

# download google nasnet large pre-trained model
model = NASNetLargeExtractor(331, 47, model_path="models/dtd", data_path="datasets/dtd", require_resize=False)
print("Pre-trained NASNetLarge is loaded.")

# features_train = model.extract(x_train, batch_size=batch_size)
# print("The shape of the extracted training sample features is: ", features_train.shape)
#
# # save features
# model.save_features()
#
# # use dense layer to test feature quality
# history = model.train_classifier(y_train, epochs=100, batch_size=batch_size, validation_data=(x_valid, y_valid))
# model.save_history(history, name="train_classifier_his")
#
# history = model.train_classifier(y_train, epochs=100, batch_size=batch_size, validation_data=(x_valid, y_valid),
#                                  learning_rate=0.001)
# model.save_history(history, name="train_classifier_his_2")
#
# history = model.train_classifier(y_train, epochs=100, batch_size=batch_size, validation_data=(x_valid, y_valid),
#                                  learning_rate=0.0001)
# model.save_history(history, name="train_classifier_his_3")
#
# # save trained model
# model.save_classifier()
# model.save_extractor()

# save compressed features
model.load_features()
model.load_classifier()
model.load_extractor()
model.extract(model.features, y_train, batch_size=batch_size, compression=True)
model.save_features()
