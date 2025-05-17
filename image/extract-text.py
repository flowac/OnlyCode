import matplotlib.pyplot as plt
import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images
images = [
    keras_ocr.tools.read(img) for img in [
        'test1.png',
        'test2.png',
    ]
]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
prediction_groups = pipeline.recognize(images)

# Plot the predictions
#fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
#for ax, image, predictions in zip(axs, images, prediction_groups):
#    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

for index, predicted_image in enumerate(prediction_groups):
    print(index, end=':')
    for text, box in predicted_image:
        print(" {0}".format(text), end='')
    print(".")

