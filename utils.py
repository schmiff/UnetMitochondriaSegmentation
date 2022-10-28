from keras.preprocessing.image import ImageDataGenerator
from numpy import ndarray

def createImageSegmentationGenerator(images: ndarray, masks: ndarray, augmentation_dict: dict, batch_size: int):
        """
        Normalize images and encode all existing mask pixel values to 0 and 1 with a treshhold of 0.5
        """
        image_gen = ImageDataGenerator(**augmentation_dict)
        mask_gen = ImageDataGenerator(**augmentation_dict)

        image_gen = image_gen.flow(images, batch_size=batch_size)
        mask_gen = mask_gen.flow(masks, batch_size=batch_size)

        return_gen = zip(image_gen, mask_gen)

        for (img, mask) in return_gen:
                img, mask = normalize_and_diagnose(img, mask)
                yield img, mask


def normalize_and_diagnose(img, mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask