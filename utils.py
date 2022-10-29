from keras.preprocessing.image import ImageDataGenerator
from numpy import ndarray

def createImageSegmentationGenerator(images, masks, batch_size: int):
        """
        Normalize images and encode all existing mask pixel values to 0 and 1 with a treshhold of 0.5
        """
        image_gen = ImageDataGenerator()
        mask_gen = ImageDataGenerator()

        image_generator = image_gen.flow(x=images, y=masks, batch_size=batch_size, shuffle=True,seed=1)
        #mask_generator = mask_gen.flow(masks, batch_size=batch_size, shuffle=False, seed=1)

        #return_gen = zip(image_generator, mask_generator)

        for (img, mask) in image_generator:
                img, mask = normalize_and_diagnose(img, mask)
                yield img, mask


def normalize_and_diagnose(img, mask):
        img = img/255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return img, mask