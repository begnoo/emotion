from os import path


class ImageSizeProps:
    image_height = 48
    image_width = 48
    window_size = 24
    window_step = 6


class ImageProps:
    IMAGES_PER_EMOTION = -1  # -1 if there is no upper limit
    IMAGES_PER_EMOTION = IMAGES_PER_EMOTION if IMAGES_PER_EMOTION >= 0 else float('inf')
    DATASET = 'fer2013'  # 'fer2013' | 'ck+' | 'ck+extracted'
    OUTPUT_FOLDER = path.join(DATASET + '_features' + str(IMAGES_PER_EMOTION if IMAGES_PER_EMOTION >= 0 else ''))


class FeatureExtractionProps:
    LANDMARKS = True
    HOG_FEATURES = True
    HOG_WINDOWS_FEATURES = True


SIZE_PROPS = ImageSizeProps()
IMAGE_PROPS = ImageProps()
FEATURE_PROPS = FeatureExtractionProps()
