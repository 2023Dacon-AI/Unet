import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.simplefilter("ignore")

# --------------------------------------------------------------------
# Helpful functions
# --------------------------------------------------------------------

def post_transform(image, **kwargs):
    if image.ndim == 3:
        return image.transpose(2, 0, 1).astype("float32")
    else:
        return image.astype("float32")



# --------------------------------------------------------------------
# Segmentation transforms
# --------------------------------------------------------------------

post_transform_fn = A.Lambda(name="post_transform", image=post_transform, mask=post_transform)

# crop 224
train_transform_1 = A.Compose([
    A.RandomCrop(224, 224, p=1.),
    A.Flip(p=0.75),
    A.RandomBrightnessContrast(p=0.5),
    post_transform_fn,
])

test_transform_1 = A.Compose([
    post_transform_fn,
])

# crop 224 and very hard augs
train_transform_2 = A.Compose([

    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7),
    A.PadIfNeeded(224, 224, border_mode=0, value=0, p=1.),
    A.RandomCrop(224, 224, p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
    A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),


    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
        ],
        p=0.8,
    ),

    # distortion
    A.OneOf(
        [
            A.ElasticTransform(p=1),
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
            A.Perspective(p=1),
        ],
        p=0.2,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            A.Sharpen(p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
    ),
    post_transform_fn,
])
