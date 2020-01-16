
#from augmentation import augmentate
from PIL import Image, ImageOps
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import os
import hashlib
import time
from tqdm import tqdm
import json

ia.seed(1)


# augmentate images
def augmentate_image(images: list):
    seq = iaa.Sequential(
            [np.random.choice(
                [
                    iaa.Sometimes(0.85,
                        iaa.Crop(percent=(0, 0.1))
                    ),
                    iaa.Sometimes(0.85,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.Sometimes(0.85,
                        iaa.contrast.LinearContrast((0.75, 1.5))
                    ),
                    iaa.Sometimes(0.85,
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
                    ),
                    iaa.Sometimes(0.85,
                        iaa.Multiply((0.8, 1.2), per_channel=0.2)
                    ),
                    iaa.Sometimes(0.85,
                        iaa.Affine(
                            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                            rotate=(-4.5, 4.5),
                            shear=(-8, 8)
                        )
                    )
                ]
            ) for _ in range(4)], random_order=True)

    augmented = []
    # flipped
    flipped = iaa.Sequential([iaa.Fliplr(1)])(images=images)
    augmented.extend(flipped)
    # noised
    noised = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)])(images=images)
    augmented.extend(noised)
    # random on originals
    augmented.extend(seq(images=images))
    # random on flipped
    augmented.extend(seq(images=flipped))
    # origianls
    augmented.extend(images)

    return augmented

# reshape pil image (without respect to proportion)
def reshape(pil_image, size=(400, 400)):
    return ImageOps.fit(pil_image, size, Image.ANTIALIAS)

# apply augmentations and save new generated images
def apply_augmentation(folder_list: list, folder_in: str="", folder_out: str="", resize=None):
    
    """ 
        . applies augmentation to images
        . reshapes images
        . saves images with id

    """

    images = []
    for idx, image in enumerate(folder_list):
        try:
            image = np.array(Image.open(folder_in + "/" + image))
            augmented_images = augmentate_image([image])

            for image_matrix in augmented_images:
                img = Image.fromarray(image_matrix)

                if resize is not None:
                    img = reshape(img, size=resize)

                id_ = str(hashlib.sha256((str(idx) + str(time.time())).encode("utf-8")).hexdigest())[:24]
                prefix = folder_in.split("/")[-1].split("_")[0] + "_"
                img.save(folder_out + "/" + prefix + id_ + ".png")

            print("\npreprocessing " + folder_in + ":" + folder_out.split("/")[-1] + " image no.", idx, ", id:", id_)

        except IOError:
            pass

# split list into train-, test- and validation-set
def split(batch: list, testing_size: float=0.1, validation_size: float=0.1):
    test_size = int(np.round(len(batch)*testing_size))
    val_size = int(np.round(len(batch)*validation_size))
    train_set, test_set, validation_set = batch[(test_size+val_size):], batch[:test_size], batch[test_size:(test_size+val_size)]

    return [train_set, test_set, validation_set]

# create labels
def create_labels(folder_in: str, json_file: str):
    labels = []
    files = os.listdir(folder_in)
    for f in files:
        if f.split("_")[0] == "kitchen":
            labels.append([[folder_in + "/" + f], [1, 0, 0, 0]]) 

        elif f.split("_")[0] == "living":
            labels.append([[folder_in + "/" + f], [0, 1, 0, 0]]) 

        elif f.split("_")[0] == "bath":
            labels.append([[folder_in + "/" + f], [0, 0, 1, 0]]) 

        elif f.split("_")[0] == "bed":
            labels.append([[folder_in + "/" + f], [0, 0, 0, 1]]) 

    with open(json_file, "w") as f:
        json.dump(labels, f)
  
# renames files (helper method, not in usage)
def rename(folder, prefix=""):
    files = os.listdir(folder)
    for i in range(len(files)):
        os.rename(folder + files[i], folder + prefix + files[i])

# resize and apply augmentation on all images
def preprocess(raw_dataset_folder: str, finished_dataset_folder: str):
    """ split folders into three batches (train, test, validation) """
    kitchen_room = split(os.listdir(raw_dataset_folder + "/kitchen_room"), testing_size=0.1, validation_size=0.15)
    living_room = split(os.listdir(raw_dataset_folder + "/living_room"), testing_size=0.1, validation_size=0.15)
    bath_room = split(os.listdir(raw_dataset_folder + "/bath_room"), testing_size=0.1, validation_size=0.15)
    bed_room = split(os.listdir(raw_dataset_folder + "/bed_room"), testing_size=0.1, validation_size=0.15)

    """ augmentate, reshape, rename kitchen_room """
    apply_augmentation(kitchen_room[0], folder_in=(raw_dataset_folder + "/kitchen_room"), folder_out=(finished_dataset_folder + "/train"), resize=(475, 350))
    apply_augmentation(kitchen_room[1], folder_in=(raw_dataset_folder + "/kitchen_room"), folder_out=(finished_dataset_folder + "/test"), resize=(475, 350))
    apply_augmentation(kitchen_room[2], folder_in=(raw_dataset_folder + "/kitchen_room"), folder_out=(finished_dataset_folder + "/val"), resize=(475, 350))
    print("finished preprocessing 'kitchen_room'")

    """ augmentate, reshape, rename living_room """
    apply_augmentation(living_room[0], folder_in=(raw_dataset_folder + "/living_room"), folder_out=(finished_dataset_folder + "/train"), resize=(475, 350))
    apply_augmentation(living_room[1], folder_in=(raw_dataset_folder + "/living_room"), folder_out=(finished_dataset_folder + "/test"), resize=(475, 350))
    apply_augmentation(living_room[2], folder_in=(raw_dataset_folder + "/living_room"), folder_out=(finished_dataset_folder + "/val"), resize=(475, 350))
    print("finished preprocessing 'living_room'")

    """ augmentate, reshape, rename bath_room """
    apply_augmentation(bath_room[0], folder_in=(raw_dataset_folder + "/bath_room"), folder_out=(finished_dataset_folder + "/train"), resize=(475, 350))
    apply_augmentation(bath_room[1], folder_in=(raw_dataset_folder + "/bath_room"), folder_out=(finished_dataset_folder + "/test"), resize=(475, 350))
    apply_augmentation(bath_room[2], folder_in=(raw_dataset_folder + "/bath_room"), folder_out=(finished_dataset_folder + "/val"), resize=(475, 350))
    print("finished preprocessing 'bath_room'")

    """ augmentate, reshape, rename bed_room """
    apply_augmentation(bed_room[0], folder_in=(raw_dataset_folder + "/bed_room"), folder_out=(finished_dataset_folder + "/train"), resize=(475, 350))
    apply_augmentation(bed_room[1], folder_in=(raw_dataset_folder + "/bed_room"), folder_out=(finished_dataset_folder + "/test"), resize=(475, 350))
    apply_augmentation(bed_room[2], folder_in=(raw_dataset_folder + "/bed_room"), folder_out=(finished_dataset_folder + "/val"), resize=(475, 350))
    print("finished preprocessing 'bed_room'")
   

if __name__ == "__main__":
    raw_dataset = "dataset/raw_data"
    finished_dataset = "dataset/data"

    #preprocess(raw_dataset, finished_dataset)

    create_labels("dataset/data/train", "dataset/data/train_set.json")
    create_labels("dataset/data/test", "dataset/data/test_set.json")
    create_labels("dataset/data/val", "dataset/data/val_set.json")






"""rename("dataset/data/kitchen_room/train/", prefix="kitchen_")
rename("dataset/data/kitchen_room/test/", prefix="kitchen_")
rename("dataset/data/kitchen_room/val/", prefix="kitchen_")

rename("dataset/data/living_room/train/", prefix="living_")
rename("dataset/data/living_room/test/", prefix="living_")
rename("dataset/data/living_room/val/", prefix="living_")

rename("dataset/data/bath_room/train/", prefix="bath_")
rename("dataset/data/bath_room/test/", prefix="bath_")
rename("dataset/data/bath_room/val/", prefix="bath_")

rename("dataset/data/bed_room/train/", prefix="bed_")
rename("dataset/data/bed_room/test/", prefix="bed_")
rename("dataset/data/bed_room/val/", prefix="bed_")"""



    
