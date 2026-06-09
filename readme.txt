The complete dataset folder structure is explain in DirectoryStructure.png file. 
In this dataset we have collected two types of images that contain RBC elements.
1) Healthy Images
2) Anemic Images

Further these images is divided into two datasets. i.e. AneRBC-I and AneRBC-II

AneRBC-I:
AneRBC-I consists of 1000 original images (500 healthy and 500 anemic images) with 1224x960 pixels resolution. These 1000 original images are further processed and generated 2000 images with 1000 binary and 1000 RGB ground truth respectively. The CBC and morphology reports of all 1000 images with pathologist’s interpretation are also included in the database. Resultantly, 3000 images (1000 original, 1000 binary, and 1000 RGB ground-truth) were stored in the AneRBC-I data repository along with CBC, morphology, and doctor decision report of each image.

AneRBC-II:
Due to high resolution, most ML-Models cannot process these images in the current resolution size, i.e., 1224x960. So, each image is cropped and further divided into twelve images with 306x320 pixels resolution. As a result, AneRBC-II comprises 12,000 original images (6000 healthy and 6000 anemic images) along with 12,000 binary and 12,000 RGB ground truths. Resultantly, AneRBC-II stored a total of 36,000 (12,000 original and 12,000 binary and 12,000 RGB ground truth) images.

Image name representation:
The AneRBC-I images are named with annotations xxx_a.png and xxx_h.png. Here xxx is a three-digit integer counter, "a" represents images with anemic RBC elements, and "h" represents healthy elements. The naming annotation for AneRBC-II is xxxx_yy_a.png and xxxx_yy_h.png. The AneRBC-II dataset is the sub-division of AneRBC-I. Each image in AneRBC-I is divided into 12 sub-images. So, here "xxxx" represents the 4-digit counter for the total number of images, while "yy" represents the counter for each image's sub-division. The naming representation of relevant manually segmented images is also the same, but the segmented images are placed in a separate directory named "segmented". 

Reports:
This folder contains two sub-folders named as Healthy and Anemic. Both Anemic and healthy folders are again divided into 2 sub-folder named as CBC and Morphology. Both CBC and Morphology folders contains relevant reports in .txt file format. The name of the each report is set as the name of the original image, that create analogy to create connection between image and its CBC and morphology reports.  

How to upload image data ,  CBC and morphology reports for training and testing:
A complete executable code is given in "upload_data_for_training.ipynb" that will guide the user how to upload image data ,  CBC and morphology reports from the dataset for training and testing of CNN model. 