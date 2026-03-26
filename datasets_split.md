# Tasks and Datasets
## 1. **TUMOR**


### - **Figshare:**


This brain tumor dataset containing 3064 T1-weighted contrast-inhanced images from 233 patients with three kinds of brain tumor.
Contains cjdata.PID: patient ID


_tumor only, no normal images_


link: [https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=51340418](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427?file=51340418)


### - **BR35H**


Br35H public dataset, which includes 801 annotated brain tumor MRI images. The dataset is divided into a training set (500 images), a validation set (201 images), and a test set (100 images), used for model training, validation, and testing, respectively.
Instead of the 3000 images yes/no -> use train, test, val split


_Other literature use random split of the full DS of 3000 images (80/20) for train/test:_ [https://pmc.ncbi.nlm.nih.gov/articles/PMC12852129/#Sec3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12852129/#Sec3)


link: [https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)


### - **17c**


Images of real exams, without any data from the patient's medical record, thus preserving their identity. Exams interpreted by radiologists and provided for study purposes.


link: [https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes)


### - **44c**


Images without any type of marking or patient identification, interpreted by radiologists and provided for study purposes.


link: [https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c)


## 2. **MS**


The study dataset comprised axial and sagittal brain MRI images that were prospectively acquired from 72 MS and 59 healthy subjects. The dataset was divided into three study subsets: axial images only (n = 1652), sagittal images only (n = 1775), and combined axial and sagittal images (n = 3427) of both MS and healthy classes.


link: [https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis/data](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis/data)


## 3. **STROKE**


### - **AISD**


345 scans are used to train and validate the model, and the remaining 52 scans are used for testing.
The data ids in test set are defines in the github repo.


link: [https://github.com/GriffinLiang/AISD](https://github.com/GriffinLiang/AISD)


### - **CT STROKE**


The data set was anonymized - no patient ids.


link: [https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset](https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset)


# Plan for splitting


## Tumor binary
### DS 1: Figshare
experiment 1

Use patient IDs provided for the 233 patients.
### DS 2: BR35H
experiment 2

Use the predefined 500/201/100 (train/val/test) split exactly as provided. Do not reshuffle. This also enables direct comparison with published baselines. 
## Tumor multiclass
Neither 17c nor 44c have patient ids - could also overlap among each other.

1. Deduplicate first: hash all images (MD5/perceptual hash) across both datasets and remove exact/near-duplicates before any split                                                                                                    
2. After deduplication, apply stratified image-level split (70/10/20) preserving class proportions across all 12 classes                                                                                                                
3. Note the limitation explicitly
### 17 classes
- Glioma (Astrocytoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependymoma)
- Meningioma (Low Grade, Atypical, Anaplastic, Transitional)
- Neurocytoma (Central - Intraventricular, Extraventricular)
- NORMAL
- Other Types of Injuries (Abscesses, Cysts, Miscellaneous Encephalopathies)
- Schwannoma (Acoustic, Vestibular - Trigeminal)
### 44 classes
- Astrocytoma, carcinoma, ependymoma, ganglioglioma, germinoma, glioblastoma, granuloma, medulloblastoma, meningioma, neurocytoma, oligodendroglioma, papilloma, schwannoma and tuberculoma.
## Stroke
### AISD 
Has patient IDs split.

The 52 test-set scan IDs from the GitHub repo must go only into test — extract their slices first and lock them
### CT Stroke 
No patient IDs.

→ available for train/val pool only (never test, since it can't be guaranteed no patient appears in AISD test) 
## MS
No patient IDs.

Stratified image-level split (70/10/20). Flag in limitations.