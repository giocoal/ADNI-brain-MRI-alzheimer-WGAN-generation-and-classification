# ADNI Brain T1-Weighted MRI Images Classification and WGAN Generation (Alzheimer's and Healthy patients) for the purpose of data augmentation. Implemented in TensorFlow, trained on ADNI dataset.

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

## Table of contents
* [Abstract](#abstract)
* [Report](https://www.slideshare.net/Giorgio469575/cxracgan-auxiliary-classifier-gan-for-conditional-generation-of-chest-xray-images-pneumonia-covid19-and-healthy-patients-255905299)
* [Requirements](#requirements)
* [COVIDx CXR-3: Dataset and Image Pre-processing](#covidx-cxr-3-dataset-and-image-pre-processing)
* [AC-GAN Training and Generation](#ac-gan-training-and-generation)
* [AC-GAN Evaluation: FID, Intra FID, Inception Score (IS), t-SNE](#ac-gan-evaluation-fid-intra-fid-inception-score-is-t-SNE)
* [Chest X-Ray Classification: Pneumonia and COVID-19 detection with GAN augmentation](#chest-x-ray-classification-pneumonia-and-covid-19-detection-with-gan-augmentation)
* [Status](#status)
* [Contact](#contact)
* [License](#license)
* [Contributing](#contributing)

## Abstract

This project focused on tackling Alzheimer's disease through three main objectives. Firstly, a dataset of axial 2D slices was created from 3D T1-weighted MRI brain images, integrating clinical, genetic, and biological sample data. Secondly, a deep neural network was trained to classify these images, distinguishing between healthy individuals and those with Alzheimer's. Lastly, different techniques for managing class imbalance were evaluated to improve the classifier's performance and reduce bias. The ultimate goal was to predict diagnoses solely based on cognitive assessments and clinical data. Finally, a generative model (Wasserstein GAN) was trained on Alzheimer's Disease and Healthy for data augmentation and class balancing. This approach overcame the limitations of traditional undersampling, avoiding data loss and replacing problematic geometric data augmentation in the medical imaging context. The Wasserstein GAN provided an effective method to generate new samples of the minority class, contributing to improved performance of the final classifier.

![GenetationExample](https://raw.githubusercontent.com/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification/main/reports/Generation_Example.png?token=GHSAT0AAAAAABXWNKOGR3AJPT65PXRQXNNUZEJW6TA)

## Requirements

- python==3.8
- ipython
- ipykernel
- matplotlib
- pandas
- seaborn
- tqdm==4.64.1
- scikit-learn==1.2.0
- glob2==0.7
- keras==2.10.0
- Keras-Preprocessing==1.1.2
- tensorflow==2.10.1
- numpy==1.24.2
- opencv_python==4.7.0.68
- pandas==1.5.3
- Pillow==9.4.0
- imageio==2.25.0

## ADNI: Image Dataset Acquisition

0. Request access to the Alzheimer's Disease Neuroimaging Initiative: https://adni.loni.usc.edu/data-samples/access-data/ (tipically 1 week  to be accepted)
1. Visit the following website: https://ida.loni.usc.edu/login.jsp?project=ADNI&page=HOME#
2. Sign up if you haven't already.
3. Click on the "SEARCH" button.
4. Select "Advanced Image Search (beta)."
5. In the "Search Options" menu, choose "Pre-processed" under the "IMAGE TYPES" sub-menu.
6. In the "Search Criteria" menu, select "MRI" under the "IMAGE" sub-menu. Then, under the "IMAGING PROTOCOL" sub-menu, choose "T1" for the weighting (Do not select "AXIAL" under the "Aquisition Plane" option, as most of the original data are volumetric .nii files.).
7. To download subjects with Alzheimer's Disease (AD), go to the "Search Options" menu, select "Subject" under the "Search Sections" sub-menu, and then choose "AD" under the "subject" sub-menu. The same process applies to Normal Control (NC) subjects.
8. Download only the "MPR; GradWarp; B1 Correction; N3; Scaled" images from the downloaded data.
9. For compatibility, store the downloaded images of AD subjects in a directory named 'ad' and the images of NC subjects in a directory named 'nor'.

## Image Preprocessing

0. Go in the `scripts/1_preprocessing` folder
1. Run the `1_ttv_split.py` script to randomly split the patients into training, validation, and test sets.
2. Optionally, you can run the `2_subjects_lists.py` script to log which subjects are assigned to each set.
3. Run the `3_nii_to_png_all.py` script to convert the volumetric .nii data to axial .png MRI images.
4. Optionally, use the `4_check_shapes.ipynb` notebook to determine the number of different slices and dimensions for each visit's corresponding images.
5. Run the `5_select_sequences_ttv.py` script to discard the images that are irrelevant for AD diagnosis, focusing only on the relevant parts of a subject's head.
6. Finally, execute the `6_resize_images.py` script to resize all the images to a consistent size of (192, 160).

## Wasserstein GAN Training and Generation

0. Go in the `scripts/3_GAN/WGAN` folder
1. Execute `wgan_bs_32_lat_128_eps_600.py` to train the GAN
2. Execute `generate_imgs.py` to generate the images

## Random and Stratified Undersampling

0. Go to the `scripts/2_class_imbalance_management` folder
1. Execute the various notebooks in it to perform class balancing

## Brain MRI Classification: Alzheimer's Disease (AD) and Cognitive Normal (CN) detection

Use the notebooks in `scripts/3_Resnet18_first` folder to train and evaluate classification models for the detection of Alzheimer's Disease  and Absence of symptoms in Brain MRI. Some forms of data augmentation are tested, including generation by trained W-GAN. 

## Status

 Project is: ![##c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)  _Done_

## Contact

[Giorgio Carbone](https://github.com/giocoal) - feel free to contact me!


## License
* >You can check out the full license [here](https://github.com/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification/blob/main/README.md)

This project is licensed under the terms of the **MIT** license.

## Contributing

1. Fork it (<https://github.com/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification.git>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

# Contributors

* [Giorgio Carbone](https://github.com/giocoal) 

<!-- Project is: ![##c5f015](https://via.placeholder.com/15/c5f015/000000?text=+)  _Done_
 Project is: ![##ff0000](https://via.placeholder.com/15/ff0000/000000?text=+)  _Under-Proccess_

[![Build](https://github.com/SimonIT/spotifylyrics/workflows/Build/badge.svg)](https://github.com/SimonIT/spotifylyrics/actions?query=workflow%3ABuild)
[![Current Release](https://img.shields.io/github/release/SimonIT/spotifylyrics.svg)](https://github.com/SimonIT/spotifylyrics/releases)
[![License](https://img.shields.io/github/license/SimonIT/spotifylyrics.svg)](https://github.com/SimonIT/spotifylyrics/blob/master/LICENSE)
[![GitHub All Releases](https://img.shields.io/github/downloads/SimonIT/spotifylyrics/total)](https://github.com/SimonIT/spotifylyrics/releases)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification.svg?style=for-the-badge
[contributors-url]: https://github.com/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification.svg?style=for-the-badge
[forks-url]: https://github.com/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification/network/members
[stars-shield]: https://img.shields.io/github/stars/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification.svg?style=for-the-badge
[stars-url]: https://github.com/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification/stargazers
[issues-shield]: https://img.shields.io/github/issues/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification.svg?style=for-the-badge
[issues-url]: https://github.com/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification/issues
[license-shield]: https://img.shields.io/github/license/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification.svg?style=for-the-badge
[license-url]: https://github.com/giocoal/ADNI-brain-MRI-alzheimer-WGAN-generation-and-classification/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/giorgio-carbone-63154219b/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
