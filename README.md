# Multimodal survival prediction for pancreatic ductal adenocarcinoma

Welcome to the official GitHub repository for our '' 2024 paper, "_End-to-end prognostication in pancreatic cancer by multimodal deep learning: a retrospective, multi-centre study_". This project was developed by the [Diagnostic Image Analysis Group](https://www.diagnijmegen.nl/) at Radboud University Medical Center. 

![pipeline_2](https://github.com/meganschuurmans/pdac-survival-prediction/assets/86470433/f47fbed8-18fd-418e-84d6-c905c2db449a)

## Highlights

In our study, we explore the integration of clinical data and contrast-enhanced computed tomography (CECT) at time of diagnosis of pancreatic ductal adenocarcinoma (PDAC) to enhance patient survival prediction. We focus on addressing two key challenges: (1) developing unimodal AI models for clinical data and CECT scans (2) developing a multimodal AI system combining the short-term survival riskscores of both unimodal models. Our proposed model surpasses the current TNM stagig system in providing prognosis, showcasing stable performance across two external validation cohorts. 

## Installation Guide for Linux 

### The development data and internal test set
To download development data and internal test set for the unimodal model for CECT, please refer to [the PANORAMA study](https://zenodo.org/records/10599559) dataset. [Labels and clinical data of the PANORAMA study dataset](https://github.com/DIAGNijmegen/panorama_labels) are also made publicly available. 

### Processing CECT images
To preprocess your CECT images, please segment the pancreas parenchyma and the tumor using a segmentation tool (e.g. ]ITKsnap 3.8](http://www.itksnap.org/pmwiki/pmwiki.php)). After segmentating the pancreas parenchyma and lesion, mask out the CECT and crop to (96, 256, 256) around the pancreas. This can all be automatically done with the [PDAC detection algorithm](https://grand-challenge.org/algorithms/pdac-detection/) by Alves et al. (2022). Lastly, the cropped and masked CECT needs to be clipped in HU intensity to account for high attenuation creating noise in the masked area. 

### Training-Validation Splits
For evaluating the algorithm's performance, we partitioned each dataset using 5-fold cross-validation (stratified randomly). Splits for each cancer type are found in the splits folder, which each contain splits_{k}.csv for k = 1 to 5. In each splits_{k}.csv, the first column corresponds to the PANORAMA IDs used for training, and the second column corresponds to the PANORAMA IDs used for validation. Alternatively, one could define their own splits, however, the files would need to be defined in this format. The dataset loader for using these train-val splits are defined in the '' function in the ''.

### Running Experiments
Refer to scripts folder for source files to train the unimodal AI models and the multimodal AI models presented in the paper. Refer to the paper to find the hyperparameters required for training.

### Issues
The preferred mode of communication is via GitHub issues.
If GitHub issues are inappropriate, email megan.schuurmans@radboudumc.nl.
Immediate response to minor issues may not be available.
License and Usage
If you find our work useful in your research, please consider citing our paper at:
@article{schuurmans2024pdacsurvpred,
  title={End-to-end prognostication in pancreatic cancer by multimodal deep learning: a retrospective, multi-centre study},
  author={Schuurmans, Megan and Saha, Anindo and Alves, Natalia and Vendittelli, Pierpaolo and Yakar, Derya and Sabroso, Sergio and Malats, Nuria and Huisman, Henkjan and Hermans, John and Litjens, Geert},
  journal={},
  year={2024}
}
DIAG - This code is made available under is available for non-commercial academic purposes.

