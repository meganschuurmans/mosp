
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import SimpleITK as sitk
from .utils.efficientnet import EfficientNetBN, EfficientNetBNFeatures
from torch.utils.data import Dataset, DataLoader
import os

class BinaryClassification2Layer(nn.Module):
    def __init__(self):
        super(BinaryClassification2Layer, self).__init__()
        # Modality 'rad': Number of input features is 7, 9, 12.
        # Modality 'rad_path': Number of input features is 8.
        self.layer_1 = nn.Linear(6, 128) 
        self.layer_2 = nn.Linear(128, 128)
        self.layer_out = nn.Linear(128, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

# acquire clinical features as tensor for patient ID
#clinic_age_diagnosis	lab_base_CA19_9	lab_base_CEA	clinic_BMI	clinic_weightloss	clinic_ECOG	chemo	base_lesionsize	    base_lesionlocation	    clinic_TNM_tstage	clinic_TNM_nstage	clinic_TNM_mstage: 13
#clinic_age_diagnosis	lab_base_CA19_9	lab_base_CEA	clinic_BMI	clinic_weightloss	clinic_ECOG	chemo	base_lesionsize	    base_lesionlocation: 10
#clinic_age_diagnosis	lab_base_CA19_9	lab_base_CEA	clinic_BMI	clinic_weightloss	clinic_ECOG: 6
def clinic_features_for_pat_id(args, pat_id):
    clinical_features_marksheet = args.clinical_variables_dir
    overview = pd.read_excel(clinical_features_marksheet)
    clinical_features = overview.loc[overview[overview.columns[0]] == int(pat_id)].iloc[:,1:7].values
    x_train = torch.tensor(clinical_features, dtype=torch.float32).to('cuda')     
    return x_train


# normalize features to range 0-1
def norm(features, stats=[0,1], mode='train'):
    if mode=='train':
        return (features - features.min()) / (features.max() - features.min()), \
               (features.min(), features.max())
    if mode=='test':
        return (features - stats['train_norm_min']) / \
               (stats['train_norm_max'] - stats['train_norm_min'])
 
    
# load training/validation dataset given overview marksheet
def load_dataset(data_overview_path):
    data_overview = pd.read_excel(data_overview_path)
    all_pat_ids = [str(pat_id) for pat_id in data_overview['pat_ids']] 
    all_labels = [int(lbl) for lbl in data_overview['lbls']] 
    all_images = \
        [np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(\
         img_dir)), axis=(0,1))\
         for img_dir in data_overview['img_paths']]
    
    return all_images, all_labels, all_pat_ids

# load trained EfficientNet-B0 imaging model given checkpoint path
def load_efficientnet_b0_imaging_model(imaging_model_ckpt_path, head=False):
    
    if not head:
        imaging_model = EfficientNetBNFeatures(model_name = 'efficientnet-b4', 
                                               in_channels = 1,
                                               image_size = [96, 192, 320], 
                                               num_classes = 2, 
                                               spatial_dims = 3).to('cuda')
    else:
        imaging_model = EfficientNetBN(model_name = 'efficientnet-b4', 
                                       in_channels = 1,
                                       image_size = [96, 192, 320], 
                                       num_classes = 2, 
                                       spatial_dims = 3).to('cuda')
    checkpoint = torch.load(imaging_model_ckpt_path)
    imaging_model.load_state_dict(checkpoint['model_state_dict'])
    imaging_model.to('cuda')  
    
    return imaging_model


# load trained 2-Layer NN clinical model given checkpoint path
def load_clinic_model(clinical_model_ckpt_path):
    clinical_model = BinaryClassification2Layer()
    checkpoint = torch.load(clinical_model_ckpt_path)
    clinical_model.load_state_dict(checkpoint['model_state_dict'])
    clinical_model.to('cuda') 
    return clinical_model

def model_paths(args):
    clinic_performances = pd.DataFrame(columns = ['runs', 'folds', 'aucs'])
    optimal_clinic_model_paths = {}

    for fold in range(args.num_folds):
        for run in os.listdir(args.clinical_model_dir):
            run_path = os.path.join(args.clinical_model_dir, run)
            metrics_path = os.path.join(run_path, 'log_F{}.xlsx'.format(fold))
            df_metrics = pd.read_excel(metrics_path)
            auc = df_metrics.loc[:,['test_AUC']].iloc[0].to_numpy()[0]
            new_row = {'runs': run,
                    'folds': 'F' + str(fold),
                        'aucs': auc}
            clinic_performances.loc[len(clinic_performances)] = new_row
            
    for fold in range(args.num_folds):
        max_auc = np.max(clinic_performances.loc[clinic_performances['folds'] == 'F' + str(fold), 'aucs'])
        run = clinic_performances.loc[clinic_performances['aucs'] == max_auc, 'runs'].iloc[0]
        
        optimal_clinic_model_paths['F{}'.format(fold)] = args.clinical_model_dir + str(run) + '/2_layer_nn_F' + str(fold) + '.pth'

    optimal_mm_model_paths = {}
    optimal_imaging_model_paths = {}
    for fold in range(args.num_folds):
        df_metrics = pd.read_excel(args.imaging_model_dir + 'efficientnet-b4_F{}_metrics.xlsx'.format(fold))
        df_metrics = df_metrics.sort_values(by=['valid_auroc'], ascending=False).head(5)
        epoch = np.max(df_metrics['epoch'])
        optimal_imaging_model_paths['F{}'.format(fold)] = args.imaging_model_dir+'efficientnet-b4_F{}_E{}.pt'.format(fold, epoch)
        optimal_mm_model_paths['F{}'.format(fold)] = 0
        # if args.ensemble and args.modal_type == 'mm':
        #     optimal_mm_model_paths['F{}'.format(fold)] = args.multimodal_dir + args.modal_type + '_ensemble_no_ema_F{}.joblib'.format(fold)
        if args.modal_type == 'mm':
            df_metrics = pd.read_excel('/data/pelvis/projects/megan/models/2023/os_prediction_mm/results/mm/'+ args.modal_type + '_validation_performances_no_ensemble_no_ema_all_epochs_F{}.xlsx'.format(fold))
            df_metrics = df_metrics.sort_values(by=['aucs'], ascending=False).head(5)
            epoch = np.max(df_metrics['epochs'])
            optimal_mm_model_paths['F{}'.format(fold)] = args.multimodal_dir + args.modal_type +  '_no_ensemble_no_ema_F{}_E{}.joblib'.format(fold, epoch)
            optimal_imaging_model_paths['F{}'.format(fold)] = args.imaging_dir +'efficientnet-b4_F{}_E{}.pt'.format(fold, epoch)

    return optimal_clinic_model_paths, optimal_imaging_model_paths, optimal_mm_model_paths