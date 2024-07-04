import torch
import os
import numpy as np
import pandas as pd
from .data_generator import load_dataset, load_efficientnet_b0_imaging_model, load_clinic_model, clinic_features_for_pat_id
from tqdm import tqdm
from .ensemble import imaging_ensemble, clinic_ensemble
from sklearn.metrics import roc_auc_score, roc_curve
from joblib import dump, load



def validation(args):
    for fold in range(args.num_folds):
        fpr, tpr, auc = [],[],[]
        if args.ensemble: 
            destination_validationsheet_path = args.results_dir + args.modal_type + '_validation_performances_ensemble_F{}.xlsx'.format(fold)
        else:
            destination_validationsheet_path = args.results_dir + args.modal_type + '_validation_performances_no_ensemble_F{}.xlsx'.format(fold)
        all_features = []
        all_valid_preds = []
        #-----------------------------------------------------------------------------------------
        print('-'*80)
        # load training dataset for given fold
        print(f"Loading Validation Dataset for Fold {fold}")
        all_valid_images, all_valid_labels, all_valid_ids = load_dataset(
            args.overview_dir + 'valid-fold-{}.xlsx'.format(fold))
        #-----------------------------------------------------------------------------------------
        if args.ensemble == True:
    #-----------------------------------------------------------------------------------------
    # ENSEMBLE VALIDATION
    #-----------------------------------------------------------------------------------------
            print(f'Ensemble is turned on')
            print("Loading {} LogRegr Model for Fold {}".format(args.modal_type, fold))
            mm_model = load(args.multimodal_model_dir + args.modal_type + '_ensemble_no_ema_F{}.joblib'.format(fold)) 

            if args.modal_type == 'mm':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #                                         -----> mm[F1, F2, F3, F4, F5]                                       
    # args.ensemble(uni_clinic[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------  
                with torch.no_grad():

                    # for all validation cases in given fold
                    for x in tqdm.tqdm(range(len(all_valid_images))):                
                        # inference with test-time augmentation
                        valid_image = torch.from_numpy(all_valid_images[x]).to('cuda')
                        valid_image = [valid_image, torch.flip(valid_image, [4]).to('cuda')]

                        imaging_prediction = imaging_ensemble(args, valid_image)
                        clinic_prediction = clinic_ensemble(args, all_valid_ids[x])

                        input_features =  np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)
                        mm_predictions = mm_model.predict_proba(
                            input_features) 
                        all_valid_preds.append(mm_predictions[0][1])

            elif args.modal_type == 'uni_clinic':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_clinic[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------
                with torch.no_grad():
                    # for all validation cases in given fold
                    for x in tqdm.tqdm(range(len(all_valid_images))):  
                        clinic_prediction = clinic_ensemble(args, all_valid_ids[x])

                        input_features =  clinic_prediction.squeeze().reshape(1, -1)
                        all_valid_preds.append(input_features)                    

            elif args.modal_type == 'uni_imaging':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------
                with torch.no_grad():
                    # for all validation cases in given fold
                    for x in tqdm.tqdm(range(len(all_valid_images))):  
                        # inference with test-time augmentation
                        valid_image = torch.from_numpy(all_valid_images[x]).to('cuda')
                        valid_image = [valid_image, torch.flip(valid_image, [4]).to('cuda')]

                        imaging_prediction = imaging_ensemble(args, valid_image)

                        input_features =  imaging_prediction.squeeze().reshape(1, -1)
                        all_valid_preds.append(input_features)                      
        else:
    #-----------------------------------------------------------------------------------------
    # NO ENSEMBLE VALIDATION
    #----------------------------------------------------------------------------------------
            print(f'Ensemble is turned off')
        
            if args.modal_type == 'mm':
    #-----------------------------------------------------------------------------------------
    # uni_imaging[F1, F2, F3, F4, F5])
    #                                 -----> mm[F1, F2, F3, F4, F5]                                       
    # uni_clinic[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------  
                print("Loading {} LogRegr Model for Fold {}".format(args.modal_type, fold))
                mm_model = load(
                    args.multimodal_model_dir + args.modal_type + '_F{}.joblib'.format(fold))
                 
                print("Loading Imaging and Clinical Model for Fold {}".format(fold))
                imaging_model = load_efficientnet_b0_imaging_model(
                    args.optimal_imaging_model_paths['F{}'.format(fold)], head=True)
                imaging_model.eval()

                clinic_model = load_clinic_model(
                    args.optimal_clinic_model_paths['F{}'.format(fold)])
                clinic_model.eval()

                with torch.no_grad():

                    # for all validation cases in given fold
                    for x in tqdm.tqdm(range(len(all_valid_images))):
                        valid_image = torch.from_numpy(all_valid_images[x]).to('cuda')
                        valid_image = [valid_image, torch.flip(valid_image, [4]).to('cuda')]
                        imaging_prediction = np.mean([
                                imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                for x in valid_image])     
                        
                        clinic_prediction = clinic_model(
                            clinic_features_for_pat_id(args, all_valid_ids[x])).sigmoid().squeeze().cpu().detach().numpy()  
                        
                        input_features = np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)
                        mm_predictions = mm_model.predict_proba(
                            input_features) 
                        all_valid_preds.append(mm_predictions[0][1])
                        all_features.append(imaging_prediction)
                        
            elif args.modal_type == 'uni_clinic':
    #-----------------------------------------------------------------------------------------
    # uni_clinic[F1, F2, F3, F4, F5]) 
    #----------------------------------------------------------------------------------------- 
                print("Loading Clinical Model for Fold {}".format(fold))
                clinic_model = load_clinic_model(
                    args.optimal_clinic_model_paths['F{}'.format(fold)])
                clinic_model.eval()
                with torch.no_grad():

                    # for all validation cases in given fold
                    for x in tqdm.tqdm(range(len(all_valid_images))):
                        imaging_prediction = 0

                        clinic_prediction = clinic_model(
                            clinic_features_for_pat_id(args,all_valid_ids[x])).sigmoid().cpu().detach().numpy()

                        input_features = clinic_prediction.squeeze().reshape(1, -1)
                        all_valid_preds.append(input_features[0][0])  
                            
            elif args.modal_type == 'uni_imaging':
    #-----------------------------------------------------------------------------------------
    # uni_imaging[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------
                print("Loading Imaging Model for Fold {}".format(fold))
                imaging_model = load_efficientnet_b0_imaging_model(
                    args.optimal_imaging_model_paths['F{}'.format(fold)], head=True)      
                imaging_model.eval()            
                with torch.no_grad():

                    # for all validation cases in given fold
                    for x in tqdm.tqdm(range(len(all_valid_images))):  
                        # inference with test-time augmentation
                        valid_image = torch.from_numpy(all_valid_images[x]).to('cuda')
                        valid_image = [valid_image, torch.flip(valid_image, [4]).to('cuda')]
                        imaging_prediction = np.mean([
                                imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                for x in valid_image])    

                        clinic_prediction = 0

                        input_features = imaging_prediction.squeeze().reshape(1, -1)

                        all_valid_preds.append(input_features)  
                        all_features.append(imaging_prediction)                        

        fpr_, tpr_, _ = roc_curve(all_valid_labels, all_valid_preds)
        auc_ = roc_auc_score(all_valid_labels, all_valid_preds)
        fpr.append(fpr_)
        tpr.append(tpr_)
        auc.append(auc_)  
        validation_sheet = pd.DataFrame()
        validation_sheet['aucs'] = auc
        validation_sheet['fprs'] = fpr
        validation_sheet['tprs'] = tpr  
        validation_sheet.to_excel(destination_validationsheet_path, index=False)
        print("Complete.")    

def validation_all_epochs(args):
    for fold in range(args.num_folds):
        if args.ensemble: 
            destination_validationsheet_path = args.results_dir + args.modal_type + '_validation_performances_ensemble_all_epochs_F{}.xlsx'.format(fold)
        else:
            destination_validationsheet_path = args.results_dir + args.modal_type + '_validation_performances_no_ensemble_all_epochs_F{}.xlsx'.format(fold)
        fpr, tpr, auc = [],[],[]
        if not os.path.exists(destination_validationsheet_path):
            # load training dataset for given fold
            print("Loading Validation Dataset for Fold {}".format(fold))
            all_valid_images, all_valid_labels, all_valid_ids = load_dataset(
                args.overview_dir + 'valid-fold-{}.xlsx'.format(fold))
            for epoch in range(args.num_epochs):
                all_valid_preds = []
                all_features = []
                #-----------------------------------------------------------------------------------------
                print('-'*80)
                #-----------------------------------------------------------------------------------------
                # LogReg args.ensemble validation
                if args.ensemble:
    #-----------------------------------------------------------------------------------------
    # ENSEMBLE VALIDATION
    #-----------------------------------------------------------------------------------------
                    print('Ensemble is turned on')
                    print("Loading {} LogRegr Model for Fold {}".format(args.modal_type, fold))
                    mm_model = load(
                        args.multimodal_model_dir + args.modal_type + '_ensemble_no_ema_F{}_E{}.joblib'.format(fold, epoch))     

                    if args.modal_type == 'mm':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #                                         -----> mm[F1, F2, F3, F4, F5]                                       
    # args.ensemble(uni_clinic[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------                              
                        with torch.no_grad():

                            # for all validation cases in given fold
                            for x in tqdm.tqdm(range(len(all_valid_images))):
                                # inference with test-time augmentation
                                valid_image = torch.from_numpy(all_valid_images[x]).to('cuda')
                                valid_image = [valid_image, torch.flip(valid_image, [4]).to('cuda')]

                                imaging_prediction = imaging_ensemble(args, valid_image)
                                clinic_prediction = clinic_ensemble(args, all_valid_ids[x])

                                input_features =  np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)
                                mm_predictions = mm_model.predict_proba(
                                    input_features) 
                                all_valid_preds.append(mm_predictions[0][1])
                                
                    elif args.modal_type == 'uni_imaging':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #-----------------------------------------------------------------------------------------                              
                        with torch.no_grad():

                            # for all validation cases in given fold
                            for x in tqdm.tqdm(range(len(all_valid_images))):
                                # inference with test-time augmentation
                                valid_image = torch.from_numpy(all_valid_images[x]).to('cuda')
                                valid_image = [valid_image, torch.flip(valid_image, [4]).to('cuda')]

                                imaging_prediction = imaging_ensemble(args, valid_image)

                                input_features =  imaging_prediction.squeeze().reshape(1, -1)
                                mm_predictions = mm_model.predict_proba(
                                    input_features) 
                                all_valid_preds.append(mm_predictions[0][1])
                #-----------------------------------------------------------------------------------------
                # LogReg validation       
                else:
    #-----------------------------------------------------------------------------------------
    # NO ENSEMBLE VALIDATION
    #-----------------------------------------------------------------------------------------   
                    print(f'Ensemble is turned off')
                    if args.modal_type == 'mm':
    #-----------------------------------------------------------------------------------------                                  
    # uni_imaging[F1, F2, F3, F4, F5]
    #                                -----> mm[F1, F2, F3, F4, F5]                                     
    # uni_clinic[F1, F2, F3, F4, F5]  
    #-----------------------------------------------------------------------------------------
                        print("Loading {} LogReg Model for validation of Fold {} and Epoch {}".format(args.modal_type, fold, epoch))
                        mm_model = load(args.multimodal_model_dir + 'without_chemo/' + args.modal_type + '_no_ensemble_no_ema_F{}_E{}.joblib'.format(fold, epoch))     

                        print("Loading Imaging and Clinical Model for Fold {} and Epoch {}".format(fold, epoch))
                        imaging_model = load_efficientnet_b0_imaging_model(
                            args.imaging_model_dir + 'efficientnet-b4_F{}_E{}.pt'.format(fold, epoch), head=True)
                        imaging_model.eval()

                        clinic_model = load_clinic_model(
                            args.optimal_clinic_model_paths['F{}'.format(fold)])
                        clinic_model.eval()

                        with torch.no_grad():
                            for x in tqdm.tqdm(range(len(all_valid_images))):                                
                                # inference with test-time augmentation
                                valid_image = torch.from_numpy(all_valid_images[x]).to('cuda')
                                valid_image = [valid_image, torch.flip(valid_image, [4]).to('cuda')]
                                imaging_prediction = np.mean([
                                        imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                        for x in valid_image])     

                                clinic_prediction = clinic_model(
                                    clinic_features_for_pat_id(args, all_valid_ids[x])).sigmoid().squeeze().cpu().detach().numpy()  

                                input_features = np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)
                                mm_predictions = mm_model.predict_proba(
                                    input_features) 
                                all_valid_preds.append(mm_predictions[0][1])

                    elif args.modal_type == 'uni_imaging':
    #-----------------------------------------------------------------------------------------                                  
    # uni_imaging[F1, F2, F3, F4, F5] 
    #-----------------------------------------------------------------------------------------

                        print("Loading Imaging Model for Fold {} and Epoch {}".format(fold, epoch))
                        imaging_model = load_efficientnet_b0_imaging_model(
                            args.imaging_model_dir + 'efficientnet-b4_F{}_E{}.pt'.format(fold, epoch), head=True) 
                        imaging_model.eval()

                        with torch.no_grad():
                            for x in tqdm.tqdm(range(len(all_valid_images))):                            
                                clinic_prediction = 0

                                valid_image = torch.from_numpy(all_valid_images[x]).to('cuda')
                                valid_image = [valid_image, torch.flip(valid_image, [4]).to('cuda')]
                                imaging_prediction = np.mean([
                                        imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                        for x in valid_image])         

                                input_features = imaging_prediction.squeeze().reshape(1, -1)
                                all_valid_preds.append(input_features)  
                                all_features.append(imaging_prediction)                                  

            fpr_, tpr_, _ = roc_curve(all_valid_labels, all_valid_preds)
            auc_ = roc_auc_score(all_valid_labels, all_valid_preds)
            if epoch > 0:
                if auc_ > np.max(auc):
                    print(f'Epoch {epoch} with AUC: {auc_}')
            fpr.append(fpr_)
            tpr.append(tpr_)
            auc.append(auc_)  

        validation_sheet = pd.DataFrame()
        validation_sheet['epochs'] = np.arange(0, 200, 1)
        validation_sheet['aucs'] = auc
        validation_sheet['fprs'] = fpr
        validation_sheet['tprs'] = tpr  
        validation_sheet.to_excel(destination_validationsheet_path, index=False)
    
        print("Complete.") 