import torch
import os
import numpy as np
import pandas as pd
from .data_generator import load_dataset, load_efficientnet_b0_imaging_model, load_clinic_model, clinic_features_for_pat_id
import tqdm
from .ensemble import imaging_ensemble, clinic_ensemble
from sklearn.metrics import roc_auc_score, roc_curve
from joblib import dump, load


def internal_testing(args):
    fpr, tpr, auc = [],[],[]
    pred_per_fold = []
    #-----------------------------------------------------------------------------------------
    print('-'*80)
    # load training dataset for given fold
    print(f"Loading Testing Dataset for testing")
    all_test_images, all_test_labels, all_test_ids = load_dataset(
        args.overview_dir + 'overview_testset.xlsx')
    all_test_preds = []

    if args.ensemble:
    #-----------------------------------------------------------------------------------------
    # ENSEMBLE TESTING
    #-----------------------------------------------------------------------------------------
        print(f'Ensemble is turned on')
        print(f"Loading {args.modal_type} Model for testing")
        if args.double_ensemble: 
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #                                          -----> args.ensemble(mm[F1, F2, F3, F4, F5])                                       
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------
            with torch.no_grad():
                    # for all test cases in given fold
                for x in tqdm.tqdm(range(len(all_test_images))):
                    # inference with test-time augmentation
                    test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                    test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                    imaging_prediction = imaging_ensemble(args, test_image)

                    clinic_prediction = clinic_ensemble(args, all_test_ids[x])

                    input_features =  np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)

                    mm_ensemble = []
                    for fold in range(args.num_folds):
                        mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)]) 
                        mm_predictions = mm_model.predict_proba(input_features) 
                        mm_ensemble.append(mm_predictions[0][1])
                    mm_ens = np.sum(mm_ensemble)/5                                       
                    all_test_preds.append(mm_ens)
            fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
            auc_ = roc_auc_score(all_test_labels, all_test_preds)
            fpr.append(fpr_)
            tpr.append(tpr_)
            auc.append(auc_)  
            pred_per_fold.append(all_test_preds)
        else:
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #                                          -----> mm[F1, F2, F3, F4, F5]                                       
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------
            if args.modal_type == 'mm' and args.early_ensemble:
                print("Loading {} LogRegr Model with early args.ensemble".format(args.modal_type))
                with torch.no_grad():
                        # for all test cases in given fold
                    for fold in range(args.num_folds):
                        all_test_preds = []
                        for x in tqdm.tqdm(range(len(all_test_images))):
                        # inference with test-time augmentation
                            test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                            test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                            imaging_prediction = imaging_ensemble(args, test_image)

                            clinic_prediction = clinic_ensemble(args, all_test_ids[x])

                            input_features =  np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)

                            mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)]) 
                            mm_predictions = mm_model.predict_proba(input_features)                                      
                            all_test_preds.append(mm_predictions[0][1])
                        fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
                        auc_ = roc_auc_score(all_test_labels, all_test_preds)
                        fpr.append(fpr_)
                        tpr.append(tpr_)
                        auc.append(auc_)  
                        pred_per_fold.append(all_test_preds)
    #-----------------------------------------------------------------------------------------
    # uni_imaging[F1, F2, F3, F4, F5]
    #                          ----> mm[F1, F2, F3, F4, F5]  ----> args.ensemble(mm[F1, F2, F3, F4, F5])                                   
    # uni_imaging[F1, F2, F3, F4, F5]
    #-----------------------------------------------------------------------------------------
            if args.modal_type == 'mm' and args.early_ensemble == False:
                print("Loading {} LogRegr Model with early args.ensemble".format(args.modal_type))
                with torch.no_grad():
                        # for all test cases in given fold
                    for fold in range(args.num_folds):
                        all_test_preds = []
                        for x in tqdm.tqdm(range(len(all_test_images))):
                        # inference with test-time augmentation
                            test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                            test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                            mm_prediction = mm_ensemble(args, test_image, all_test_ids[x])

                            input_features =  mm_prediction.squeeze().reshape(1, -1)

                            mm_ensemble_model = load(args.optimal_mm_ensemble_model_paths['F{}'.format(fold)]) 
                            mm_ensemble_predictions = mm_ensemble_model.predict_proba(input_features)                                      
                            all_test_preds.append(mm_ensemble_predictions[0][1])
                        fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
                        auc_ = roc_auc_score(all_test_labels, all_test_preds)
                        fpr.append(fpr_)
                        tpr.append(tpr_)
                        auc.append(auc_)  
                        pred_per_fold.append(all_test_preds)
            elif args.modal_type == 'uni_imaging':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #-----------------------------------------------------------------------------------------
                with torch.no_grad():
                        # for all test cases in given fold
                    for x in tqdm.tqdm(range(len(all_test_images))):
                        # inference with test-time augmentation
                        test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                        test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                        
                        imaging_prediction = imaging_ensemble(args, test_image)

                        input_features =  imaging_prediction.squeeze().reshape(1, -1)
                        all_test_preds.append(input_features[0][0])                    

            elif args.modal_type == 'uni_clinic':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_clinic[F1, F2, F3, F4, F5])
    #-----------------------------------------------------------------------------------------
                with torch.no_grad():
                        # for all test cases in given fold
                    for x in tqdm.tqdm(range(len(all_test_images))):
                        # inference with test-time augmentation
                        clinic_prediction = clinic_ensemble(args, all_test_ids[x])

                        input_features =  clinic_prediction.squeeze().reshape(1, -1)
                        all_test_preds.append(input_features[0][0])        

            else:
    #-----------------------------------------------------------------------------------------
    # uni_imaging[F1, F2, F3, F4, F5]
    #                                -----> args.ensemble(mm[F1, F2, F3, F4, F5])                                       
    # uni_clinic[F1, F2, F3, F4, F5] 
    #-----------------------------------------------------------------------------------------
                print("Loading {} LogRegr Model with late args.ensemble".format(args.modal_type))
                with torch.no_grad():
                        # for all test cases in given fold
                    for x in tqdm.tqdm(range(len(all_test_images))): 
                        mm_ensemble = []
                        test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                        test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                        
                        for fold in range(args.num_folds):
                            imaging_model = load_efficientnet_b0_imaging_model(
                                args.optimal_imaging_model_paths['F{}'.format(fold)], head=True)
                            imaging_model.eval()

                            clinic_model = load_clinic_model(
                                args.optimal_clinic_model_paths['F{}'.format(fold)])
                            clinic_model.eval()

                            imaging_prediction = np.mean([
                                    imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                    for x in test_image])                                     

                            clinic_prediction = clinic_model(
                                clinic_features_for_pat_id(args, all_test_ids[x])).sigmoid().squeeze().cpu().detach().numpy()  

                            input_features = np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)

                            mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)]) 
                            mm_predictions = mm_model.predict_proba(input_features) 
                            mm_ensemble.append(mm_predictions[0][1])
                        mm_ens = np.sum(mm_ensemble)/5                                       
                        all_test_preds.append(mm_ens)
            fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
            auc_ = roc_auc_score(all_test_labels, all_test_preds)
            fpr.append(fpr_)
            tpr.append(tpr_)
            auc.append(auc_)  
            pred_per_fold.append(all_test_preds)
    else:    
    #-----------------------------------------------------------------------------------------
    # NO ENSEMBLE TESTING
    #-----------------------------------------------------------------------------------------        
        print(f'Ensemble is turned off')
        if args.modal_type != 'nomogram_SEER' and args.modal_type != 'nomogram_dc':
            for fold in range(args.num_folds):
                all_test_preds = []
                if args.modal_type == 'mm':
        #-----------------------------------------------------------------------------------------                                  
        # uni_imaging[F1, F2, F3, F4, F5]
        #                                -----> mm[F1, F2, F3, F4, F5]                                     
        # uni_clinic[F1, F2, F3, F4, F5]  
        #----------------------------------------------------------------------------------------- 
                    print(f"Loading MM Model for Fold {fold}")
                    mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)])
                    clinic_model = load_clinic_model(args.optimal_clinic_model_paths['F{}'.format(fold)])
                    clinic_model.eval()                
                    imaging_model = load_efficientnet_b0_imaging_model(
                        args.optimal_imaging_model_paths['F{}'.format(fold)], head=True) 
                                        
                    imaging_model.eval()  
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_test_images))):
                            # inference with test-time augmentation
                            test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                            test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]

                            imaging_prediction = np.mean([
                                    imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                    for x in test_image])    
                            
                            
                            clinic_prediction = clinic_model(
                                clinic_features_for_pat_id(args, all_test_ids[x])).sigmoid().cpu().detach().numpy()  

                            input_features = np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)
                            mm_predictions = mm_model.predict_proba(
                                input_features)                                  
                            all_test_preds.append(mm_predictions[0][1])

                elif args.modal_type == 'uni_clinic':
        #-----------------------------------------------------------------------------------------                                  
        # uni_clinic[F1, F2, F3, F4, F5] 
        #-----------------------------------------------------------------------------------------   
                    print(f"Loading Clinical Model for Fold {fold}")
                    clinic_model = load_clinic_model(args.optimal_clinic_model_paths['F{}'.format(fold)])
                    clinic_model.eval()      
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_test_images))):
                            imaging_prediction = np.nan

                            clinic_prediction = clinic_model(
                                clinic_features_for_pat_id(args, all_test_ids[x])).sigmoid().cpu().detach().numpy()

                            input_features = clinic_prediction.squeeze().reshape(1, -1)
                            all_test_preds.append(input_features[0][0])  
                            
                elif args.modal_type =='uni_imaging':
        #-----------------------------------------------------------------------------------------                                  
        # uni_imaging[F1, F2, F3, F4, F5] 
        #-----------------------------------------------------------------------------------------
                    print(f"Loading Imaging Model for Fold {fold}")
                    imaging_model = load_efficientnet_b0_imaging_model(
                        args.optimal_imaging_model_paths['F{}'.format(fold)], head=True)  
                    imaging_model.eval()  
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_test_images))):                
                            clinic_prediction = np.nan

                            test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                            test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                            imaging_prediction = np.mean([
                                    imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                    for x in test_image])
                            
                            input_features = imaging_prediction.squeeze().reshape(1, -1)
                            all_test_preds.append(input_features[0][0])   
                fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
                auc_ = roc_auc_score(all_test_labels, all_test_preds)
                fpr.append(fpr_)
                tpr.append(tpr_)
                auc.append(auc_)  
                pred_per_fold.append(all_test_preds)

    validation_sheet = pd.DataFrame()
    validation_sheet['aucs'] = auc
    validation_sheet['fprs'] = fpr
    validation_sheet['tprs'] = tpr  

    prediction_sheet = pd.DataFrame()
    prediction_sheet['archiveID'] = all_test_ids
    num = 0
    for elem in pred_per_fold:
        prediction_sheet['F{}'.format(num)] = np.transpose(elem)
        num = num + 1
    prediction_sheet['labels'] = all_test_labels
    if args.modal_type == 'mm' and args.double_ensemble:
        validation_sheet.to_excel(args.results_dir + args.modal_type + '_test_performances_double_ensemble.xlsx', index=False)
        prediction_sheet.to_excel(args.results_dir + args.modal_type + '_test_predictions_double_ensemble.xlsx', index=False)
    elif args.early_ensemble:
        validation_sheet.to_excel(args.results_dir + args.modal_type + '_test_performances_early_ensemble.xlsx', index = False)
        prediction_sheet.to_excel(args.results_dir + args.modal_type + '_test_predictions_early_ensemble.xlsx', index=False)
    elif args.ensemble:
        validation_sheet.to_excel(args.results_dir + args.modal_type + '_test_performances_ensemble.xlsx', index = False)
        prediction_sheet.to_excel(args.results_dir + args.modal_type + '_test_predictions_ensemble.xlsx', index=False)
    else:
        validation_sheet.to_excel(args.results_dir + args.modal_type + '_test_performances_no_ensemble.xlsx'.format(fold), index=False)
        prediction_sheet.to_excel(args.results_dir + args.modal_type + '_test_predictions_no_ensemble.xlsx', index=False)
    print("Complete.")    


def external_testing(args):
    fpr, tpr, auc = [],[],[]
    pred_per_fold = []
    #-----------------------------------------------------------------------------------------
    print('-'*80)
    # load training dataset for given fold
    print(f"Loading Testing Dataset for external testing")
    all_test_images, all_test_labels, all_test_ids = load_dataset(
        args.external_dir + 'overview.xlsx')
    all_test_preds = []

    if args.ensemble:
    #-----------------------------------------------------------------------------------------
    # ENSEMBLE TESTING
    #-----------------------------------------------------------------------------------------
        print(f'Ensemble is turned on')
        print(f"Loading {args.modal_type} Model for testing")
        if args.double_ensemble: 
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #                                          -----> args.ensemble(mm[F1, F2, F3, F4, F5])                                       
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------
            with torch.no_grad():
                    # for all test cases in given fold
                for x in tqdm.tqdm(range(len(all_test_images))):
                    # inference with test-time augmentation
                    test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                    test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                    imaging_prediction = imaging_ensemble(args, test_image)

                    clinic_prediction = clinic_ensemble(args, all_test_ids[x])

                    input_features =  np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)

                    mm_ensemble = []
                    for fold in range(args.num_folds):
                        mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)]) 
                        mm_predictions = mm_model.predict_proba(input_features) 
                        mm_ensemble.append(mm_predictions[0][1])
                    mm_ens = np.sum(mm_ensemble)/5                                       
                    all_test_preds.append(mm_ens)
            fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
            auc_ = roc_auc_score(all_test_labels, all_test_preds)
            fpr.append(fpr_)
            tpr.append(tpr_)
            auc.append(auc_)  
            pred_per_fold.append(all_test_preds)
        else:
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #                                          -----> mm[F1, F2, F3, F4, F5]                                       
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------
            if args.modal_type == 'mm' and args.early_ensemble:
                print("Loading {} LogRegr Model with early args.ensemble".format(args.modal_type))
                with torch.no_grad():
                        # for all test cases in given fold
                    for fold in range(args.num_folds):
                        all_test_preds = []
                        for x in tqdm.tqdm(range(len(all_test_images))):
                        # inference with test-time augmentation
                            test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                            test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                            imaging_prediction = imaging_ensemble(args, test_image)

                            clinic_prediction = clinic_ensemble(args, all_test_ids[x])

                            input_features =  np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)

                            mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)]) 
                            mm_predictions = mm_model.predict_proba(input_features)                                      
                            all_test_preds.append(mm_predictions[0][1])
                        fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
                        auc_ = roc_auc_score(all_test_labels, all_test_preds)
                        fpr.append(fpr_)
                        tpr.append(tpr_)
                        auc.append(auc_)  
                        pred_per_fold.append(all_test_preds)
            elif args.modal_type == 'uni_imaging':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #-----------------------------------------------------------------------------------------
                with torch.no_grad():
                        # for all test cases in given fold
                    for x in tqdm.tqdm(range(len(all_test_images))):
                        # inference with test-time augmentation
                        test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                        test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                        
                        imaging_prediction = imaging_ensemble(args, test_image)

                        input_features =  imaging_prediction.squeeze().reshape(1, -1)
                        all_test_preds.append(input_features[0][0])                    

            elif args.modal_type == 'uni_clinic':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_clinic[F1, F2, F3, F4, F5])
    #-----------------------------------------------------------------------------------------
                with torch.no_grad():
                        # for all test cases in given fold
                    for x in tqdm.tqdm(range(len(all_test_images))):
                        # inference with test-time augmentation
                        clinic_prediction = clinic_ensemble(args, all_test_ids[x])

                        input_features =  clinic_prediction.squeeze().reshape(1, -1)
                        all_test_preds.append(input_features[0][0])        

            else:
    #-----------------------------------------------------------------------------------------
    # uni_imaging[F1, F2, F3, F4, F5]
    #                                -----> args.ensemble(mm[F1, F2, F3, F4, F5])                                       
    # uni_clinic[F1, F2, F3, F4, F5] 
    #-----------------------------------------------------------------------------------------
                print("Loading {} LogRegr Model with late args.ensemble".format(args.modal_type))
                with torch.no_grad():
                        # for all test cases in given fold
                    for x in tqdm.tqdm(range(len(all_test_images))): 
                        mm_ensemble = []
                        test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                        test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                        
                        for fold in range(args.num_folds):
                            imaging_model = load_efficientnet_b0_imaging_model(
                                args.optimal_imaging_model_paths['F{}'.format(fold)], head=True)
                            imaging_model.eval()

                            clinic_model = load_clinic_model(args.optimal_clinic_model_paths['F{}'.format(fold)])
                            clinic_model.eval()

                            imaging_prediction = np.mean([
                                    imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                    for x in test_image])                                     

                            clinic_prediction = clinic_model(
                                clinic_features_for_pat_id(args, all_test_ids[x])).sigmoid().squeeze().cpu().detach().numpy()  

                            input_features = np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)

                            mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)]) 
                            mm_predictions = mm_model.predict_proba(input_features) 
                            mm_ensemble.append(mm_predictions[0][1])
                        mm_ens = np.sum(mm_ensemble)/5                                       
                        all_test_preds.append(mm_ens)
            fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
            auc_ = roc_auc_score(all_test_labels, all_test_preds)
            fpr.append(fpr_)
            tpr.append(tpr_)
            auc.append(auc_)  
            pred_per_fold.append(all_test_preds)
    else:    
    #-----------------------------------------------------------------------------------------
    # NO ENSEMBLE TESTING
    #-----------------------------------------------------------------------------------------        
        print(f'Ensemble is turned off')
        if args.modal_type != 'nomogram_SEER' and args.modal_type != 'nomogram_dc':
            for fold in range(args.num_folds):
                all_test_preds = []
                if args.modal_type == 'mm':
        #-----------------------------------------------------------------------------------------                                  
        # uni_imaging[F1, F2, F3, F4, F5]
        #                                -----> mm[F1, F2, F3, F4, F5]                                     
        # uni_clinic[F1, F2, F3, F4, F5]  
        #----------------------------------------------------------------------------------------- 
                    print(f"Loading MM Model for Fold {fold}")
                    mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)])
                    clinic_model = load_clinic_model(args.optimal_clinic_model_paths['F{}'.format(fold)])
                    clinic_model.eval()                
                    imaging_model = load_efficientnet_b0_imaging_model(
                        args.optimal_imaging_model_paths['F{}'.format(fold)], head=True) 
                                        
                    imaging_model.eval()  
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_test_images))):
                            # inference with test-time augmentation

                            test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                            test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                            imaging_prediction = np.mean([
                                    imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                    for x in test_image])    
                            
                            clinic_prediction = clinic_model(
                                clinic_features_for_pat_id(args, all_test_ids[x])).sigmoid().cpu().detach().numpy()  

                            input_features = np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)
                            mm_predictions = mm_model.predict_proba(
                                input_features)                                  
                            all_test_preds.append(mm_predictions[0][1])

                elif args.modal_type == 'uni_clinic':
        #-----------------------------------------------------------------------------------------                                  
        # uni_clinic[F1, F2, F3, F4, F5] 
        #-----------------------------------------------------------------------------------------   
                    print(f"Loading Clinical Model for Fold {fold}")
                    clinic_model = load_clinic_model(args.optimal_clinic_model_paths['F{}'.format(fold)])
                    clinic_model.eval()      
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_test_images))):
                            imaging_prediction = np.nan

                            clinic_prediction = clinic_model(
                                clinic_features_for_pat_id(args, all_test_ids[x])).sigmoid().cpu().detach().numpy()

                            input_features = clinic_prediction.squeeze().reshape(1, -1)
                            all_test_preds.append(input_features[0][0])  
                            
                elif args.modal_type =='uni_imaging':
        #-----------------------------------------------------------------------------------------                                  
        # uni_imaging[F1, F2, F3, F4, F5] 
        #-----------------------------------------------------------------------------------------
                    print(f"Loading Imaging Model for Fold {fold}")
                    imaging_model = load_efficientnet_b0_imaging_model(
                        args.optimal_imaging_model_paths['F{}'.format(fold)], head=True)  
                    imaging_model.eval()  
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_test_images))):                
                            clinic_prediction = np.nan

                            test_image = torch.from_numpy(all_test_images[x]).to('cuda')
                            test_image = [test_image, torch.flip(test_image, [4]).to('cuda')]
                            imaging_prediction = np.mean([
                                    imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                    for x in test_image])
                            
                            input_features = imaging_prediction.squeeze().reshape(1, -1)
                            all_test_preds.append(input_features[0][0])   
                fpr_, tpr_, _ = roc_curve(all_test_labels, all_test_preds)
                auc_ = roc_auc_score(all_test_labels, all_test_preds)
                fpr.append(fpr_)
                tpr.append(tpr_)
                auc.append(auc_)  
                pred_per_fold.append(all_test_preds)
    validation_sheet = pd.DataFrame()
    validation_sheet['aucs'] = auc
    validation_sheet['fprs'] = fpr
    validation_sheet['tprs'] = tpr  

    prediction_sheet = pd.DataFrame()
    prediction_sheet['archiveID'] = all_test_ids
    num = 0
    for elem in pred_per_fold:
        prediction_sheet['F{}'.format(num)] = np.transpose(elem)
        num = num + 1
    prediction_sheet['labels'] = all_test_labels
    if args.modal_type == 'mm' and args.double_ensemble:
        validation_sheet.to_excel(args.results_dir + args.modal_type + '_external_test_performances_double_ensemble.xlsx', index=False)
        prediction_sheet.to_excel(args.results_dir + args.modal_type + '_external_test_predictions_double_ensemble.xlsx', index=False)
    elif args.early_ensemble:
        validation_sheet.to_excel(args.results_dir + args.modal_type + '_external_test_performances_early_ensemble.xlsx', index = False)
        prediction_sheet.to_excel(args.results_dir + args.modal_type + '_external_test_predictions_early_ensemble.xlsx', index=False)
    elif args.ensemble:
        validation_sheet.to_excel(args.results_dir + args.modal_type + '_external_test_performances_ensemble.xlsx', index = False)
        prediction_sheet.to_excel(args.results_dir + args.modal_type + '_external_test_predictions_ensemble.xlsx', index=False)
    else:
        validation_sheet.to_excel(args.results_dir + args.modal_type + '_external_test_performances_no_ensemble.xlsx'.format(fold), index=False)
        prediction_sheet.to_excel(args.results_dir + args.modal_type + '_external_test_predictions_no_ensemble.xlsx', index=False)
    print("Complete.")    
