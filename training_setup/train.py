
import torch
import os
import numpy as np
from .data_generator import load_dataset, load_efficientnet_b0_imaging_model, load_clinic_model, clinic_features_for_pat_id
import tqdm
from .ensemble import imaging_ensemble, clinic_ensemble, mm_ensemble
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

def training(args):
    for fold in range(args.num_folds):
        #-----------------------------------------------------------------------------------------
        print('-'*80)
        # load training dataset for given fold
        print(f"Loading Training Dataset for Fold {fold}")
        all_train_images, all_train_labels, all_train_ids = load_dataset(
            args.overview_dir + 'train-fold-{}.xlsx'.format(fold))
        #-----------------------------------------------------------------------------------------
        train_features = []
    #-----------------------------------------------------------------------------------------
        # set multimodal destination path
        if args.ema and args.ensemble:
            destination_multimodal_path = args.results_dir + args.modal_type + '_ensemble_ema_F{}.joblib'.format(fold)
        elif not args.ema and args.ensemble:
            destination_multimodal_path = args.results_dir + args.modal_type + '_late_ensemble_no_ema_F{}.joblib'.format(fold)
        elif args.ema and not args.ensemble:
            destination_multimodal_path = args.results_dir + args.modal_type + '_no_ensemble_ema_F{}.joblib'.format(fold)  
        else:
            destination_multimodal_path = args.results_dir + args.modal_type + '_no_ensemble_no_ema_F{}.joblib'.format(fold)       
    #-----------------------------------------------------------------------------------------
        if not os.path.exists(destination_multimodal_path):

            if args.ensemble:
    #-----------------------------------------------------------------------------------------
    # ENSEMBLE TRAINING
    #-----------------------------------------------------------------------------------------
                print(f'Ensemble is turned on')
                if args.modal_type == 'mm' and args.early_ensemble == True:
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #                                         -----> mm[F1, F2, F3, F4, F5]                                       
    # args.ensemble(uni_clinic[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------                              
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_train_images))):                
                            # inference with test-time augmentation
                            train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                            train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]

                            imaging_prediction = imaging_ensemble(train_image)
                            clinic_prediction = clinic_ensemble(all_train_ids[x])

                            train_features.append([clinic_prediction, imaging_prediction])

                elif args.modal_type == 'mm' and args.early_ensemble == False:
    #-----------------------------------------------------------------------------------------
    # uni_imaging[F1, F2, F3, F4, F5]
    #                             -----> mm[F1, F2, F3, F4, F5] -----> args.ensemble(mm[F1, F2, F3, F4, F5])                                      
    # uni_clinic[F1, F2, F3, F4, F5] 
    #-----------------------------------------------------------------------------------------                              
                    with torch.no_grad():
                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_train_images))):                
                            # inference with test-time augmentation
                            train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                            train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]
                            mm_prediction = mm_ensemble(args, train_image, all_train_ids[x])
                            train_features.append(mm_prediction.squeeze())


                elif args.modal_type == 'uni_imaging':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5]) 
    #----------------------------------------------------------------------------------------- 
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_train_images))):  
                            # inference with test-time augmentation
                            train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                            train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]

                            imaging_prediction = imaging_ensemble(train_image)

                            train_features.append(imaging_prediction.squeeze().reshape(1, -1))      
            else:
    #-----------------------------------------------------------------------------------------
    # NO ENSEMBLE TRAINING
    #----------------------------------------------------------------------------------------
                print(f'Ensemble is turned off')
                if args.modal_type == 'mm':
    #-----------------------------------------------------------------------------------------                                  
    # uni_imaging[F1, F2, F3, F4, F5]
    #                                -----> mm[F1, F2, F3, F4, F5]                                     
    # uni_clinic[F1, F2, F3, F4, F5]  
    #----------------------------------------------------------------------------------------- 
                    print(f"Loading Imaging and Clinical Model for Fold {fold}")
                    imaging_model = load_efficientnet_b0_imaging_model(
                        args.imaging_model_dir + '_F{}.pt'.format(fold), head=True)
                    imaging_model.eval()

                    clinic_model = load_clinic_model(
                        args.optimal_clinic_model_paths['F{}'.format(fold)])
                    clinic_model.eval()

                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_train_images))):
                            train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                            train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]
                            imaging_prediction = np.mean([
                                    imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                    for x in train_image])     

                            clinic_prediction = clinic_model(
                                clinic_features_for_pat_id(args, all_train_ids[x])).sigmoid().squeeze().cpu().detach().numpy()  

                            train_features.append([clinic_prediction, imaging_prediction])

                elif args.modal_type == 'uni_clinic':
    #----------------------------------------------------------------------------------------- 
    # uni_clinic[F1, F2, F3, F4, F5]  
    #----------------------------------------------------------------------------------------- 
                    print(f"Loading Clinical Model for Fold {fold}")
                    clinic_model = load_clinic_model(
                        args.optimal_clinic_model_paths['F{}'.format(fold)])
                    clinic_model.eval() 
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_train_images))):

                            clinic_prediction = clinic_model(
                                clinic_features_for_pat_id(args, all_train_ids[x])).sigmoid().cpu().detach().numpy()

                            train_features.append(clinic_prediction.squeeze().reshape(1, -1))

                elif args.modal_type == 'uni_imaging':
    #----------------------------------------------------------------------------------------- 
    # uni_imaging[F1, F2, F3, F4, F5]  
    #-----------------------------------------------------------------------------------------
                    print(f"Loading Imaging Model for Fold {fold}")

                    imaging_model = load_efficientnet_b0_imaging_model(
                        args.optimal_imaging_model_paths['F{}'.format(fold)], head=True)      
                    imaging_model.eval()            
                    with torch.no_grad():

                        # for all validation cases in given fold
                        for x in tqdm.tqdm(range(len(all_train_images))):  
                            # inference with test-time augmentation
                            train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                            train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]
                            imaging_prediction = np.mean([
                                    imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                    for x in train_image])    

                            train_features.append(imaging_prediction.squeeze().reshape(1, -1))

    #-----------------------------------------------------------------------------------------        
    # normalize features across training dataset and train multimodal LogRegr model
        print("Fitting LogRegr Model for Multimodal Predictions...")
    # multimodal_model = SVC(class_weight={
        multimodal_model = LogisticRegression(class_weight={
    # multimodal_model = RandomForestClassifier(class_weight={
            0: np.sum(all_train_labels)/len(all_train_labels), 
            1: 1 - np.sum(all_train_labels)/len(all_train_labels)})                                     
        multimodal_model.fit(np.array(train_features), all_train_labels)  

    #-----------------------------------------------------------------------------------------
    # save model weights and train norm stats
        dump(multimodal_model, destination_multimodal_path)                                               
        print("Saved.")   
    print("Complete.")    
    return

def training_all_epochs(args):
    for fold in range(args.num_folds):
        #-----------------------------------------------------------------------------------------
        print('-'*80)
        # load training dataset for given fold
        print(f"Loading Training Dataset for Fold {fold}")
        all_train_images, all_train_labels, all_train_ids = load_dataset(
            args.overview_dir + 'train-fold-{}.xlsx'.format(fold))
        #-----------------------------------------------------------------------------------------
        for epoch in range(args.num_epochs):                
            train_features = []
    #-----------------------------------------------------------------------------------------
            # set multimodal destination path
            if args.ema and args.ensemble:
                destination_multimodal_path = args.results_dir + args.modal_type + '_ensemble_ema_F{}_E{}.joblib'.format(fold, epoch)
            elif not args.ema and args.ensemble:
                destination_multimodal_path = args.results_dir + args.modal_type + '_late_ensemble_no_ema_F{}_E{}.joblib'.format(fold, epoch)
            elif args.ema and not args.ensemble:
                destination_multimodal_path = args.results_dir + args.modal_type + '_no_ensemble_ema_F{}_E{}.joblib'.format(fold, epoch)  
            else:
                destination_multimodal_path = args.results_dir + args.modal_type + '_no_ensemble_no_ema_F{}_E{}.joblib'.format(fold, epoch)       
    #-----------------------------------------------------------------------------------------
            if not os.path.exists(destination_multimodal_path):

                if args.ensemble:
    #-----------------------------------------------------------------------------------------
    # ENSEMBLE TRAINING
    #-----------------------------------------------------------------------------------------
                    print(f'Ensemble is turned on')
                    if args.modal_type == 'mm' and args.early_ensemble:
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5])
    #                                         -----> mm[F1, F2, F3, F4, F5]                                       
    # args.ensemble(uni_clinic[F1, F2, F3, F4, F5]) 
    #-----------------------------------------------------------------------------------------                              
                        with torch.no_grad():

                            # for all validation cases in given fold
                            for x in tqdm.tqdm(range(len(all_train_images))):                
                                # inference with test-time augmentation
                                train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                                train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]

                                imaging_prediction = imaging_ensemble(train_image)
                                clinic_prediction = clinic_ensemble(all_train_ids[x])

                                train_features.append([clinic_prediction, imaging_prediction])


                    elif args.modal_type == 'uni_imaging':
    #-----------------------------------------------------------------------------------------
    # args.ensemble(uni_imaging[F1, F2, F3, F4, F5]) 
    #----------------------------------------------------------------------------------------- 
                        with torch.no_grad():

                            # for all validation cases in given fold
                            for x in tqdm.tqdm(range(len(all_train_images))):  
                                # inference with test-time augmentation
                                train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                                train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]

                                imaging_prediction = imaging_ensemble(train_image)

                                train_features.append(imaging_prediction.squeeze().reshape(1, -1))      
                else:
    #-----------------------------------------------------------------------------------------
    # NO ENSEMBLE TRAINING
    #----------------------------------------------------------------------------------------
                    print(f'Ensemble is turned off')
                    if args.modal_type == 'mm':
    #-----------------------------------------------------------------------------------------                                  
    # uni_imaging[F1, F2, F3, F4, F5]
    #                                -----> mm[F1, F2, F3, F4, F5]                                     
    # uni_clinic[F1, F2, F3, F4, F5]  
    #----------------------------------------------------------------------------------------- 
                        print(f"Loading Imaging and Clinical Model for Fold {fold}")
                        imaging_model = load_efficientnet_b0_imaging_model(
                            args.imaging_model_dir + '_F{}_E{}.pt'.format(fold, epoch), head=True)
                        imaging_model.eval()

                        clinic_model = load_clinic_model(
                            args.optimal_clinic_model_paths['F{}'.format(fold)])
                        clinic_model.eval()

                        with torch.no_grad():

                            # for all validation cases in given fold
                            for x in tqdm.tqdm(range(len(all_train_images))):
                                train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                                train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]
                                imaging_prediction = np.mean([
                                        imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                        for x in train_image])     

                                clinic_prediction = clinic_model(
                                    clinic_features_for_pat_id(args, all_train_ids[x])).sigmoid().squeeze().cpu().detach().numpy()  

                                train_features.append([clinic_prediction, imaging_prediction])

                    elif args.modal_type == 'uni_clinic':
    #----------------------------------------------------------------------------------------- 
    # uni_clinic[F1, F2, F3, F4, F5]  
    #----------------------------------------------------------------------------------------- 
                        print(f"Loading Clinical Model for Fold {fold}")
                        clinic_model = load_clinic_model(
                            args.optimal_clinic_model_paths['F{}'.format(fold)])
                        clinic_model.eval() 
                        with torch.no_grad():

                            # for all validation cases in given fold
                            for x in tqdm.tqdm(range(len(all_train_images))):

                                clinic_prediction = clinic_model(
                                    clinic_features_for_pat_id(args, all_train_ids[x])).sigmoid().cpu().detach().numpy()

                                train_features.append(clinic_prediction.squeeze().reshape(1, -1))

                    elif args.modal_type == 'uni_imaging':
    #----------------------------------------------------------------------------------------- 
    # uni_imaging[F1, F2, F3, F4, F5]  
    #-----------------------------------------------------------------------------------------
                        print(f"Loading Imaging Model for Fold {fold}")

                        imaging_model = load_efficientnet_b0_imaging_model(
                            args.optimal_imaging_model_paths['F{}'.format(fold)], head=True)      
                        imaging_model.eval()            
                        with torch.no_grad():

                            # for all validation cases in given fold
                            for x in tqdm.tqdm(range(len(all_train_images))):  
                                # inference with test-time augmentation
                                train_image = torch.from_numpy(all_train_images[x]).to('cuda')
                                train_image = [train_image, torch.flip(train_image, [4]).to('cuda')]
                                imaging_prediction = np.mean([
                                        imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                                        for x in train_image])    

                                train_features.append(imaging_prediction.squeeze().reshape(1, -1))

                #-----------------------------------------------------------------------------------------        
                # normalize features across training dataset and train multimodal LogRegr model
                print("Fitting LogRegr Model for Multimodal Predictions...")
                # multimodal_model = SVC(class_weight={
                print(train_features)
                multimodal_model = LogisticRegression(class_weight={
                # multimodal_model = RandomForestClassifier(class_weight={
                    0: np.sum(all_train_labels)/len(all_train_labels), 
                    1: 1 - np.sum(all_train_labels)/len(all_train_labels)})                                     
                multimodal_model.fit(np.array(train_features), all_train_labels)  

                #-----------------------------------------------------------------------------------------
                # save model weights and train norm stats
                dump(multimodal_model, destination_multimodal_path)                                               
                print("Saved.")   
        print("Complete.")    
    return