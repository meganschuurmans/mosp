from .data_generator import load_efficientnet_b0_imaging_model, load_clinic_model, clinic_features_for_pat_id
import numpy as np
from joblib import dump, load

def imaging_ensemble(args, image):
    imaging_predictions = []
    
    for fold in range(args.num_folds):
        imaging_model = load_efficientnet_b0_imaging_model(
        args.optimal_imaging_model_paths['F{}'.format(fold)], head=True) 
        if args.ema:
            ema = ExponentialMovingAverage(imaging_model.parameters(), decay=0.95) 
            imaging_model.eval()
            with ema.average_parameters():
                imaging_prediction = np.mean([
                imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                for x in image])     
                imaging_predictions.append(imaging_prediction)
        else:
            imaging_model.eval()
            imaging_prediction = np.mean([
            imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
            for x in image])     
            imaging_predictions.append(imaging_prediction)
            
    ensemble_imaging_prediction = sum(imaging_predictions)/args.num_folds
    # print(f"ensemble prediction:{ensemble_imaging_prediction} {imaging_predictions} ")
    return ensemble_imaging_prediction

def clinic_ensemble(args, pat_id):
    
    clinic_predictions = []
    
    for fold in range(args.num_folds):
        clinic_model = load_clinic_model(args.optimal_clinic_model_paths['F{}'.format(fold)])
        clinic_model.eval()       

        clinic_prediction = clinic_model(clinic_features_for_pat_id(args, pat_id)).sigmoid().squeeze().cpu().detach().numpy()
        clinic_predictions.append(clinic_prediction)

    ensemble_clinic_prediction = sum(clinic_predictions)/args.num_folds

    return ensemble_clinic_prediction

def mm_ensemble(args, image, pat_id):
    mm_predictions = []
    for fold in range(args.num_folds):
        imaging_model = load_efficientnet_b0_imaging_model(
        args.optimal_imaging_model_paths['F{}'.format(fold)], head=True) 
        clinic_model = load_clinic_model(args.optimal_clinic_model_paths['F{}'.format(fold)])
        mm_model = load(args.optimal_mm_model_paths['F{}'.format(fold)]) 

        if args.ema:
            ema = ExponentialMovingAverage(imaging_model.parameters(), decay=0.95) 
            imaging_model.eval()
            with ema.average_parameters():
                imaging_prediction = np.mean([
                imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
                for x in image])     
        else:
            imaging_model.eval()
            clinic_model.eval()   

            imaging_prediction = np.mean([
            imaging_model(x).sigmoid()[0][1].cpu().detach().numpy() 
            for x in image])    
            clinic_prediction = clinic_model(clinic_features_for_pat_id(args, pat_id)).sigmoid().squeeze().cpu().detach().numpy()

        input_features =  np.append(clinic_prediction, imaging_prediction).squeeze().reshape(1, -1)
        mm_prediction = mm_model.predict_proba(input_features) 
        mm_predictions.append(mm_prediction[0][1])
        print(input_features, mm_prediction)

    ensemble_imaging_prediction = mm_predictions
    
    # print(f"ensemble prediction:{ensemble_imaging_prediction} {imaging_predictions} ")
    return ensemble_imaging_prediction
