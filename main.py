#  Copyright 2024 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
from training_setup.train import training, training_all_epochs
from training_setup.validation import validation, validation_all_epochs
from training_setup.testing import internal_testing, external_testing
from training_setup.data_generator import model_paths
from training_setup.create_gradcam import create_gradcam
#-----------------------------------------------------------------------------------------
def main():
    # command line arguments for hyperparameters and I/O paths
    parser = argparse.ArgumentParser(description='Command Line Arguments for Training Script')

    # data I/0 + experimental setup
    parser.add_argument('--archive_overview_dir', type=str, required=True,            
                        help="Path to archive overview")
    parser.add_argument('--overview_dir', type=str, required=True,            
                        help="Base path to training, validation and test data sheets")
    parser.add_argument('--external_dir', type=str, required=True,            
                        help="Base path to external data sheets")
    parser.add_argument('--external_center', type=str, required=False,            
                        help="Center of the external data: umcg, cnio, tcia")    
    parser.add_argument('--multimodal_model_dir', type=str, required=True,
                        help="Base path to multimodal models")
    parser.add_argument('--imaging_model_dir', type=str, required=True,
                        help="Base path to imaging models")
    parser.add_argument('--clinical_model_dir', type=str, required=True,
                        help="Base path to clinical models")
    parser.add_argument('--clinical_variables_dir', type=str, required=True,
                        help="Base path to clinical variable excel")
    parser.add_argument('--results_dir', type=str, required=True,           
                        help="Destination path for results")
    parser.add_argument('--model_phase', type=str, required=True,            
                        help="Model development, internal testing or external testing: training, validation, internal_testing, external_testing, grad_cam")
    
    # training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=200,              
                        help="Number of training epochs")
    parser.add_argument('--num_folds', type=int, default=5,              
                        help="Number of folds")
    parser.add_argument('--ensemble', type=bool, default=True,              
                        help="Ensemble the imaging- or clinical models")
    parser.add_argument('--double_ensemble', type=bool, default=False,                
                        help="Ensemble the imaging-, clinical- and multimodals models")
    parser.add_argument('--early_ensemble', type=bool, default=False,            
                        help="Ensemble the imaging- and clinical models but not the multimodal model")


    # neural network-specific hyperparameters
    parser.add_argument('--modal_type', type=str, default='mm',                                                    
                        help="Networks: uni_clinic, uni_imaging, mm")

    # run model of choice 
    args = parser.parse_args()    


    # --------------------------------------------------------------------------------------------------------------------------
    # derive dataLoaders
    ##  REQUIRES REPLACEMENTS TO PATHS OF THE MODEL OPTIMAL PATHS IN REPO
    optimal_clinic_model_paths, optimal_imaging_model_paths, optimal_mm_model_paths = model_paths(args=args)
    args.optimal_clinic_model_paths = optimal_clinic_model_paths
    args.optimal_imaging_model_paths = optimal_imaging_model_paths
    args.optimal_mm_model_paths = optimal_mm_model_paths
    # --------------------------------------------------------------------------------------------------------------------------
    # run model of choice 
    if args.ensemble == True:
        print(f'Start {args.model_phase} for ensembled {args.modal_type} model!')
    else:
        print(f'Start {args.model_phase} for {args.modal_type} model without ensemble!')

    if args.model_phase == 'training':
        if 'clinic' in args.modal_type:
            training(args=args)
        else:
            training_all_epochs(args=args)

    elif args.model_phase == 'validation':
        if 'clinic' in args.modal_type or args.ensemble == True:
            validation(args=args)
        else:
            validation_all_epochs(args=args)
    elif args.model_phase == 'internal_testing':
        internal_testing(args=args)
    elif args.model_phase == 'external_testing':
        external_testing(args=args)
    elif args.model_phase == 'gradcam':
        create_gradcam(args=args)

if __name__ == '__main__':
    main()