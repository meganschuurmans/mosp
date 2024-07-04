import numpy as np
import pandas as pd



#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7164422/
def nomogram_score(args, xtrain, resection_status):
     #clinic_age_diagnosis	lab_base_CA19_9	lab_base_CEA	clinic_BMI	clinic_weightloss	clinic_ECOG	
    features = pd.DataFrame(xtrain, columns = ['archiveID', 'clinic_age_diagnosis', 'lab_base_CA19_9', 'lab_base_CEA',
                                            'clinic_BMI', 'clinic_weightloss', 'clinic_ECOG', 'clinic_TNM_tstage', 'clinic_TNM_nstage', 'clinic_TNM_mstage',
                                              'base_lesionsize', 'base_lesionlocation'])
    archive_overview = pd.read_excel(args.archive_overview_dir)
    gender = archive_overview.loc[archive_overview['archiveID'] == features['archiveID'].iloc[0], 'gender'].iloc[0]

    score = 0
    if not resection_status:
        score = score + 55

    T_stage = features['clinic_TNM_tstage'].values[0]
    N_stage = features['clinic_TNM_nstage'].values[0]
    M_stage = features['clinic_TNM_mstage'].values[0]
    
    #PDAC for all patients, race and grade unknown, average taken for both race and grade
    score = score + 79
    if gender == 'M':
        score = score + 7

    if T_stage == 3 and N_stage == 0 and M_stage != 1:
        score = score + 70
    elif T_stage == 1 and N_stage == 0 and M_stage != 1:
        score = score + 70
    elif T_stage == 1 and N_stage == 1 and M_stage != 1:
        score = score + 70
    elif T_stage == 2 and N_stage == 1 and M_stage != 1:
        score = score + 70
    elif T_stage == 3 and N_stage == 1 and M_stage != 1:
        score = score + 70
    elif T_stage == 1 and N_stage == 1 and M_stage != 1:
        score = score + 72
    elif T_stage == 2 and N_stage == 2 and M_stage != 1:
        score = score + 72
    elif T_stage == 3 and N_stage == 2 and M_stage != 1:
        score = score + 72
    elif T_stage == 3:
        score = score + 72
    elif M_stage == 1:
        score = score + 100
        
    if score < 63:
        probability = 0.9
    elif score < 110 and score > 62:
        probability = 0.9 - (0.1*((score - 63)/46))
    elif score < 142 and score > 109:
        probability = 0.8 - (0.1*((score - 110)/31))            
    elif score < 165 and score > 141:
        probability = 0.7 - (0.1*((score - 142)/22))       
    elif score < 185 and score > 164:
        probability = 0.6 - (0.1*((score - 165)/19))       
    elif score < 205 and score > 184:
        probability = 0.5 - (0.1*((score - 185)/19))  
    elif score < 220 and score > 201:
        probability = 0.4 - (0.1*((score - 205)/17))  
    elif score < 240 and score > 219:
        probability = 0.3 - (0.1*((score - 220)/19))  
    elif score < 262 and score > 239:
        probability = 0.2 - (0.1*((score - 240)/21))  
    else:
        probability = 0.1 
    return probability

#Precision Oncology in Surgery: Patient Selection for Operable Pancreatic Cancer
def nomogram_score_david_chang(args, xtrain):
    features = pd.DataFrame(xtrain, columns = ['archiveID', 'clinic_age_diagnosis', 'lab_base_CA19_9', 'lab_base_CEA',
                                            'clinic_BMI', 'clinic_weightloss', 'clinic_ECOG', 'clinic_TNM_tstage', 'clinic_TNM_nstage', 'clinic_TNM_mstage',
                                              'base_lesionsize', 'base_lesionlocation'])
    score = 0
    if features['base_lesionsize'].iloc[0] < 121:
        score = score + features['base_lesionsize'].iloc[0]/1.2
    else:
        score = score + 100
    if features['clinic_age_diagnosis'].iloc[0] < 91:
        score = score + (features['clinic_age_diagnosis'].iloc[0]-25)/65 * 66
    else:
        score = score + 66
        
    if features['base_lesionlocation'].iloc[0] == 0:
        score = score + 20
        
    #average for S100A2
    score = score + 3
    #average for S100A4
    score = score + 17
    
    # one year survival probability
    if score < 35:
        probability = 0.9
    elif score < 75 and score > 35:
        probability = 0.9 - (0.1*((score - 36)/(38)))
    elif score < 105 and score > 74:
        probability = 0.8 - (0.1*((score - 75)/29))            
    elif score < 125 and score > 104:
        probability = 0.7 - (0.1*((score - 105)/19))       
    elif score < 140 and score > 124:
        probability = 0.6 - (0.1*((score - 125)/14))       
    elif score < 155 and score > 139:
        probability = 0.5 - (0.1*((score - 140)/14))  
    elif score < 170 and score > 154:
        probability = 0.4 - (0.1*((score - 155)/14))  
    elif score < 188 and score > 169:
        probability = 0.3 - (0.1*((score - 170)/17))  
    elif score < 208 and score > 187:
        probability = 0.2 - (0.1*((score - 188)/19))  
    elif score < 225 and score > 207:
        probability = 0.2 - (0.1*((score - 208)/16))  
    else:
        probability = 0.1

    return probability

#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7164422/
def nomogram_score_external(args, xtrain):
    features = pd.DataFrame(xtrain, columns = ['archiveID', 'race', 'gender', 'clinic_age_diagnosis', 'ajcc_stage', 'base_lesionlocation', 'base_lesionsize', 'grade'])

    #all patients are PDAC + resected, therefore start with 60 
    score = 60

    #if grade unknown, it was set to unknown
    if features['gender'].iloc[0] == 'M':
        score = score + 7
    
    if features['grade'].iloc[0] == 'high':
        score = score + 0
    elif features['grade'].iloc[0] == 'unknown':
        score = score + 10
    elif features['grade'].iloc[0] == 'low':
        score = score + 30

    #if race unknown, it was set to others
    if features['race'].iloc[0] == 'black':
        score = score + 15
    elif features['race'].iloc[0] == 'white':
        score = score + 10

    if features['ajcc_stage'].iloc[0] == 2:
        score = score + 70
    elif features['ajcc_stage'].iloc[0] == 3:
        score = score + 72
    elif features['ajcc_stage'].iloc[0] == 4:
        score = score + 100

    if score < 63:
        probability = 0.9
    elif score < 110 and score > 62:
        probability = 0.9 - (0.1*((score - 63)/46))
    elif score < 142 and score > 109:
        probability = 0.8 - (0.1*((score - 110)/31))            
    elif score < 165 and score > 141:
        probability = 0.7 - (0.1*((score - 142)/22))       
    elif score < 185 and score > 164:
        probability = 0.6 - (0.1*((score - 165)/19))       
    elif score < 205 and score > 184:
        probability = 0.5 - (0.1*((score - 185)/19))  
    elif score < 220 and score > 201:
        probability = 0.4 - (0.1*((score - 205)/17))  
    elif score < 240 and score > 219:
        probability = 0.3 - (0.1*((score - 220)/19))  
    elif score < 262 and score > 239:
        probability = 0.2 - (0.1*((score - 240)/21))  
    else:
        probability = 0.1 
    return probability

#Precision Oncology in Surgery: Patient Selection for Operable Pancreatic Cancer
def nomogram_score_david_chang_external(args, xtrain):
    
     #clinic_age_diagnosis	lab_base_CA19_9	lab_base_CEA	clinic_BMI	clinic_weightloss	clinic_ECOG	
    features = pd.DataFrame(xtrain, columns = ['archiveID', 'race', 'gender', 'clinic_age_diagnosis', 'ajcc_stage', 'base_lesionlocation', 'base_lesionsize', 'grade'])

    score = 0
    if features['base_lesionsize'].iloc[0] < 121:
        score = score + features['base_lesionsize'].iloc[0]/1.2
    else:
        score = score + 100
    if features['clinic_age_diagnosis'].iloc[0] < 91:
        score = score + (features['clinic_age_diagnosis'].iloc[0]-25)/65 * 66
    else:
        score = score + 66
        
    if features['base_lesionlocation'].iloc[0] == 0:
        score = score + 20
        
    #average for S100A2
    score = score + 3
    #average for S100A4
    score = score + 17

    # one year survival probability    
    if score < 35:
        probability = 0.9
    elif score < 75 and score > 35:
        probability = 0.9 - (0.1*((score - 36)/(38)))
    elif score < 105 and score > 74:
        probability = 0.8 - (0.1*((score - 75)/29))            
    elif score < 125 and score > 104:
        probability = 0.7 - (0.1*((score - 105)/19))       
    elif score < 140 and score > 124:
        probability = 0.6 - (0.1*((score - 125)/14))       
    elif score < 155 and score > 139:
        probability = 0.5 - (0.1*((score - 140)/14))  
    elif score < 170 and score > 154:
        probability = 0.4 - (0.1*((score - 155)/14))  
    elif score < 188 and score > 169:
        probability = 0.3 - (0.1*((score - 170)/17))  
    elif score < 208 and score > 187:
        probability = 0.2 - (0.1*((score - 188)/19))  
    elif score < 225 and score > 207:
        probability = 0.2 - (0.1*((score - 208)/16))  
    else:
        probability = 0.1
    
    #270 day probability
    return probability


