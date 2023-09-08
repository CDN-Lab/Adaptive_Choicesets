import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

###Functions###############

def get_extreme(p_options, 
                domain='gain'): 
    ###Obtains extreme gain/loss for each probability
    """
    Inputs: 
        - p_options = list of probability values (float)
    Output:
        - dict containing highest possible gain or loss for each probability ($50 or -$50)
    """
    extr = {}
    V = Vmax #default gain domain (max gain)
    if domain == 'loss': #loss domain 
        V = Vmin #assign min value (max loss)
    for p in p_options: #iterate through probability levels
        extr[p] = V #assign max value (gain) or min value (loss) for each probability
    return extr

def get_extr_df(extr_dict, 
                domain='gain'):
    ###Create DataFrame of extreme values for gain/loss trial probabilities
    """
    Inputs:
        - extr_dict = dict containing highest possible gain or loss for each probability ($50 or -$50) 
    Output:
        - Dataframe containing column of extreme values for all trials
    """
    df_app = pd.DataFrame(extr_dict.items(), #all columns names and values
                          columns=['p_reward', 'value_reward']) #add extra columns
    Extr_df = pd.concat([empty_df, df_app], ignore_index=True)
    category = 'Extr_pos'
    if domain == 'loss':
        category = 'Extr_neg'
    Extr_df['category'] = category
    Extr_df['ambiguity'] = A_null
    return Extr_df

def add_ambig_extr(df, 
                   amb_options, 
                   domain='gain'):
    ###Add ambiguity trials to df
    """
    Inputs:
        - df
        - amb_options = list of possible ambiguity values
    Output:
        - Dataframe with appended ambiguity max values
    """
    amb_dict = {amb_prob: amb_options}
    df_app = pd.DataFrame(amb_dict.items(), 
                          columns=['p_reward', 'ambiguity'])
    df_app = df_app.explode('ambiguity') #flatten ambiguity column
    df_app['category'] = df['category'] 
    if domain == 'gain': #gain domain
        df_app['value_reward'] = Vmax #Max gain = $50
    else: #loss domain
        df_app['value_reward'] = Vmin #Max loss = -$50
    df = pd.concat([df, df_app], ignore_index=True)
    return df

def append_SVreward(df, 
                    alpha, 
                    beta, 
                    domain='gain'):
    ###Calculate SV of lotteries given Ss alpha and beta
    """
    Inputs:
        - df
        - alpha
        - beta
    Output:
        - Dataframe contain new SV reward values
    """
    Amp = 1.0
    if domain == 'loss':
        Amp = -1.0
    df['SV_reward'] = (df['p_reward'] - beta * df['ambiguity']/2) * Amp * (abs(df['value_reward']))**alpha
    return df

def append_Vsafe(df, 
                 alpha, 
                 beta, 
                 domain='gain'): 
    ###Calculate SV of safe options given Ss alpha and beta
    """
    Inputs:
        - df
        - alpha
        - beta
    Output:
        - Dataframe
    """
    Amp = 1.0
    if domain == 'loss':
        Amp = -1.0
    df['SV_New_Safe'] = 0.5 * df['SV_reward']
    df['value_lott_SE'] = Amp * (abs(df['SV_New_Safe']) / (df['p_reward'] - beta * df['ambiguity']/2))**(1/alpha)
    df['value_safe'] = Amp * (abs(df['SV_New_Safe']))**(1/alpha)
    return df

def calculate_value_reward(df, 
                           alpha, 
                           beta,
                           domain='gain'):
    ###Calculate value of lotteries for new SV with same SVdelta
    """
    Inputs:
        - df
        - alpha
        - beta
    Output:
        - Dataframe
    """
    new_rows = []
    for index, row in df.iterrows():
        SV = row['SV_reward'] 
        value_safe = row['value_safe'] 
        p = row['p_reward']
        A = row['ambiguity'] 
        if A != 0:
            p = 0.50
        if domain == 'gain':
            if SV < 0.0: 
                continue    
            value_reward = (SV / (p - beta * A / 2))**(1/alpha)
            if (value_reward < 100) and (value_reward >= 0.0):
                new_rows.append({'category': row['category'], 'SV_reward': SV, 'p_reward': p, 'ambiguity': A, 'value_safe': value_safe, 'value_reward': value_reward}) 
                #calculated values are used to create new dictionary for each combination, and collected in new rows list 
                new_df = pd.DataFrame(new_rows, columns=['category', 'p_reward','ambiguity', 'value_reward', 'SV_reward', 'value_safe']) #new DF from list
                new_df = new_df.round(2)
        else: #loss domain
            if SV > 0.0: 
                continue    
            value_reward = -((abs(SV) / (p - beta * A / 2))**(1/alpha))
            if (value_reward >- 100) and (value_reward <= 0.0):
                new_rows.append({'category': row['category'], 'SV_reward': SV, 'p_reward': p, 'ambiguity': A, 'value_safe': value_safe, 'value_reward': value_reward}) 
                new_df = pd.DataFrame(new_rows, columns=['category', 'p_reward', 'ambiguity', 'value_reward', 'SV_reward', 'value_safe'])
                new_df = new_df.round(2) 
    return new_df

def determine_nonzero_side(row):
    ###Determine nonzero side of lottery
    """
    Inputs:
        - row = current row values for each column
    Output:
        - String containing "top" or "bottom"
    """
    if row['crdm_lott_top'] != 0.00: #current trial lottery top is not zero
        return 'top' #red is non-zero side
    else:
        return 'bottom' #blue is non-zero side

def determine_risk_image_file(row):
    ###Selects lottery image
    """
    Inputs:
        - row = current row values for each column
    Output:
        - f[0] or f[1] = risk trial image filename
    """
    for p, f in risk_images.items():
        if row['crdm_lott_p'] == p and row['crdm_nonzero_side'] == 'top':
            return  f[1] #2nd idx for red
        elif row['crdm_lott_p'] == p and row['crdm_nonzero_side'] == 'bottom':
            return f[0] #1st idx dict for blue

########################


subid = expInfo['participant'] #subject id
p_options = [0.13, 0.25, 0.38, 0.5, 0.75] #possible probability values (lott)
amb_options = [0.24, 0.5, 0.74] #possible portions of probability obscured (lott)
amb_prob = 0.5 #probability for all ambiguity trials
A_null = 0.0 #zero ambiguity
Vmax = 50.0 #maximum possible payment
Vmin = -50.0 #minimum possible payment
Vsafe_pos = 5.0 #old safe option in positive trials
Vsafe_neg = -5.0 #old safe option in negative trials
desired_trials = 50 #number of trials needed for each domain
empty_df = pd.DataFrame([], columns=['category', 'p_reward', 'value_reward']) #template df

amb_images = {} #ambiguity images filesnames
risk_images = {} #risk images filesnames
for img_type in ["amb_images", "risk_images"]: #flexibly obtain image filenames for ambiguity and risk trials
    if img_type == "amb_images": #ambiguity trial images
        for option in amb_options:
            amb_images[int(option*100)] = ["ambig_{0}.bmp".format(str(int(option*100)))]
    else: #risk trial images
        for option in p_options:
            risk_images[int(option*100)] = ["risk_blue_{0}.bmp".format(str(int(option*100))), 
                                            "risk_red_{0}.bmp".format(str(int(option*100)))]    

###GAIN DOMAIN###
extrpos = get_extreme(p_options) #dict of extreme gain values for each probability ($50)
post_mean_gain = rg.get_post_mean(np.exp(g_log_post), g_sets.param) #posterior mean for gain
alpha_pos, beta_pos = post_mean_gain[0], post_mean_gain[1] #parameters for gains
df_pos = get_extr_df(extrpos)
df_pos = add_ambig_extr(df_pos, amb_options)
df_pos = append_SVreward(df_pos, alpha_pos, beta_pos) #append SV of lottery trials to df
df_pos = append_Vsafe(df_pos, alpha_pos, beta_pos) #append SV of safe trials to df
#safe option dataframe for merging multiple dataframes
df_safe_pos = df_pos[['p_reward', 'ambiguity', 'SV_New_Safe', 'value_lott_SE', 'value_safe']].copy()
#define trials of subjective equality (SE)
df_SE_pos = df_pos[['category', 'p_reward', 'ambiguity', 'value_lott_SE', 'SV_New_Safe']].copy()
df_SE_pos['category'] = 'SE_pos'
df_SE_pos = df_SE_pos.rename(columns={'value_lott_SE': 'value_reward'})
df_SE_pos['value_safe'] = df_safe_pos['value_safe']
df_SE_pos = df_SE_pos.rename(columns={'SV_New_Safe': 'SV_reward'})
#center around trial of subjective equality
df_cent_pos = df_SE_pos.copy()
df_cent_pos['category'] = 'Cent_SE_pos'
df_adjusted_pos = df_cent_pos.copy()
df_adjusted_pos['value_reward'] = df_adjusted_pos['value_reward'] + 2.0
df_adjusted_neg = df_cent_pos.copy()
df_adjusted_neg['value_reward'] = df_adjusted_neg['value_reward'] - 2.0
df_cent_pos = pd.concat([df_cent_pos, df_adjusted_pos, df_adjusted_neg], ignore_index=True)
df_cent_pos = append_SVreward(df_cent_pos, alpha_pos, beta_pos) #centered SE DF
#GAIN ONLY-- filter rows with negative SV_reward values to avoid weird numbers in lottery
df_cent_pos = df_cent_pos[df_cent_pos['SV_reward'] >= 0] #conditional statement may drop trials
df_cent_pos = df_cent_pos.round(2)
df_pos = df_pos.sort_values('SV_reward', ascending=False).reset_index(drop=True) #SVdiff sampling
index = 3 #fourth ranked SV_reward
col = 4 #SV_reward
SV_max_pos = df_pos.iloc[index, col]
_SV_safe = df_pos.iloc[index, col + 1]
_delta_SV = SV_max_pos - _SV_safe #sampling here
_delta_SV2 = 0.5 * _delta_SV # second sampling
#GAIN -- Maximum SV Delta
pos_df1 = df_pos[['category', 'p_reward', 'ambiguity', 
                  'value_safe', 'SV_New_Safe']].copy()
pos_df1['category'] = 'MaxSVdelta'
pos_df1['deltaSV'] = _delta_SV
pos_df1['SV_reward'] = pos_df1['deltaSV'] + pos_df1['SV_New_Safe']
#GAIN -- Half-Maximum SV Delta
pos_df2 = df_pos[['category', 'p_reward', 'ambiguity', 
                  'value_safe', 'SV_New_Safe']].copy()
pos_df2['category'] = 'half_maxSV_delta'
pos_df2['deltaSV'] = _delta_SV2
pos_df2['SV_reward'] = pos_df2['deltaSV'] + pos_df2['SV_New_Safe']
#GAIN -- Minimum SV Delta
pos_df3 = df_pos[['category', 'p_reward', 'ambiguity', 
                  'value_safe', 'SV_New_Safe']].copy()
pos_df3['category'] = 'MinSVdelta'
pos_df3['deltaSV'] = -1.0 * _delta_SV
pos_df3['SV_reward'] = pos_df3['deltaSV'] + pos_df3['SV_New_Safe']
#GAIN -- Half-Minimum SV Delta
pos_df4 = df_pos[['category', 'p_reward', 'ambiguity', 
                  'value_safe', 'SV_New_Safe']].copy()
pos_df4['category'] = 'Half_minSV_delta'
pos_df4['deltaSV'] = -1.0 * _delta_SV2
pos_df4['SV_reward'] = pos_df4['deltaSV'] + pos_df4['SV_New_Safe']
#combine 4 GAIN SV Delta dfs
pos_df = pd.concat([pos_df1, pos_df2, pos_df3, pos_df4], ignore_index=True) 

df_SVdeltas_gains = calculate_value_reward(pos_df, alpha_pos, beta_pos)
df_Trials_gains = pd.concat([df_SVdeltas_gains, df_cent_pos], ignore_index=True)
df_Trials_gains['SV_New_Safe'] = df_Trials_gains['value_safe']**alpha_pos
df_Trials_gains['deltaSV'] = df_Trials_gains['SV_reward'] - df_Trials_gains['SV_New_Safe']
columns_to_convert = ['value_reward', 'value_safe', 'SV_reward', 'deltaSV'] #convert selected columns to numeric values for rounding
df_Trials_gains[columns_to_convert] = df_Trials_gains[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df_Trials_gains = df_Trials_gains.round(2)
if len(df_Trials_gains) < desired_trials: #check if current number of trials is less than desired number
    trials_needed = desired_trials - len(df_Trials_gains)
    #calculate number of additional trials for each category (gains and losses)
    trials_needed_gains = math.ceil(trials_needed)  #round up to nearest integer
    additional_trials_gains = df_pos.sample(n=trials_needed_gains, replace=True) #new trials from extremes reward or loss
    df_Trials_gains = pd.concat([df_Trials_gains, additional_trials_gains], ignore_index=True) #add trials to existing DataFrame
    df_Trials_gains = df_Trials_gains.drop(columns=['value_lott_SE'])
if len(df_Trials_gains) > desired_trials: #check if current number of trials is greater than 50
    trials_to_delete = len(df_Trials_gains) - desired_trials
    #randomly sample rows to delete
    rows_to_delete = df_Trials_gains[df_Trials_gains['category'] == 'Cent_SE_pos'].sample(n=trials_to_delete)
    df_Trials_gains = df_Trials_gains.drop(rows_to_delete.index) #remove sampled rows
df_Trials_gains['value_reward'] = df_Trials_gains['value_reward'].apply(lambda x: round(x * 2) / 2) #round to nearest 50 cents
df_Trials_gains['value_safe'] = df_Trials_gains['value_safe'].apply(lambda x: round(x * 2) / 2) #round to nearest 50 cents
#update value_safe and value_reward when it's 0. Note that the SV_reward for those trials will stay as previously calculate as 0.0
df_Trials_gains.loc[df_Trials_gains['value_safe'] == 0, 'value_safe'] = 0.5
df_Trials_gains.loc[df_Trials_gains['value_reward'] == 0, 'value_reward'] = 0.5
#format to match previous PsychoPy input csv and add lott_top and lott_bot values
crdm_trials_gains = df_Trials_gains.copy()
crdm_trials_gains['category'] = 'gain'
crdm_trials_gains = crdm_trials_gains.sort_values('p_reward', ascending=True).reset_index(drop=True)
crdm_trials_gains = crdm_trials_gains.drop(columns=['SV_reward', 'SV_New_Safe', 'deltaSV'])
crdm_trials_gains['ambiguity'] = (crdm_trials_gains['ambiguity'] * 100).astype(int)
crdm_trials_gains['p_reward'] = (crdm_trials_gains['p_reward'] * 100).astype(int)
crdm_trials_gains = crdm_trials_gains.rename(columns={'value_safe': 'crdm_sure_amt',
                                                      'value_reward': 'crdm_lott',
                                                      'ambiguity': 'crdm_amb_lev',
                                                      'p_reward': 'crdm_lott_p', 
                                                      'category': 'crdm_domain'})
crdm_trials_gains['crdm_sure_p'] = 100
column_order = ['crdm_sure_amt', 'crdm_sure_p', 'crdm_lott', 'crdm_lott_p', 'crdm_amb_lev', 'crdm_domain'] #reordering columns
crdm_trials_gains = crdm_trials_gains[column_order]
crdm_trials_gains = crdm_trials_gains.sort_values(by=['crdm_lott_p', 'crdm_amb_lev'], ascending=[True, True])
zero_ambiguity_rows = crdm_trials_gains[crdm_trials_gains['crdm_amb_lev'] == 0] #find rows with zero 'crdm_amb_lev'
non_zero_ambiguity_rows = crdm_trials_gains[crdm_trials_gains['crdm_amb_lev'] != 0] #find rows with nonzero 'crdm_amb_lev'
crdm_trials_gains = pd.concat([zero_ambiguity_rows, non_zero_ambiguity_rows], ignore_index=True) #join zero ambiguity trials first
random_assignments = np.random.choice(['crdm_lott_top', 'crdm_lott_bot'], size=len(crdm_trials_gains))
crdm_trials_gains['crdm_lott_top'] = np.where(random_assignments == 'crdm_lott_top', crdm_trials_gains['crdm_lott'], 0)
crdm_trials_gains['crdm_lott_bot'] = np.where(random_assignments == 'crdm_lott_bot', crdm_trials_gains['crdm_lott'], 0)
column_order = ['crdm_sure_amt', 'crdm_sure_p', 'crdm_lott_top', 'crdm_lott_bot', 'crdm_lott_p', 'crdm_amb_lev', 'crdm_domain']
crdm_trials_gain = crdm_trials_gains[column_order]


###LOSS DOMAIN###
extrneg = get_extreme(p_options, domain='loss') #dict of extreme loss values for each probability
post_mean_loss = rl.get_post_mean(np.exp(l_log_post), l_sets.param) #posterior mean for loss
alpha_neg, beta_neg = post_mean_loss[0], post_mean_loss[1] #parameters for losses
df_neg = get_extr_df(extrneg, domain='loss')
df_neg = add_ambig_extr(df_neg, amb_options, domain='loss')
df_neg = append_SVreward(df_neg, alpha_neg, beta_neg, domain='loss')
df_neg = append_Vsafe(df_neg, alpha_neg, beta_neg, domain='loss')
#safe option dataframe for merging multiple dataframes
df_safe_neg = df_neg[['p_reward', 'ambiguity', 'SV_New_Safe', 'value_lott_SE', 'value_safe']].copy()
#define trials of subjective equality (SE)
df_SE_neg = df_neg[['category', 'p_reward', 'ambiguity', 'value_lott_SE', 'SV_New_Safe']].copy()
df_SE_neg['category'] = 'SE_neg'
df_SE_neg = df_SE_neg.rename(columns={'value_lott_SE': 'value_reward'})
df_SE_neg['value_safe'] = df_safe_neg['value_safe']
df_SE_neg = df_SE_neg.rename(columns={'SV_New_Safe': 'SV_reward'})
##center around trial of subjective equality
df_cent_neg = df_SE_neg.copy()
df_cent_neg['category'] = 'Cent_SE_neg'
df1_adjusted_pos = df_cent_neg.copy()
df1_adjusted_pos['value_reward'] = df1_adjusted_pos['value_reward'] + 2.0
df1_adjusted_neg = df_cent_neg.copy()
df1_adjusted_neg['value_reward'] = df1_adjusted_neg['value_reward'] - 2.0
df_cent_neg = pd.concat([df_cent_neg, df1_adjusted_pos, df1_adjusted_neg], ignore_index=True)
df_cent_neg = append_SVreward(df_cent_neg, alpha_neg, beta_neg, domain='loss')
#SVdiff sampling
df_neg = df_neg.sort_values('SV_reward', ascending=True).reset_index(drop=True)
neg_index = 4 #fourth ranked SV_reward
neg_col = 4 #SV_reward
SV_min_neg = df_neg.iloc[neg_index, neg_col]
neg_SV_safe = df_neg.iloc[neg_index, neg_col + 1]
neg_delta_SV = SV_min_neg - neg_SV_safe 
neg_delta_SV2 = 0.5 * neg_delta_SV
#LOSS -- Maximum SV Delta
neg_df1 = df_neg[['category', 'p_reward', 'ambiguity', 
                  'value_safe', 'SV_New_Safe']].copy()
neg_df1['category'] = 'MaxSVdelta_Loss'
neg_df1['deltaSV'] = neg_delta_SV
neg_df1['SV_reward'] = neg_df1['deltaSV'] + neg_df1['SV_New_Safe']
#LOSS -- Half-Maximum SV Delta
neg_df2 = df_neg[['category', 'p_reward', 'ambiguity', 
                  'value_safe', 'SV_New_Safe']].copy()
neg_df2['category'] = 'Neg_Half_maxSV_delta'
neg_df2['deltaSV'] = neg_delta_SV2
neg_df2['SV_reward'] = neg_df2['deltaSV'] + neg_df2['SV_New_Safe']
#LOSS -- Minimum SV Delta
neg_df3 = df_neg[['category', 'p_reward', 'ambiguity', 
                  'value_safe', 'SV_New_Safe']].copy()
neg_df3['category'] = 'MinSVdelta'
neg_df3['deltaSV'] = -1.0 * neg_delta_SV
neg_df3['SV_reward'] = neg_df3['deltaSV'] + neg_df3['SV_New_Safe']
#LOSS -- Half-Minimum SV Delta
neg_df4 = df_neg[['category', 'p_reward', 'ambiguity', 
                  'value_safe', 'SV_New_Safe']].copy()
neg_df4['category'] = 'Neg_Half_minSV_delta'
neg_df4['deltaSV'] = -1.0 * neg_delta_SV2
neg_df4['SV_reward'] = neg_df4['deltaSV'] + neg_df4['SV_New_Safe']
#merge all delta LOSS dfs
neg_df = pd.concat([neg_df1, neg_df2, neg_df3, neg_df4], ignore_index=True) 

df_SVdeltas_losses =  calculate_value_reward(neg_df, alpha_neg, beta_neg, domain='loss')
df_Trials_losses = pd.concat([df_SVdeltas_losses, df_cent_neg], ignore_index=True)
df_Trials_losses['SV_New_Safe'] = -abs(df_Trials_losses['value_safe'])**alpha_neg
df_Trials_losses['deltaSV'] = df_Trials_losses['SV_reward'] - df_Trials_losses['SV_New_Safe']
columns_to_convert = ['value_reward', 'value_safe', 'SV_reward']
df_Trials_losses[columns_to_convert] = df_Trials_losses[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df_Trials_losses = df_Trials_losses.round(2)
if len(df_Trials_losses) < desired_trials:
    trials_needed = desired_trials - len(df_Trials_losses)
    trials_needed_losses = math.floor(trials_needed) 
    additional_trials_losses = df_neg.sample(n=trials_needed_gains, replace=True)
    df_Trials_losses = pd.concat([df_Trials_losses, additional_trials_losses], ignore_index=True)
    df_Trials_losses = df_Trials_losses.drop(columns=['value_lott_SE'])
if len(df_Trials_losses) > desired_trials:
    trials_to_delete = len(df_Trials_losses) - desired_trials
    rows_to_delete = df_Trials_losses[df_Trials_losses['category'] == 'Cent_SE_neg'].sample(n=trials_to_delete) 
    df_Trials_losses = df_Trials_losses.drop(rows_to_delete.index)
df_Trials_losses = df_Trials_losses.reset_index(drop=True)
df_Trials_losses['value_reward'] = df_Trials_losses['value_reward'].apply(lambda x: round(x * 2) / 2)
df_Trials_losses['value_safe'] = df_Trials_losses['value_safe'].apply(lambda x: round(x * 2) / 2)
df_Trials_losses.loc[df_Trials_losses['value_safe'] == 0, 'value_safe'] = -0.5
df_Trials_losses.loc[df_Trials_losses['value_reward'] == 0, 'value_reward'] = -0.5
#format to match previous PsychoPy input csv and add lott_top and lott_bot values
crdm_trials_losses = df_Trials_losses.copy()
crdm_trials_losses = crdm_trials_losses.sort_values('p_reward', ascending=True).reset_index(drop=True)
crdm_trials_losses['category'] = 'loss'
crdm_trials_losses = crdm_trials_losses.drop(columns=['SV_reward', 'SV_New_Safe', 'deltaSV'])
crdm_trials_losses['ambiguity'] = (crdm_trials_losses['ambiguity'] * 100).astype(int)
crdm_trials_losses['p_reward'] = (crdm_trials_losses['p_reward'] * 100).astype(int)
crdm_trials_losses = crdm_trials_losses.rename(columns={'value_safe': 'crdm_sure_amt',
                                                        'value_reward': 'crdm_lott',
                                                        'ambiguity': 'crdm_amb_lev',
                                                        'p_reward': 'crdm_lott_p', 
                                                        'category': 'crdm_domain'})
crdm_trials_losses['crdm_sure_p'] = 100
column_order = ['crdm_sure_amt', 'crdm_sure_p', 'crdm_lott', 'crdm_lott_p', 'crdm_amb_lev', 'crdm_domain'] #reordering columns
crdm_trials_losses = crdm_trials_losses[column_order]
crdm_trials_losses = crdm_trials_losses.sort_values(by=['crdm_lott_p', 'crdm_amb_lev'], ascending=[True,True])
zero_ambiguity_rows_loss = crdm_trials_losses[crdm_trials_losses['crdm_amb_lev'] == 0]
non_zero_ambiguity_rows_loss = crdm_trials_losses[crdm_trials_losses['crdm_amb_lev'] != 0]
crdm_trials_losses = pd.concat([zero_ambiguity_rows_loss, non_zero_ambiguity_rows_loss], ignore_index=True)
random_assignments = np.random.choice(['crdm_lott_top', 'crdm_lott_bot'], size=len(crdm_trials_losses))
crdm_trials_losses['crdm_lott_top'] = np.where(random_assignments == 'crdm_lott_top', crdm_trials_losses['crdm_lott'], 0)
crdm_trials_losses['crdm_lott_bot'] = np.where(random_assignments == 'crdm_lott_bot', crdm_trials_losses['crdm_lott'], 0)
column_order = ['crdm_sure_amt', 'crdm_sure_p', 'crdm_lott_top', 'crdm_lott_bot', 'crdm_lott_p', 'crdm_amb_lev', 'crdm_domain']
crdm_trials_loss = crdm_trials_losses[column_order]


#Joining GAIN & LOSS DataFrames and formatting for trial input CSV
crdm_trials = pd.concat([crdm_trials_gain, crdm_trials_loss], ignore_index=True)
crdm_trials['crdm_nonzero_side'] = crdm_trials.apply(determine_nonzero_side, axis=1) #create "crdm_nonzero_side
crdm_trials['crdm_img'] = crdm_trials.apply(determine_risk_image_file, axis=1) #create 'crdm_img' column
column_order = ['crdm_sure_amt', 'crdm_sure_p', 'crdm_lott_top', 'crdm_lott_bot', 'crdm_lott_p', 'crdm_amb_lev', 'crdm_domain', 'crdm_img', 'crdm_nonzero_side']
crdm_trials = crdm_trials[column_order]

#saving adaptive CSV to subject-specific folder
crdm_trials_csv = 'data/{0}/session{1}/{0}-S{1}_ADOtrials.csv'.format(subid, expInfo['session']) #csv location/format
crdm_trials.to_csv(crdm_trials_csv, float_format='%.2f', index=False) #saving csv