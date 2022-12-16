import pandas as pd
import numpy as np
import sys
import argparse
from scipy import stats
from statsmodels.stats.multitest import multipletests
import rpy2
import rpy2.robjects.packages as rpackages
from rpy2 import robjects
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)

# Install packages
packnames = ('exact2x2')
utils.install_packages(StrVector(packnames))
exact2x2 = importr('exact2x2')

## Calculate one-off OR
def calc_fisher(a,b,c,d,alternative="two.sided",unpack=False):
    m_test = robjects.r['matrix'](robjects.IntVector([a,b,c,d]), nrow=2)
    test_result = exact2x2.fisher_exact(m_test,tsmethod="minlike",alternative=alternative)
    p_val = test_result.rx2("p.value")[0]
    or_estimate = test_result.rx2("estimate")[0]
    conf_lower = float(test_result.rx2("conf.int")[0])
    conf_upper = float(test_result.rx2("conf.int")[1])
    if unpack:
        return or_estimate,conf_lower,conf_upper,p_val
    return_dict = {"OR":or_estimate,"OR_95CI_lower":conf_lower,"OR_95CI_upper":conf_upper,"p_val":p_val}
    return return_dict

# Given an nx4 array, where n1=x11, n2=x12, n3=x21, x4=x22 corresponding to contingency table
# return dataframe with OR, OR lower 95% CI, OR upper 95% CI, p-value, and q-value (alpha=0.05)
def calc_fisher_df(arrays,fdr_alpha=0.05):
    fisher_items = []
    for item in arrays:
        x11 = item[0]
        x12 = item[1]
        x21 = item[2]
        x22 = item[3]
        fisher_output = calc_fisher(x11,x12,x21,x22)
        fisher_items.append(fisher_output)
    to_return_df = pd.DataFrame(columns=["OR","OR_95CI_lower","OR_95CI_upper","p_val"],data=pd.json_normalize(fisher_items))
    uncorrected_p_values = to_return_df['p_val']
    reject,corrected_p_val,_,_ = multipletests(pvals=uncorrected_p_values,
                                            alpha=fdr_alpha,
                                            method='fdr_bh',
                                            is_sorted=False,
                                            returnsorted=False)

    to_return_df['fdr_corrected_p_val'] = corrected_p_val
    to_return_df['reject_fdr_corrected'] = reject
    return to_return_df

def calc_lr(a,b,c,d,conf=0.95):
    '''
    Formulation from: https://stats.stackexchange.com/questions/61349/how-to-calculate-the-confidence-intervals-for-likelihood-ratios-from-a-2x2-table
    '''
    alpha=1-conf
    critical_val = stats.norm.ppf(1-(alpha/2))
    spec = d/(b+d)
    sens = a/(a+c)

    if spec == 1:
        return {"LR+":np.Inf,"LR+_95CI_lower":np.NaN,"LR+_95CI_upper":np.NaN}

    lr_pos = sens/(1-spec)

    if (a!=0 and b!=0):
        sigma2 = (1/a) - (1/(a+c)) + (1/b) - (1/(b+d))
        lower_pos = lr_pos * np.exp(-critical_val*np.sqrt(sigma2))
        upper_pos = lr_pos * np.exp(critical_val*np.sqrt(sigma2))
    elif (a==0 and b==0):
        lower_pos = 0
        upper_pos = np.Inf
    elif (a==0 and b!=0):
        a_temp = (1/2)
        spec_temp = d/(b+d)
        sens_temp = a_temp/(a+c)
        lr_pos_temp = sens_temp/(1 - spec_temp)  
        sigma2 = (1/a_temp) - (1/(a_temp+c)) + (1/b) - (1/(b+d))
        lower_pos = 0
        upper_pos = lr_pos_temp * np.exp(critical_val*np.sqrt(sigma2))
    elif (a!=0 and b==0):
        b_temp = 1/2
        spec_temp = d/(b_temp+d)
        sens_temp = a/(a+c)
        lr_pos_temp = sens_temp / (1-spec_temp)
        sigma2 = (1/a) - (1/(a+c)) + (1/b_temp) - (1/(b_temp+d))
        lower_pos = lr_pos_temp * np.exp(-critical_val*np.sqrt(sigma2))
        upper_pos = np.Inf
    elif (a==(a+c)) and (b==(b+d)):
        a_temp = a - (1/2)
        b_temp = b - (1/2)
        spec_temp = d/(b_temp+d)
        sens_temp = a_temp/(a+c)
        lr_pos_temp = sens_temp/(1-spec_temp)
        sigma2 = (1/a_temp) - (1/(a_temp+c)) + (1/b_temp) - (1/(b_temp+d))
        lower_pos = lr_pos_temp * np.exp(-critical_val*np.sqrt(sigma2))
        upper_pos = lr_pos_temp * np.exp(critical_val*np.sqrt(sigma2))
    
    return_dict = {"LR+":lr_pos,"LR+_95CI_lower":lower_pos,"LR+_95CI_upper":upper_pos}
    return return_dict


# def calc_or(a,b,c,d,correction=True):
#     if (a==0 or b==0 or c==0 or d==0) and correction:
#         a+=0.5
#         b+=0.5
#         c+=0.5
#         d+=0.5
#     _,p_val = stats.fisher_exact([[a,b],[c,d]])
#     odds_ratio = (a/b)/(c/d)
#     se = np.sqrt(1/a+1/b+1/c+1/d)
#     upper = np.exp(np.log(odds_ratio)+1.96*se)
#     lower = np.exp(np.log(odds_ratio)-1.96*se)
#     return odds_ratio,p_val,lower,upper


