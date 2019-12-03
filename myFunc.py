# -- Set of Functions --
# -- Amirhessam Tahmassebi --

# Loading Libraries

import os, sys, gc
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
import shap
from scipy import stats
from scipy.special import expit
import statsmodels.nonparametric.api as smnp
from glmnet import LogitNet
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from IPython.display import display 
import seaborn as sns
sns.set_style("ticks")
import warnings
warnings.simplefilter("ignore")


# Pandas options
pd.set_option('expand_frame_repr', True)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_colwidth', -1) 
 
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

# My database info
my_db = "user_at066763"

# Tracking Python Kernel Performance
# . activate /tmp/vkpy36
# psrecord --log memuse.txt --plot memuse.png --duration 2000 interval 1 --include-children MYID

# --------------------------------------------------------------------------------------

class myColor:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# --------------------------------------------------------------------------------------

def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve, auc, confusion_matrix
    import itertools
    """
    Function that prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")

    plt.ylabel('True Class', fontsize = 14)
    plt.xlabel('Predicted Class', fontsize = 14)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()


# --------------------------------------------------------------------------------------

def _plot_xgboost_importance(bst, importance_type = "total_gain", color = "fuchsia", figsize = (10,10), max_num_features = 20):
    """
    Function to plot xgboost feature importance
    ----------------
    Parameters:
              - bst: trained xgboost model
              - importance_type: (default = "total_gain")
              - color: (default = "fuchsia")
              - figsize: (default = (10,10))
              - max_num_features: (default = 20)
    """
    from xgboost import plot_importance
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =2
    from pylab import rcParams
    rcParams['figure.figsize'] = figsize
    plot_importance(bst, importance_type = importance_type, color = color, xlabel = importance_type, max_num_features = max_num_features)
    plt.show()

# --------------------------------------------------------------------------------------

def _corr(df_, threshold = 0.9, impute = None, fillna = None):
    '''
    Function to find the ocrrelated variables
    -----------------
    Parameters:
               - df : dataframe
               - threshold : correlation threshold value as a cut-off (default = 0.9)
               - impute : string for dataframes with Null values : can be "mean" and "median"
                          If nothing passes, it will be None. Pandas by default drops the null values
               - fillna: a value to fill the null values. (default = None)

    '''
    # check for imputation
    if(impute == "median"):
        df = df_.copy()
        df.fillna(df.median(), inplace = True)
    elif(impute == "mean"):
        df = df_.copy()
        df.fillna(df.mean(), inplace = True)
    else:
        df = df_.copy()
        
    # check for fillna
    if(fillna != None):
        df.fillna(fillna, inplace = True)
        
        
    # pair-wised correlation:
        # methods can be pearosn, kendall, and spearman
        # min_periods sets the minimum number of observation for each feature
    corr = df.corr(method = "pearson", min_periods = 1)
    
    # mean of absoulte value of the of each column of the correlation matrix
    mean_corr_col = abs(corr).mean(axis = 1)
    
    # main Loop:
    
    # empty list for adding correlated columns
    correlated_cols = []
    for i in range(len(corr)):
        for j in range(len(corr)):
            if(i != j and corr.iloc[i,j] > threshold and mean_corr_col[i] > mean_corr_col[j]):
                correlated_cols.append(corr.columns[i])

    # dropping correlated features
    df_pruned = df.drop(list(set(correlated_cols)), axis = 1)
    
    return df_pruned

# --------------------------------------------------------------------------------------

def _plot_glmnet_cv_score(model):
    """
    Function to plot cv scores vs lambda
    ------------------
    Parameters:
               - model : a fitted glmnet object
    """
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =2
    plt.figure(figsize=(10,6))
    plt.errorbar(-np.log(model.lambda_path_), model.cv_mean_score_, yerr=model.cv_standard_error_ , c = "r", ecolor="k", marker = "o" )
    plt.vlines(-np.log(model.lambda_best_), ymin = min(model.cv_mean_score_) - 0.05 , ymax = max(model.cv_mean_score_) + 0.05, lw = 3, linestyles = "--", colors = "b" ,label = "best $\lambda$")
    plt.vlines(-np.log(model.lambda_max_), ymin = min(model.cv_mean_score_) - 0.05 , ymax = max(model.cv_mean_score_) + 0.05, lw = 3, linestyles = "--", colors = "c" ,label = "max $\lambda$")
    plt.tick_params(axis='both', which='major', labelsize = 12)
    plt.grid(True)
    plt.ylim([min(model.cv_mean_score_) - 0.05, max(model.cv_mean_score_) + 0.05])
    plt.legend(loc = 0, prop={'size': 20})
    plt.xlabel("$-Log(\lambda)$" , fontsize = 20)
    plt.ylabel(F"Mean {model.n_splits} Folds CV {(model.scoring).upper()}", fontsize = 20)
    plt.title(F"Best $\lambda$ = {model.lambda_best_[0]:.2} with {len(np.nonzero(  model.coef_)[1])} Features" , fontsize = 20)
    plt.show()

# --------------------------------------------------------------------------------------

def _plot_glmnet_coeff_path(model, df):
    """
    Function to plot coefficients vs lambda
    ------------------
    Parameters:
               - model : a fitted glmnet object
               - df: in case that the input is the pandas dataframe,
                   the column names of the coeff. will appear as a legend
    """
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =2
    plt.figure(figsize=(10,6))
    if not df.empty:
        for i in list(np.nonzero(np.reshape(model.coef_, (1,-1)))[1]):
            plt.plot(-np.log(model.lambda_path_) ,(model.coef_path_.reshape(-1,model.coef_path_.shape[-1]))[i,:], label = df.columns.values.tolist()[i]);
        plt.legend(loc= "right", bbox_to_anchor=(1.8 , .5), ncol=1, fancybox=True, shadow=True)

    else:
        for i in list(np.nonzero(np.reshape(model.coef_, (1,-1)))[1]):
            plt.plot(-np.log(model.lambda_path_) ,(model.coef_path_.reshape(-1,model.coef_path_.shape[-1]))[i,:]);
            
    plt.tick_params(axis='both', which='major', labelsize = 12)    
    plt.ylabel("Coefficients", fontsize = 20)
    plt.xlabel("-$Log(\lambda)$", fontsize = 20)
    plt.title(F"Best $\lambda$ = {model.lambda_best_[0]:.2} with {len(np.nonzero(  model.coef_)[1])} Features" , fontsize = 20)
    plt.grid(True)
    plt.show()

# --------------------------------------------------------------------------------------

def _df_glmnet_coeff_path(model, df):
    """
    Function to build a dataframe for nonzero coeff.
    ---------------------
    Parameters:
               - model : a fitted glmnet object
               - df: in case that the input is the pandas dataframe, the column names of the coeff. will appear as a legend
                   
    """
    idx = list(np.nonzero(np.reshape(model.coef_, (1,-1)))[1])
    dct = dict(zip([df.columns.tolist()[i] for i in idx], [model.coef_[0][i] for i in idx]))
        
    return pd.DataFrame(data = dct.items() , columns = ["Features", "Coeffs"]).sort_values(by = "Coeffs", ascending = False).reset_index(drop = True)
    
# --------------------------------------------------------------------------------------
def _df_A_glmnet_raw_coeff_path(model, df):
    """
    Function to build a dataframe for nonzero raw coeff based on the adaptive glmnet
    ---------------------
    Parameters:
               - model : a fitted glmnet object
               - df: in case that the input is the pandas dataframe, the column names of the coeff. will appear as a legend
                   
    """
    idx = list(np.nonzero(np.reshape(model.coef_/model.scales, (1,-1)))[1])
    dct = dict(zip([df.columns.tolist()[i] for i in idx], [(model.coef_/model.scales)[0][i] for i in idx]))
        
    return pd.DataFrame(data = dct.items() , columns = ["Features", "Coeffs"]).sort_values(by = "Coeffs", ascending = False).reset_index(drop = True)
# --------------------------------------------------------------------------------------

def _plot_roc_nfolds_glmnet(X_, Y,
                            n_folds = 10,
                            alpha = 0.5,
                            n_lambda = 100,
                            cut_point = 1.0,
                            shuffle = True,
                            random_state = 1367):
    """
    Function to plot k-fold cv ROC using glmnet
    ------------------------
    Parameters:
               - X_ : features (pandas dataframe, numpy array will be built inside the function)
               - Y : target values
               - n_folds : number of folds for cv (default = 10)
               - alpha : stability paramter for glmnet (default = 0.5, 0 for ridge and 1 for lasso)
               - n_lambda : number of penalty values for glmnet tunning (default = 100)
               - cut_point : the number of standard error between lambda best and lambda max (default = 1.0)
               - shuffle : shuffle flag for cv (default = True)
               - random_state : (default = 1367)
                    
    """
    
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import scale
    import matplotlib as mpl

    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =7


    # Defining the data + scaling
    if isinstance(X_, pd.DataFrame):
        X = scale(X_.values)
    else:
        X = scale(X_)

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits = n_folds , shuffle = shuffle , random_state = random_state)
    classifier = LogitNet(alpha = alpha,
                          n_lambda = n_lambda,
                          n_splits = n_folds,
                          cut_point = cut_point,
                          scoring = "roc_auc",
                          n_jobs = -1,
                          random_state = random_state
                         )
    

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(18 , 13))
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=3, alpha=0.5, 
                 label='ROC Fold %d (AUC = %0.2f)' % (i+1, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='k',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='navy',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=4)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4,
                     label=r'$\pm$ 1 Standard Deviation')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate' ,  fontweight = "bold" , fontsize=30)
    plt.ylabel('True Positive Rate',fontweight = "bold" , fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend( prop={'size':20} , loc = 4)
    # plt.savefig("./roc_glmnet.pdf" ,bbox_inches='tight')
    plt.show()
    
# --------------------------------------------------------------------------------------

def _plot_auc_hist_glmnet(X_, Y,
                            n_iter = 10,
                            n_folds = 10,
                            alpha = 0.5,
                            n_lambda = 100,
                            cut_point = 1.0,
                            shuffle = True,
                            seed = 1367):
    """
    a function to plot k-fold cv ROC using glmnet
    input parameters:
                     - X_ : features (pandas dataframe, numpy array will be built inside the function)
                     - Y : target values
                     - n_folds : number of folds for cv (default = 10)
                     - alpha : stability paramter for glmnet (default = 0.5, 0 for ridge and 1 for lasso)
                     - n_lambda : number of penalty values for glmnet tunning (default = 100)
                     - cut_point : the number of standard error between lambda best and lambda max (default = 1.0)
                     - shuffle : shuffle flag for cv (default = True)
                     - random_state : (default = 1367)
                    
    """
    
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import scale
    import matplotlib as mpl

    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =7

    # Defining the data + scaling
    if isinstance(X_, pd.DataFrame):
        X = scale(X_.values)
    else:
        X = scale(X_)
        
    y = Y
    n_samples, n_features = X.shape
    aucs = []
    for iteration in range(n_iter):
        random_state = seed * iteration
        cv = StratifiedKFold(n_splits = n_folds , shuffle = shuffle , random_state = random_state)
        classifier = LogitNet(alpha = alpha,
                              n_lambda = n_lambda,
                              n_splits = n_folds,
                              cut_point = cut_point,
                              scoring = "roc_auc",
                              n_jobs = -1,
                              random_state = random_state
                             )

        for train, test in cv.split(X, y):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
    plt.figure(figsize = (10,6))
    plt.hist(aucs, bins = 20, color = "pink", edgecolor = 'black', histtype= "bar")
    plt.xlabel('AUC' ,  fontweight = "bold" , fontsize=30)
    plt.ylabel('Frequency',fontweight = "bold" , fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    # plt.savefig("./hist_glmnet.pdf" ,bbox_inches='tight')
    plt.show()

# --------------------------------------------------------------------------------------

def _plot_roc_nfolds_xgboost(X_, Y,
                                   n_folds = 10,
                                   n_estimators = 100,
                                   learning_rate = 0.05,
                                   max_depth = 3,
                                   min_child_weight = 5.0,
                                   gamma = 0.5,
                                   reg_alpha = 0.0,
                                   reg_lambda = 1.0,
                                   subsample = 0.9,
                                   objective = "binary:logistic",
                                   scale_pos_weight = 1.0,
                                   shuffle = True,
                                   random_state = 1367,
                                   saveFigPath = None):
    """
    a function to plot k-fold cv ROC using xgboost
    input parameters:
                     df_X : features : pandas dataframe (numpy array will be built inside the function)
                     Y : targets
                     n_folds : number of cv folds (default = 10)
                     n_estimators = number of trees (default = 100)
                     learning_rate : step size of xgboost (default = 0.05)
                     max_depth : maximum tree depth for xgboost (default = 3)
                     min_child_weight : (default = 5.0)
                     gamma : (default = 0.5)
                     reg_alpha : lasso penalty (L1) (default = 0.0)
                     reg_lambda : ridge penalty (L2) (default = 1.0)
                     subsample : subsample fraction (default = 0.9)
                     objective : objective function for ML (default = "binary:logistic" for classification)
                     scale_pos_weight : (default = 1.0)
                     shuffle : shuffle flag for cv (default = True)
                     random_state : (default = 1367
                     saveFigPath: name of the file for saving the figure as PDF format. (default = None))
    
    """
    
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    from sklearn.preprocessing import scale
    import matplotlib as mpl

    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =7


    # Defining the data + scaling
    if isinstance(X_, pd.DataFrame):
        X = scale(X_.values)
    else:
        X = scale(X_)
        
    y = Y
    n_samples, n_features = X.shape

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits = n_folds , shuffle = shuffle , random_state = random_state)
    classifier = XGBClassifier(learning_rate = learning_rate,
                               n_estimators = n_estimators,
                               max_depth = max_depth,
                               min_child_weight = min_child_weight,
                               gamma = gamma,
                               reg_alpha = reg_alpha,
                               reg_lambda = reg_lambda,
                               subsample = subsample,
                               objective = objective,
                               nthread = 4,
                               scale_pos_weight = 1.,
                               base_score = np.mean(y),
                               seed = random_state,
                               random_state = random_state)
    

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(18 , 13))
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=3, alpha=0.5, 
                 label='ROC Fold %d (AUC = %0.2f)' % (i+1, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='k',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='navy',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=4)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.4,
                     label=r'$\pm$ 1 Standard Deviation')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate' ,  fontweight = "bold" , fontsize=30)
    plt.ylabel('True Positive Rate',fontweight = "bold" , fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend( prop={'size':20} , loc = 4)
    if saveFigPath:
        plt.savefig(F"./{saveFigPath}.pdf" ,bbox_inches='tight')
    plt.show()
    
# --------------------------------------------------------------------------------------    
    
def _permute(df, random_state):
    """
    Funtion to permute the rows of features and add them to the dataframe as noisy features to explore stability
    ---------------------
    Parameters:
               - df : pandas dataframe
               - random_state : random seed for noisy features
    """
    normal_df = df.copy().reset_index(drop = True)
    noisy_df = df.copy()
    noisy_df.rename(columns = {col : "noisy_" + col for col in noisy_df.columns}, inplace = True)
    np.random.seed(seed = random_state)
    noisy_df = noisy_df.reindex(np.random.permutation(noisy_df.index))
    merged_df = pd.concat([normal_df, noisy_df.reset_index(drop = True)] , axis = 1)
    
    return merged_df

# --------------------------------------------------------------------------------------

def _interaction_constraints(max_len, max_degree):
    """
    Function to produce a list of possible interactions
    up to the degree for interaction_constraints in xgboost
    ------------------------------------
    Parameters:
               - max_len : int; total number of features
               - degree: int; the max degree of interactions
               - output: string: list of lists of interactions based on the index of columns
    """
    from itertools import chain, combinations
    s = range(max_len)
    combs = list(chain.from_iterable(combinations(s, r) for r in range(2, max_degree + 1)))
    interactions = [list(comb) for comb in combs]

    return str(interactions)

# --------------------------------------------------------------------------------------

def _plot_xgboost_cv_score(cv_results):
    """
    Function to plot train/test cv score for 
    """

    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] = 2
    plt.figure(figsize=(10,8))
    plt.errorbar(range(cv_results.shape[0]), cv_results.iloc[:,0],
                yerr = cv_results.iloc[:,1], fmt = "--", ecolor="lightgreen", c = "navy", label = "Training")

    plt.errorbar(range(cv_results.shape[0]), cv_results.iloc[:,2],
                yerr = cv_results.iloc[:,3], fmt = "--", ecolor="lightblue", c = "red", label = "Testing")


    plt.xlabel("# of Boosting Rounds" ,  fontweight = "bold" , fontsize=20)
    plt.ylabel(F"""{(cv_results.columns.tolist()[0].split("-")[1]).upper()}""",fontweight = "bold" , fontsize=20)
    plt.title(F"""{(cv_results.columns.tolist()[0].split("-")[1]).upper()} vs Boosting Rounds""", fontweight = "bold" , fontsize=20) 
    plt.tick_params(axis='both', which='major', labelsize = 15)
    plt.legend(loc = 4, prop={'size': 20})
    plt.show()
    
# --------------------------------------------------------------------------------------    
    
def _binary_col_finder(df, drop_na = False):
    """
    Function find the binary features
    --------------------
    Parameters:
               - df: dataframe
               - drop_na: a flag to drop the null values, default is False
    """
    
    df = df.copy()

    if drop_na:
        bin_cols = [col for col in df if df[col].dropna().value_counts().index.isin([0, 1]).all()]
    else:
        bin_cols = [col for col in df if df[col].value_counts().index.isin([0, 1]).all()]
    
    return bin_cols

# --------------------------------------------------------------------------------------

def _continuous_col_finder(df, drop_na = False):
    """
    Function to find continous features
    --------------------------
    Parameters:
               - df: dataframe
               - drop_na: a flag to drop the null values, default is False
    """
    df = df.copy()
    bin_cols = _binary_col_finder(df, drop_na = drop_na)
    cont_cols = [col for col in df.columns.tolist() if col not in bin_cols]
    
    return cont_cols

# --------------------------------------------------------------------------------------

def _xy_interaction(df):
    """
    Function to add the second order interaction
    i.e. for x1, x2, x3: x1x2, x1x3, and x2x3 are added
    ---------------------
    Parameters :
                - df
                
    """
    df = df.copy()
    cols = df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            if(i != j):
                df[cols[i] + "*" + cols[j]] = df[cols[i]] * df[cols[j]]
                
    return df

# --------------------------------------------------------------------------------------

def _xy_interaction_shap(df, shap_values, sep):
    """
    Function to add the interacting features to the dataframe
    ----------------
    Parameters:
               - df: dataframe
               - shap_values: a list contains the interacted features like x1 * x2
               - sep: the delimiter used to separate the features: ie " * "
    """
    df_shap = df.copy()
    for col in shap_values:
        x1 = col.split(sep = sep)[0]
        x2 = col.split(sep = sep)[1]
        df_shap[x1 + " * " + x2] = df_shap[x1] * df_shap[x2]
    
    return df_shap

# --------------------------------------------------------------------------------------

def _mem_use_csr(csr):
    """
    Memory use in bytes by sparse data of a CSR matrix.
    """
    return csr.data.nbytes + csr.indptr.nbytes + csr.indices.nbytes


def _df_to_csr(df, fillna = 0., verbose = False):
    """
    Convert pandas DataFrame to a sparse CSR matrix.
    """
    csr = (
        df.astype("float")
        .fillna(fillna)
        .to_sparse(fill_value=fillna)
        .to_coo()
        .tocsr()
    )
    if verbose:
        df.info(memory_usage = "deep")
        print(F"CSR memory use: {_mem_use_csr(csr)/2**20:.3} MB")
    return csr

# --------------------------------------------------------------------------------------

def _slick_feature_selector(X, Y, n_iter = 1,
                                  nth_noisy_feature_threshold = 1,
                                  num_boost_round = 1000,
                                  nfold = 10,
                                  stratified = True,
                                  sparse_matrix = False,
                                  metrics = ("auc"),
                                  early_stopping_rounds = 20,
                                  seed = 1367,
                                  shuffle = True,
                                  show_stdv = False,
                                  params = None,
                                  importance_type = "total_gain",
                                  callbacks = False,
                                  verbose_eval = False,
                                  figsize = (8, 8)):
    """
    Function to run xgboost kfolds cv and train the model based on the best boosting round of each iteration.
    for different number of iterations. at each iteration noisy features are added as well. at each iteration
    internal and external cv calculated.
    NOTE: it is recommended to use ad-hoc parameters to make sure the model is under-fitted for the sake of feature selection
    -----------------------------
    Parameter- 
                  - X: features (pandas dataframe or numpy array)
                  - Y: targets (1D array or list)
                  - n_iter: total number of iterations (default = 1)
                  - nth_noisy_feature_threshold : threshold to apply the nth noisy threshold for feautre importance (default = 1)
                  - num_boost_rounds: max number of boosting rounds, (default = 1000)
                  - stratified: stratificaiton of the targets (default = True)
                  - sparse_matrix: converting pandas dataframe to sparse matrix (default = False)
                  - metrics: classification/regression metrics (default = ("auc))
                  - early_stopping_rounds: the criteria for stopping if the test metric is not improved (default = 20)
                  - seed: random seed (default = 1367)
                  - shuffle: shuffling the data (default = True)
                  - show_stdv = showing standard deviation of cv results (default = False)
                  - params = set of parameters for xgboost cv
                            (default_params = {
                                               "eval_metric" : "auc",
                                               "tree_method": "hist",
                                               "objective" : "binary:logistic",
                                               "learning_rate" : 0.05,
                                               "max_depth": 2,
                                               "min_child_weight": 1,
                                               "gamma" : 0.0,
                                               "reg_alpha" : 0.0,
                                               "reg_lambda" : 1.0,
                                               "subsample" : 0.9,
                                               "max_delta_step": 1,
                                               "silent" : 1,
                                               "nthread" : 4,
                                               "scale_pos_weight" : 1
                                               }
                            )
                  - importance_type = importance type of xgboost as string (default = "total_gain")
                                      the other options will be "weight", "gain", "cover", and "total_cover"
                  - callbacks = printing callbacks for xgboost cv
                                (defaults = False, if True: [xgb.callback.print_evaluation(show_stdv = show_stdv),
                                                             xgb.callback.early_stop(early_stopping_rounds)])
                  - verbose_eval : a flag to show the result during train on train/test sets (default = False)
                  - figsize : figure size for bar chart of important features (default = (8, 8))
    outputs:
            - outputs_dict: a dict contains the fitted models, internal and external cv results for train/test sets
            - df_selected: a dataframe contains feature names with their frequency of being selected at each run of folds
    """
    # sparse matrix flag
    if(sparse_matrix == True):
        print(myColor.BOLD + "-*-* " + myColor.DARKCYAN+ F"Pandas DataFrame converted to Sparse Matrix -- Sparsity Level = {_sparsity_level(X, fillna = 0.)}%" + myColor.END + myColor.BOLD + " *-*-")
    
    # callback flag
    if(callbacks == True):
        callbacks = [xgb.callback.print_evaluation(show_stdv = show_stdv),
                     xgb.callback.early_stop(early_stopping_rounds)]
    else:
        callbacks = None
    
    # params
    default_params = {
                      "eval_metric" : "auc",
                      "tree_method": "hist",
                      "objective" : "binary:logistic",
                      "learning_rate" : 0.05,
                      "max_depth": 2,
                      "min_child_weight": 1,
                      "gamma" : 0.0,
                      "reg_alpha" : 0.0,
                      "reg_lambda" : 1.0,
                      "subsample" : 0.9,
                      "max_delta_step": 1,
                      "silent" : 1,
                      "nthread" : 4,
                      "scale_pos_weight" : 1
                     }
    # updating the default parameters with the input
    if params is not None:
        for key, val in params.items():
            default_params[key] = val
            
    # total results list
    int_cv_train = []
    int_cv_test = []
    ext_cv_train = []
    ext_cv_test = []
    pruned_features = []
    
    # main loop
    
    for iteration in range(n_iter):
        
        # results list at iteration
        int_cv_train2 = []
        int_cv_test2 = []
        ext_cv_train2 = []
        ext_cv_test2 = []
        
        # update random state 
        random_state = seed * iteration 
    
        # adding noise to data
        X_permuted = _permute(df = X, random_state = random_state)
        cols = X_permuted.columns.tolist()
        Xval = X_permuted.values

        # building DMatrix for training/testing + kfolds cv
        cv = StratifiedKFold(n_splits = nfold , shuffle = shuffle , random_state = random_state)
        
        # set a counter for nfolds cv
        ijk = 1
        for train_index, test_index in cv.split(Xval, Y):
            X_train = pd.DataFrame(data = Xval[train_index], columns = cols)
            X_test = pd.DataFrame(data = Xval[test_index], columns = cols)
            Y_train = Y[train_index]
            Y_test = Y[test_index]
            
            if(sparse_matrix == False):
                dtrain = xgb.DMatrix(data = X_train, label = Y_train)
                dtest = xgb.DMatrix(data = X_test, label = Y_test)
            else:
                dtrain = xgb.DMatrix(data = _df_to_csr(X_train, fillna = 0., verbose = False), label = Y_train, feature_names = X_train.columns.tolist())
                dtest = xgb.DMatrix(data = _df_to_csr(X_test, fillna = 0., verbose = False), label = Y_test, feature_names = X_test.columns.tolist())
                

            # watchlist during final training
            watchlist = [(dtrain,"train"), (dtest,"eval")]
            
            # a dict to store training results
            evals_result = {}
            
            # xgb cv
            cv_results = xgb.cv(params = default_params,
                                dtrain = dtrain,
                                num_boost_round = num_boost_round,
                                nfold = nfold,
                                stratified = stratified,
                                metrics = metrics,
                                early_stopping_rounds = early_stopping_rounds,
                                seed = random_state,
                                verbose_eval = verbose_eval,
                                shuffle = shuffle,
                                callbacks = callbacks)
            int_cv_train.append(cv_results.iloc[-1][0])
            int_cv_test.append(cv_results.iloc[-1][2])
            int_cv_train2.append(cv_results.iloc[-1][0])
            int_cv_test2.append(cv_results.iloc[-1][2])
            
            # xgb train
            bst = xgb.train(params = default_params,
                            dtrain = dtrain,
                            num_boost_round = len(cv_results) - 1,
                            evals = watchlist,
                            evals_result = evals_result,
                            verbose_eval = verbose_eval)
            
            feature_gain = _feature_importance_to_dataframe(bst)
            # check wheather noisy feature is selected
            if feature_gain.feature.str.contains("noisy").sum() != 0:
                gain_threshold = feature_gain.loc[feature_gain.feature.str.contains("noisy") == True, importance_type].values.tolist()[nth_noisy_feature_threshold-1]
            else:
                gain_threshold = 0.0
            # subsetting features for > gain_threshold    
            gain_subset = feature_gain.loc[feature_gain[importance_type] > gain_threshold, "feature"].values.tolist()
            for c in gain_subset:
                pruned_features.append(c)
                
            # appending outputs
            ext_cv_train.append(evals_result["train"][metrics][-1])
            ext_cv_test.append(evals_result["eval"][metrics][-1])
            ext_cv_train2.append(evals_result["train"][metrics][-1])
            ext_cv_test2.append(evals_result["eval"][metrics][-1])
            
            # free memory here at each fold
            del gain_subset, feature_gain, bst, watchlist, Y_train, Y_test, cv_results, evals_result, X_train, X_test, dtrain, dtest
            
            print(myColor.BOLD + "*-*-*-*-*-*-*-*- " + myColor.RED + F"Memory Got Explicitly Free -- Fold = {ijk}/{nfold}" + myColor.END + myColor.BOLD +" -*-*-*-*-*-*-*-*")
            ijk += 1
            gc.collect()
            
        print(myColor.BOLD + "*-*-*-*-*-*-*-*- " + myColor.DARKCYAN + F"Iteration {iteration + 1}" + myColor.END + myColor.BOLD + " -*-*-*-*-*-*-*-*")
        print(myColor.BOLD + "*-*- " + myColor.GREEN+ F"Internal-{nfold} Folds-CV-Train = {np.mean(int_cv_train2):.3f} +/- {np.std(int_cv_train2):.3f}" + myColor.END + myColor.BOLD + " -*-*- " + myColor.GREEN + F"Internal-{nfold} Folds-CV-Test = {np.mean(int_cv_test2):.3f} +/- {np.std(int_cv_test2):.3f}"+ myColor.END + myColor.BOLD +" -*-*")
        print(myColor.BOLD +"*-*- "+ myColor.GREEN+ F"External-{nfold} Folds-CV-Train = {np.mean(ext_cv_train2):.3f} +/- {np.std(ext_cv_train2):.3f}"+ myColor.END + myColor.BOLD +" -*-*- " + myColor.GREEN + F"External-{nfold} Folds-CV-Test = {np.mean(ext_cv_test2):.3f} +/- {np.std(ext_cv_test2):.3f}"+ myColor.END + myColor.BOLD + " -*-*")
        
        # free memory here at iteration
        del int_cv_train2, int_cv_test2, ext_cv_train2, ext_cv_test2, X_permuted, cols, Xval, cv
        gc.collect()

        
    # putting together the outputs in one dict
    outputs = {}
    outputs["int_cv_train"] = int_cv_train
    outputs["int_cv_test"] = int_cv_test
    outputs["ext_cv_train"] = ext_cv_train
    outputs["ext_cv_test"] = ext_cv_test
    
    # Plotting
    import matplotlib as mpl

    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] = 7


    unique_elements, counts_elements = np.unique(pruned_features, return_counts = True)
    counts_elements = [float(i) for i in list(counts_elements)]
    df_features = pd.DataFrame(data = {"columns" : list(unique_elements) , "count" : counts_elements})
    df_features.sort_values(by = ["count"], ascending = False, inplace = True)

    plt.figure()
    df_features.sort_values(by = ["count"]).plot.barh(x = "columns", y = "count", color = "navy" , figsize = figsize)
    plt.show()

    plt.figure(figsize = (16,12))

    plt.subplot(2,2,1)
    plt.title(F"Internal {nfold} CV {metrics.upper()} - Train", fontsize = 20)
    plt.hist(outputs["int_cv_train"], bins = 20, color = "lightgreen")

    plt.subplot(2,2,2)
    plt.title(F"Internal {nfold} CV {metrics.upper()} - Test", fontsize = 20)
    plt.hist(outputs["int_cv_test"], bins = 20, color = "lightgreen")

    plt.subplot(2,2,3)
    plt.title(F"External {nfold} CV {metrics.upper()} - Train", fontsize = 20)
    plt.hist(outputs["ext_cv_train"], bins = 20, color = "lightblue")

    plt.subplot(2,2,4)
    plt.title(F"External {nfold} CV {metrics.upper()} - Test", fontsize = 20)
    plt.hist(outputs["ext_cv_test"], bins = 20, color = "lightblue")

    plt.show()
    
    display(df_features.reset_index(drop = True))
    
    return outputs, df_features.sort_values(by = ["count"], ascending = False).reset_index(drop = True)


# --------------------------------------------------------------------------------------

def _plot_slick_outputs(outputs, df_features):
    """
    Function to plot slick feature selection results
    ----------------------------
    Parameters:
               - outputs: dictionary of all internal/external cvs
               - df_features: dataframe with feature and count of presence
    """
    
    plt.figure()
    df_features.sort_values(by = ["count"]).plot.barh(x = "columns", y = "count", color = "navy" , figsize = (8,8))
    plt.show()

    plt.figure(figsize = (16,12))

    plt.subplot(2,2,1)
    plt.title(F"Internal {nfold} CV {metrics.upper()} - Train", fontsize = 20)
    plt.hist(outputs["int_cv_train"], bins = 20, color = "lightgreen")

    plt.subplot(2,2,2)
    plt.title(F"Internal {nfold} CV {metrics.upper()} - Test", fontsize = 20)
    plt.hist(outputs["int_cv_test"], bins = 20, color = "lightgreen")

    plt.subplot(2,2,3)
    plt.title(F"External {nfold} CV {metrics.upper()} - Train", fontsize = 20)
    plt.hist(outputs["ext_cv_train"], bins = 20, color = "lightblue")

    plt.subplot(2,2,4)
    plt.title(F"External {nfold} CV {metrics.upper()} - Test", fontsize = 20)
    plt.hist(outputs["ext_cv_test"], bins = 20, color = "lightblue")

    plt.show()

# --------------------------------------------------------------------------------------    

def _bst_shap(X_, Y, num_boost_round = 1000,
                                    nfold = 10,
                                    stratified = True,
                                    sparse_matrix = False,
                                    scale_X = False,
                                    metrics = ("auc"),
                                    early_stopping_rounds = 20,
                                    seed = 1367,
                                    shuffle = True,
                                    show_stdv = False,
                                    params = None,
                                    callbacks = False):
    """
    Function to run xgboost kfolds cv with the params and train the model based on best boosting rounds
    -------------------------
    Parameters:
               - X_: features
               - Y: targets
               - num_boost_rounds: max number of boosting rounds, (default = 1000)
               - stratified: stratificaiton of the targets (default = True)
               - sparse_matrix: converting pandas dataframe to sparse matrix (default = False)
               - scale_X: Flag to run scale preprocessing module to have a mean of zero and unit variance. It would change the data to non-sparse format. (default = False)
               - metrics: classification/regression metrics (default = ("auc"))
               -          Full list at https://xgboost.readthedocs.io/en/latest/parameter.html
               - early_stopping_rounds: the criteria for stopping if the test metric is not improved (default = 20)
               - seed: random seed (default = 1367)
               - shuffle: shuffling the data (default = True)
               - show_stdv = showing standard deviation of cv results (default = False)
               - params = set of parameters for xgboost cv ( default_params = {
                                                                                  "eval_metric" : "auc",
                                                                                  "tree_method": "hist",
                                                                                  "objective" : "binary:logistic",
                                                                                  "learning_rate" : 0.05,
                                                                                  "max_depth": 2,
                                                                                  "min_child_weight": 1,
                                                                                  "gamma" : 0.0,
                                                                                  "reg_alpha" : 0.0,
                                                                                  "reg_lambda" : 1.0,
                                                                                  "subsample" : 0.9,
                                                                                  "max_delta_step": 1,
                                                                                  "silent" : 1,
                                                                                  "nthread" : 4,
                                                                                  "scale_pos_weight" : 1
                                                                                 }
                                                                    )
                   callbacks = printing callbacks for xgboost cv (defaults: [xgb.callback.print_evaluation(show_stdv = show_stdv),
                                                                             xgb.callback.early_stop(early_stopping_rounds)])
    outputs:
            bst: the trained model based on optimum number of boosting rounds
            cv_results: a dataframe contains the cv-results (train/test + std of train/test) for metric
    """
    # Defining the data + scaling
    if scale_X == True:
        if isinstance(X_, pd.DataFrame):
            X = pd.DataFrame(scale(X_), columns = X_.columns.tolist())
        else:
            X = scale(X_)
    else:
        X = X_

    # sparse matrix flag
    if(sparse_matrix == True):
         print(myColor.BOLD + "-*-* " + myColor.DARKCYAN+ F"Pandas DataFrame converted to Sparse Matrix -- Sparsity Level = {_sparsity_level(X, fillna = 0.)}%" + myColor.END + myColor.BOLD + " *-*-")


    # callback flag
    if(callbacks == True):
        callbacks = [xgb.callback.print_evaluation(show_stdv = show_stdv),
                     xgb.callback.early_stop(early_stopping_rounds)]
    else:
        callbacks = None
    
    # params
    default_params = {
                      "eval_metric" : "auc",
                      "tree_method": "hist",
                      "objective" : "binary:logistic",
                      "learning_rate" : 0.05,
                      "max_depth": 2,
                      "min_child_weight": 1,
                      "gamma" : 0.0,
                      "reg_alpha" : 0.0,
                      "reg_lambda" : 1.0,
                      "subsample" : 0.9,
                      "max_delta_step": 1,
                      "silent" : 1,
                      "nthread" : 4,
                      "scale_pos_weight" : 1
                     }
    # updating the default parameters with the input
    if params is not None:
        for key, val in params.items():
            default_params[key] = val

    # building DMatrix for training
    if(sparse_matrix == False):
        dtrain = xgb.DMatrix(data = X, label = Y)
    else:
        dtrain = xgb.DMatrix(data = _df_to_csr(X, fillna = 0., verbose = False), label = Y, feature_names = X.columns.tolist())

    print(myColor.BOLD + "-*-*-*-*-*-*-* " + myColor.DARKCYAN + F"{nfold}-Folds Cross-Validation Started" + myColor.END + myColor.BOLD + " *-*-*-*-*-*-*-")
    


    cv_results = xgb.cv(params = default_params,
                        dtrain = dtrain,
                        num_boost_round = num_boost_round,
                        nfold = nfold,
                        stratified = stratified,
                        metrics = metrics,
                        early_stopping_rounds = early_stopping_rounds,
                        seed = seed,
                        shuffle = shuffle,
                        callbacks = callbacks)
    print(myColor.BOLD +"*-*- " + myColor.GREEN + F"Boosting Round = {len(cv_results) - 1} -*-*- Train = {cv_results.iloc[-1][0]:.3} -*-*- Test = {cv_results.iloc[-1][2]:.3}" + myColor.END + myColor.BOLD +  " -*-*")
    # now train the model at the best boosting round
    bst = xgb.train(params = default_params,
                    dtrain = dtrain,
                    num_boost_round = len(cv_results) - 1,
                   )
    
    return bst, cv_results

# --------------------------------------------------------------------------------------

def _A_glmnet(X_, Y, alpha = 0.5, n_splits = 10, alpha_ini = 0.0, penalty_gamma = 2, random_state = 1367, scale_X = True):
    """
    Function to return a trained adaptive glmnet model
    ---------------------------------
    Parameters:
               - X_ : features set which can be pandas dataframe and numpy array
               - Y : target/response values
               - alpha : stability parameter, 0.0 for ridge and 1.0 for lasso (default = 0.5)
               - n_splits : number of folds CV for finding penalty parameters lambda (default = 10)
               - alpha_ini : (default = 0.0)
               - penalty_gamma : gamma penalty value which can be lower for more features to keep (default = 2.0)
               - random_state: Random state seed (default = 1367)
               - scale_X: flag for scaling the feautures (default = True)
              
              
    """
    if scale_X == True:
        # Defining the data + scaling
        if isinstance(X_, pd.DataFrame):
            X = pd.DataFrame(scale(X_), columns = X_.columns.tolist())
        else:
            X = scale(X_)
    else:
        X = X_
        
    def get_stratified_cv_test_indices(y, n_splits = n_splits, random_state = random_state):
        """ Returns an array containing a test fold id for each sample
        that is suitable as a test_fold input parameter to PredefinedSplit(test_fold)
        """
        skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        X = np.zeros(len(y)) # dummy X
        test_folds = np.array([0.]*len(y))
        fold = 0
        for _, test_id in skf.split(X, y):
            test_folds[test_id] = fold
            fold += 1
        return test_folds
    
    test_folds = get_stratified_cv_test_indices(Y, n_splits = n_splits, random_state=random_state)
    a = AdaptiveLogitNet(alpha = alpha, alpha_ini = alpha_ini, penalty_gamma = penalty_gamma, scoring="roc_auc", n_jobs = 5, verbose = False, n_splits = n_splits, random_state = test_folds) 
    a.CV = DisguisedPredefinedSplit
    a.fit(X.values, Y);
    return a

# --------------------------------------------------------------------------------------
def _stable_alpha_glmnet(X_, Y, alpha_list = None, n_splits = 10, alpha_ini = 0.0, penalty_gamma = 2, scale_X = True):
    """
    Function to find the stable alpha for adaptive glmnet
    ---------------------
    Parameters:
               - X_: feature matrix (pandas dataframe)
               - Y: class labels
               - alpha_list: list alphas (default = [0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91])
               - n_splits: number of folds for cross-validation (default = 10)
               - alpha_ini: initialization point for alpha (default = 0.0)
               - penalty_gamma: penalizing factor for gamma (default = 2)
               - scale_X: flag for scaling the feautures (default = True)
    
    """
    if scale_X == True:
        # Defining the data + scaling
        if isinstance(X_, pd.DataFrame):
            X = pd.DataFrame(scale(X_), columns = X_.columns.tolist())
        else:
            X = scale(X_)
    else:
        X = X_
        
    def logitnet_nonzero_coef(g):
        idx = np.where(g.coef_[0] != 0)[0]
        return pd.DataFrame({"idx": idx, "coef": g.coef_[0][idx]})
    
    # updating alpha_list if needed
    if(alpha_list == None):
        alpha_list = np.arange(0.01, 1, 0.1)
        
    mean_auc = []
    std_auc = []
    n_features = []
    for alpha in alpha_list:
        model = _A_glmnet(X, Y, alpha = alpha, n_splits = n_splits, alpha_ini = alpha_ini, penalty_gamma = penalty_gamma)
        mean_auc.append(model.cv_mean_score_[model.lambda_best_inx_])
        std_auc.append(model.cv_standard_error_[model.lambda_best_inx_])
        fe = logitnet_nonzero_coef(model)
        n_features.append(fe.shape[0])
        
        
    # Plotting the results
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] =2
    plt.figure(figsize=(20,6))

    plt.subplot(1,2,1)
    plt.errorbar(alpha_list, mean_auc, yerr = std_auc, c = "r", ecolor="k", marker = "o", )
    plt.tick_params(axis='both', which='major', labelsize = 12)
    plt.xlabel(r"$\alpha$" , fontsize = 20)
    plt.ylabel("Mean 10 Folds CV AUC", fontsize = 20)
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(alpha_list, n_features, c = "r",  marker = "o", )
    plt.tick_params(axis='both', which='major', labelsize = 12)
    plt.xlabel(r"$\alpha$" , fontsize = 20)
    plt.ylabel("Number of Features", fontsize = 20)
    plt.grid(True)

    plt.show()
# --------------------------------------------------------------------------------------

def _S_glmnet(X_, Y, alpha = 0.5, n_splits = 10, scoring = "roc_auc", scale_X = True):
    """
    Function for standard glmnet
    -------------------------
    Parameters:
               X_ : feature set
               Y : target values
               alpha : stability parameter (default = 0.5)
               n_splits : number of folds CV for finding lambda (defualt = 10)
               scoring : metric for classificaiton (default = "roc_auc")
    output: trained glmnet model 

    """
    if scale_X == True:
        # Defining the data + scaling
        if isinstance(X_, pd.DataFrame):
            X = pd.DataFrame(scale(X_), columns = X_.columns.tolist())
        else:
            X = scale(X_)
    else:
        X = X_
        
    from glmnet import LogitNet  
    model = LogitNet(alpha = alpha,
                     n_lambda = 100,
                     n_splits = n_splits,
                     cut_point = 1.0,
                     scoring = scoring,
                     n_jobs = -1,
                     random_state = 1367)
    model.fit(X, Y)
    
    return model

# --------------------------------------------------------------------------------------
def _slick_bayesian_optimization(X_, Y, n_iter = 5,
                                       init_points = 5,
                                       acq = "ei",
                                       num_boost_round = 1000,
                                       nfold = 10,
                                       stratified = True,
                                       metrics = ("auc"),
                                       early_stopping_rounds = 20,
                                       seed = 1367,
                                       shuffle = True,
                                       show_stdv = False,
                                       pbounds = None,
                                       importance_type = "total_gain",
                                       callbacks = False,
                                       verbose_eval = False):
    """
    Function to run Bayesian Optimization for XGBoost
    -----------------------
    Parameters:
                    X_: features (pandas dataframe or numpy array)
                    Y: targets (1D array or list)
                    n_iter: total number of bayesian iterations (default = 5)
                    init_points: total initial points of optimization (default = 5)
                    acq
                    num_boost_rounds: max number of boosting rounds, (default = 1000)
                    stratified: stratificaiton of the targets (default = True)
                    metrics: classification/regression metrics (default = ("auc))
                    early_stopping_rounds: the criteria for stopping if the test metric is not improved (default = 20)
                    seed: random seed (default = 1367)
                    shuffle: shuffling the data (default = True)
                    show_stdv = showing standard deviation of cv results (default = False)
                    pbounds = set of parameters for bayesian optimization of xgboost cv
                            (default_params = {
                                               "eval_metric" : "auc" ("aucpr" is also recommended),
                                               "tree_method": "hist",
                                               "objective" : "binary:logistic",
                                               "learning_rate" : 0.05,
                                               "max_depth": 2,
                                               "min_child_weight": 1,
                                               "gamma" : 0.0,
                                               "reg_alpha" : 0.0,
                                               "reg_lambda" : 1.0,
                                               "subsample" : 0.9,
                                               "max_delta_step": 1,
                                               "silent" : 1,
                                               "nthread" : 4,
                                               "scale_pos_weight" : 1
                                               }
                            )
                    importance_type = importance type of xgboost as string (default = "total_gain")
                                      the other options will be "weight", "gain", "cover", and "total_cover"
                    callbacks = printing callbacks for xgboost cv
                                (defaults = False, if True: [xgb.callback.print_evaluation(show_stdv = show_stdv),
                                                             xgb.callback.early_stop(early_stopping_rounds)])
                    verbose_eval : a flag to show the result during train on train/test sets (default = False)
                    figsize: the figure size for feature importance (default = (10,15))
    outputs:
            df_res: the parameters related to the best performance
            xgb_params: a dictionary of the best parameters of xgboost                
    
    
    """
    from bayes_opt import BayesianOptimization as bo
    import xgboost as xgb
    import numpy as np
    import pandas as pd 
    import matplotlib.pyplot as plt
    import warnings
    warnings.simplefilter("ignore")
    from IPython.core.display import display, HTML
    display(HTML("<style>.container {width:95% !important; }</style>"))
    
    # callback flag
    if(callbacks == True):
        callbacks = [xgb.callback.print_evaluation(show_stdv = show_stdv),
                     xgb.callback.early_stop(early_stopping_rounds)]
    else:
        callbacks = None
        
    # Defining the data + scaling
    if isinstance(X_, pd.DataFrame):
        X = pd.DataFrame(scale(X_), columns = X_.columns.tolist())
    else:
        X = scale(X_)
    
    # pbounds
    default_pbounds = {"max_depth" : (2, 5),
                       "learning_rate" : (0, 1), 
                       "min_child_weight" : (1, 20),
                       "subsample" : (0.1, 1),
                       "gamma": (0, 1),
                       "colsample_bytree": (0.1, 1.0)
                      }
    
    # updating the default parameters of the pbounds
    if pbounds is not None:
        for key, val in pbounds.items():
            default_pbounds[key] = val
    
    
    def __xgb_eval(learning_rate,
                   max_depth,
                   gamma,
                   colsample_bytree,
                   min_child_weight,
                   subsample):

        params = {"eval_metric" : metrics,
                  "tree_method": "hist",
                  "objective" : "binary:logistic",
                  "max_delta_step": 1,
                  "silent" : 1,
                  "nthread" : 4,
                  "scale_pos_weight" : 1,
                  "reg_alpha" : 0.0,
                  "reg_lambda" : 1.0,
                  "learning_rate" : learning_rate,
                  "max_depth": int(max_depth),
                  "min_child_weight": min_child_weight,
                  "gamma" : gamma,
                  "subsample" : subsample,
                  "colsample_bytree" : colsample_bytree 
                 }
        dtrain = xgb.DMatrix(data = X, label = Y)
        cv_result = xgb.cv(params = params,
                           dtrain = dtrain,
                           num_boost_round = num_boost_round,
                           nfold = nfold,
                           stratified = stratified,
                           metrics = metrics,
                           early_stopping_rounds = early_stopping_rounds,
                           seed = seed,
                           verbose_eval = verbose_eval,
                           shuffle = shuffle,
                           callbacks = callbacks)

        return cv_result.iloc[-1][2]
    

    xgb_bo = bo(__xgb_eval, default_pbounds, random_state = seed, verbose = 2)
    xgb_bo.maximize(init_points = init_points, n_iter = n_iter, acq = acq)
    
    
    targets = []
    for i, rs in enumerate(xgb_bo.res):
        targets.append(rs["target"])
    best_params = xgb_bo.res[targets.index(max(targets))]["params"]
    best_params["max_depth"] = int(best_params["max_depth"])
    
    xgb_params = {"eval_metric" : metrics,
                  "tree_method": "hist",
                  "objective" : "binary:logistic",
                  "max_delta_step": 1,
                  "silent" : 1,
                  "nthread" : 4,
                  "scale_pos_weight" : 1,
                  "reg_alpha" : 0.0,
                  "reg_lambda" : 1.0,
                  "learning_rate" : 0.05,
                  "max_depth": 2,
                  "min_child_weight": 5,
                  "gamma" : 0.0,
                  "subsample" : 1.0,
                  "colsample_bytree" : 0.9 
                 }
    for key, val in best_params.items():
        xgb_params[key] = val
    
    dtrain = xgb.DMatrix(data = X, label = Y)
    bst = xgb.train(params = xgb_params,
                    dtrain = dtrain,
                    num_boost_round = num_boost_round)
    
    # build results dataframe
    frames = []
    for idx, res in enumerate(xgb_bo.res):
        d = res['params']
        d[metrics] = res["target"]
        frames.append(pd.DataFrame(data = d, index = [idx]))
    
    res_df = pd.concat(frames)
   
    print(F"-*-*-*-*-*-* Optimization Results -*-*-*-*-*-*")
    display(res_df)
    
    # Plotting
    import matplotlib as mpl

    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] = 3
    cols = [col for col in res_df.columns.tolist() if col != metrics]
    ip = 1
    plt.figure(figsize = (22, 10))
    colors = ["navy", "lavender", "lightblue", "cyan", "cadetblue", "slateblue"]
    for col in cols:
        res_df.sort_values(by = col, inplace=True)
        plt.subplot(2,3,ip)
        plt.plot(res_df.loc[:, col], res_df.loc[:, metrics], color = colors[ip-1])
        plt.xlabel(F"{col}", fontsize = 20)
        plt.ylabel(F"{metrics}", fontsize = 20)
        plt.tick_params(axis='both', which='major', labelsize = 12)
        ip += 1
    plt.show()
    
    print(F"-*-*-*-*-*-* Best Performance -*-*-*-*-*-*")
    display(res_df.loc[res_df[metrics] == res_df[metrics].max(), :])
    
    from xgboost import plot_importance
    from pylab import rcParams
    rcParams['figure.figsize'] = (10, int(X.shape[1] * 0.5))
    plot_importance(bst, importance_type = importance_type, color = "skyblue", xlabel = importance_type)
    plt.show()   
    
    return res_df, xgb_params

# --------------------------------------------------------------------------------------
def _pd_explode(df, column):
    
    """
    (local): Explodes a column into columnar format.
    Required Parameter(s):
        - df (Pandas Data Frame): Data Frame to be passed in.
        - column (StringType): Name of column wanting to explode.
    """
    vals = df[column].values.tolist()
    rs = [len(r) for r in vals]
    a = np.repeat(df[[col for col in df.columns.tolist() if col != column]].values, rs, axis=0)
    return pd.DataFrame(np.column_stack((a, np.concatenate(vals))), columns=df.columns)

# --------------------------------------------------------------------------------------
def _write_pickle(objectFile, path, protocol = 3):
    """
    Function to write a file into path as pickle
    --------------------------
    Parameters:
              objectFile: an object e.g. a dataframe
              path: a string
              protocol: protocol to save pickle files (default = 3).
    """
    import pickle
    pickle_file = open(file = path , mode = "wb")
    pickle.dump(obj = objectFile, file = pickle_file, protocol = protocol)
    pickle_file.close()

# --------------------------------------------------------------------------------------

def _pd_struct_explode(df, column):
    """
    (local): Explodes a column into columnar format.
    Required Parameter(s):
        - df (Pandas Data Frame): Data Frame to be passed in.
        - column (StringType): Name of column wanting to explode.
    """
    dicts = [val.asDict() for val in df[column].values.tolist()]
    keys = dicts[0].keys()
    column_names = str(list(df.columns)).replace(F"{column}", F"{str(list(keys)).replace('[','').replace(']','')}").replace("''","'")
    rs = [[d[key] for key in keys] for d in dicts]
    return pd.concat([pd.DataFrame(rs, columns=keys), df.drop([column], axis=1)], axis=1).loc[:, eval(column_names)]
# --------------------------------------------------------------------------------------

def _threshold_finder(model, X, y_true):
    """
    Function to find the optimal threshold for binary classification
    ------------------------
    Parameters:
               - model: a trained model object (such as xgboost, glmnet, ...)
               - X: the test set of features (pandas dataframe or numpy array)
               - y_true: the true class labels (list or array of 0's and 1's)    
    """
    
    y_predict_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_predict_proba)
    auc = roc_auc_score(y_true, y_predict_proba)
    precision, recall, thresholds2 = precision_recall_curve(y_true, y_predict_proba)
    
    class_names = [0, 1]
    youden_idx = np.argmax(np.abs(tpr - fpr))
    youden_threshold = thresholds[youden_idx]
    y_pred_youden = (y_predict_proba > youden_threshold).astype(int)
    cnf_matrix = confusion_matrix(y_true, y_pred_youden)
    np.set_printoptions(precision=2)
    
    f1 = []
    for i in range(len(precision)):
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
        
    queue_rate = []
    for thr in thresholds2:
        queue_rate.append((y_predict_proba >= thr).mean()) 

    plt.figure(figsize = (10, 5))
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, color = "red", label = F"AUC = {auc:.3f}")
    plt.plot(fpr[youden_idx], tpr[youden_idx], marker = "o", color = "navy", ms =10, label =F"Youden Threshold = {youden_threshold:.2f}" )
    plt.axvline(x = fpr[youden_idx], ymin = fpr[youden_idx], ymax = tpr[youden_idx], color = "navy", ls = "--")
    plt.plot([0,1], [0,1] , color = "black", ls = "--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Specificity' , fontsize=12)
    plt.ylabel('Sensitivity' , fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend( prop={'size':12} , loc = 4)

    plt.subplot(1,2,2)
    _plot_confusion_matrix(cnf_matrix, classes=class_names, normalize = False, cmap=plt.cm.Reds, title = F"Youden Threshold = {youden_threshold:.2f}\nAccuracy = {accuracy_score(y_true, y_pred_youden)*100:.2f}%")
    plt.show()
    
    plt.figure(figsize = (12, 5))
    plt.subplot(1,2,1)
    plt.plot(thresholds, 1-fpr, label = "1 - Specificity")
    plt.plot(thresholds, tpr, label = "Sensitivity")
    plt.xlabel("Threshold", fontsize = 12)
    plt.ylabel("Score", fontsize = 12)
    plt.legend(loc = 0)
    plt.xlim([0.025, 0.2])
    plt.axvline(thresholds[np.argmin(abs(tpr + fpr - 1))], color="k", ls = "--")
    plt.title(F"Threshold = {thresholds[np.argmin(abs(tpr + fpr - 1))]:.3f}", fontsize = 12)
    
    plt.subplot(1,2,2)
    plt.plot(thresholds2, precision[1:], label = "Precision")
    plt.plot(thresholds2, recall[1:], label = "Recall")
    plt.plot(thresholds2, f1[1:], label = "F1-Score")
    plt.plot(thresholds2, queue_rate, label = "Queue Rate")
    plt.legend(loc = 0)
    plt.xlim([0.025, 0.25])
    plt.xlabel("Threshold", fontsize = 12)
    plt.ylabel("Score", fontsize = 12)
    plt.axvline(thresholds2[np.argmin(abs(precision-recall))], color="k", ls = "--")
    plt.title(label = F"Threshold = {thresholds2[np.argmin(abs(precision-recall))]:.3f}", fontsize = 12)
    plt.show()
    
# --------------------------------------------------------------------------------------
   
def _plot_pr_nfolds_xgboost(X_, Y,
                                     n_folds = 10,
                                     n_estimators = 100,
                                     learning_rate = 0.05,
                                     max_depth = 3,
                                     min_child_weight = 5.0,
                                     gamma = 0.5,
                                     reg_alpha = 0.0,
                                     reg_lambda = 1.0,
                                     subsample = 0.9,
                                     objective = "binary:logistic",
                                     scale_pos_weight = 1.0,
                                     shuffle = True,
                                     random_state = 1367,
                                     saveFigPath = None
                            ):
    """
    Function to plot k-fold cv Precision-Recall (PR) using xgboost
    -----------------------------------------
    Parameters:
               - X_ : features in pandas dataframe (numpy array will be built inside the function)
               - Y : targets
               - n_folds : number of cv folds (default = 10)
               - n_estimators = number of trees (default = 100)
               - learning_rate : step size of xgboost (default = 0.05)
               - max_depth : maximum tree depth for xgboost (default = 3)
               - min_child_weight : (default = 5.0)
               - gamma : (default = 0.5)
               - reg_alpha : lasso penalty (L1) (default = 0.0)
               - reg_lambda : ridge penalty (L2) (default = 1.0)
               - subsample : subsample fraction (default = 0.9)
               - objective : objective function for ML (default = "binary:logistic" for classification)
               - scale_pos_weight : (default = 1.0)
               - shuffle : shuffle flag for cv (default = True)
               - random_state : (default = 1367
               - saveFigPath: name of the file for saving the figure as PDF format. (default = None))
    
    """
    
    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    from sklearn.preprocessing import scale
    import matplotlib as mpl

    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] = 7


    # Defining the data + scaling
    if isinstance(X_, pd.DataFrame):
        X = scale(X_.values)
    else:
        X = scale(X_)
        
    y = Y
    n_samples, n_features = X.shape

    # Run classifier with cross-validation and plot PR curves
    cv = StratifiedKFold(n_splits = n_folds , shuffle = shuffle , random_state = random_state)
    classifier = XGBClassifier(learning_rate = learning_rate,
                               n_estimators = n_estimators,
                               max_depth = max_depth,
                               min_child_weight = min_child_weight,
                               gamma = gamma,
                               reg_alpha = reg_alpha,
                               reg_lambda = reg_lambda,
                               subsample = subsample,
                               objective = objective,
                               nthread = 4,
                               scale_pos_weight = 1.,
                               base_score = np.mean(y),
                               seed = random_state,
                               random_state = random_state)
    

    # defining the lists
    prs = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(18 , 13))
    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute PR curve and area the curve
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1])
        prs.append(interp(mean_recall, precision, recall))
        pr_auc = auc(recall, precision)
        aucs.append(pr_auc)
        plt.plot(recall, precision, lw=3, alpha=0.5, label='Fold %d (AUCPR = %0.2f)' % (i+1, pr_auc))
        i += 1
    
    plt.plot([0, 1], [1, 0], linestyle='--', lw=3, color='k', label='Luck', alpha=.8)
    mean_precision = np.mean(prs, axis=0)
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    plt.plot(mean_precision, mean_recall, color='navy',
             label=r'Mean (AUCPR = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=4)
    

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall' ,  fontweight = "bold" , fontsize=30)
    plt.ylabel('Precision',fontweight = "bold" , fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend( prop={'size':20} , loc = 0)
    if saveFigPath:
        plt.savefig(F"./{saveFigPath}.pdf" ,bbox_inches='tight')
    plt.show()
    
# --------------------------------------------------------------------------------------

def _binary_classification_metrics(y_true, y_pred_proba, threshold = 0.5, average_method = "weighted", precision_digits = 3):
    """
    Function to display all possible classificaiton metrics.
    NOTE: All thresholds are applied with ">" not ">="
    --------------------------
    Parameters:
               - y_true: list of true values [0, 1]
               - y_pred_proba: list of predicted scores for class = 1 (y_pred_proba[:, 1] for sklearn)
               - threshold: threshold used for mapping y_pred_prob to y_pred (default = 0.5)
               - average_method: a string flag to calculate the metric. can be "micro", "macro", "weighted", "binary" (default = "weighted")
               - precision_digits: the number digits of scores (defualt = 3)
               
   """
    from sklearn.metrics import auc, f1_score, precision_recall_curve, accuracy_score, balanced_accuracy_score, precision_score, recall_score, fbeta_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix

    # checking the type of y_true and y_pred_proba to be numpy array 
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred_proba, np.ndarray):
        y_pred_proba = np.array(y_pred_proba)
        
    # binarizing the y_pred_proba based on the threshold
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # updating average method for roc_auc_score and precision_recall_fscore_support that does handle binary with None
    if average_method == "binary":
        ap_method = None
    else:
        ap_method = average_method

    # calculating the metrics
    accuracy = accuracy_score(y_true = y_true, y_pred = y_pred, normalize = True)
    balanced_accuracy = balanced_accuracy_score(y_true = y_true, y_pred = y_pred, adjusted = False)
    auc_s = roc_auc_score(y_true = y_true, y_score = y_pred_proba, average = ap_method)    
    pr_list, re_list, _ = precision_recall_curve(y_true, y_pred_proba)
    aucpr = auc(re_list, pr_list)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true = y_true, y_pred = y_pred, beta = 1.0, average = ap_method)
    f2 = fbeta_score(y_true = y_true, y_pred = y_pred, beta = 2.0, average= average_method)
    f50 = fbeta_score(y_true = y_true, y_pred = y_pred, beta = 0.50, average= average_method)
    ap = average_precision_score(y_true = y_true, y_score = y_pred_proba, average = ap_method)
    tn, fp, fn, tp = confusion_matrix(y_true = y_true, y_pred = y_pred).ravel()
    
    # calculating threat score
    if average_method == "weighted":
        w = tp + tn
        wp = tp/w
        wn = tn/w
        threat_score = wp * (tp/(tp + fp + fn)) + wn * (tn/(tn + fn + fp))
    elif average_method == "macro":
        threat_score = 0.5 * (tp/(tp + fp + fn)) + 0.5 * (tn/(tn + fn + fp))
    else:
        threat_score = tp/(tp + fp + fn)
    
    # dictionary for building dataframe
    metrics = {"Accuracy" : round(accuracy, precision_digits),
               "Balanced Accuracy" : round(balanced_accuracy, precision_digits),
               "ROC AUC" : round(auc_s, precision_digits),
               "PR AUC" : round(aucpr, precision_digits),
               "F-2 Score" : round(f2, precision_digits),
               "F-0.50 Score" : round(f50, precision_digits),
               "Threat Score" : round(threat_score, precision_digits),
               "Average Precision" : round(ap, precision_digits),
               "TP" : tp,
               "TN" : tn,
               "FP" : fp,
               "FN" : fn
              }
    # updating precision, recall, and f1 accordingly
    if average_method == "binary":
        metrics["Precision"] = round(precision[1], precision_digits)
        metrics["Recall"] = round(recall[1], precision_digits)
        metrics["F-1 Score"] = round(f1[1], precision_digits)

    else:
        metrics["Precision"] = round(precision, precision_digits)
        metrics["Recall"] = round(recall, precision_digits)
        metrics["F-1 Score"] = round(f1, precision_digits)

    df_show = pd.DataFrame(data = metrics, index = [F"""Threshold = {threshold:.3f} | Average = {average_method.title()}"""])
    df_show = df_show.reindex(columns = ["Accuracy","Balanced Accuracy", "ROC AUC", "PR AUC", "Precision", "Recall", \
                                         "Average Precision", "F-1 Score", "F-2 Score", "F-0.50 Score", "Threat Score", \
                                         "TP", "TN", "FP", "FN"])
    return df_show

# --------------------------------------------------------------------------------------

def _binary_classification_metrics_comparison(model, X_, y_true, scale_X=False, average_methods = ["weighted", "binary", "micro", "macro"]):
    """
    Function to find the optimal threshold for binary classification
    NOTE: All thresholds are applied with ">" not ">="
    ----------------------------
    Parameters:
               - model: a trained model object (such as xgboost, glmnet, ...)
               - X_: the test set of features (pandas dataframe or numpy array)
               - y_true: the true class labels (list or array of 0's and 1's)
               - average_methods: a method of average based on sklearn metrics API (default = ["weighted", "binary", "micro", "macro"])
    """
    from sklearn.metrics import auc, f1_score, precision_recall_curve, accuracy_score, balanced_accuracy_score, precision_score, recall_score, fbeta_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
    from IPython.core.display import display, HTML

    # Defining the data + scaling
    if scale_X:
        if isinstance(X_, pd.DataFrame):
            X = scale(X_.values)
        else:
            X = scale(X_)
    else:
        X = X_
        
    class_names = [0, 1]
    
    y_predict_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true, y_predict_proba)
    auc_s = roc_auc_score(y_true, y_predict_proba)
    precision, recall, thresholds2 = precision_recall_curve(y_true, y_predict_proba)
    aucpr = auc(recall, precision)

    # Youden Threshold
    youden_idx = np.argmax(np.abs(tpr - fpr))
    youden_threshold = thresholds[youden_idx]
    y_pred_youden = (y_predict_proba > youden_threshold).astype(int)
    cnf_matrix = confusion_matrix(y_true, y_pred_youden)
    np.set_printoptions(precision=2)
    
    # Sensitivity-Specifity Threshold
    sens_spec_threshold = thresholds[np.argmin(abs(tpr + fpr - 1))]
    y_pred_sens_spec = (y_predict_proba > sens_spec_threshold).astype(int)
    cnf_matrix2 = confusion_matrix(y_true, y_pred_sens_spec)
    
    # precision-recall threshold
    pr_re_index = np.argmin(abs(precision-recall))
    prec_rec_threshold = thresholds2[np.argmin(abs(precision-recall))]
    y_pred_prec_rec = (y_predict_proba > prec_rec_threshold).astype(int)
    cnf_matrix3 = confusion_matrix(y_true, y_pred_prec_rec)
    
    thr_set1 = np.arange(min(thresholds), max(thresholds), 0.01)
    thr_set2 = np.arange(min(thresholds2), max(thresholds2), 0.01)
    f1 = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(len(precision))]
    queue_rate = [(y_predict_proba >= thr).mean() for thr in thr_set2]
    acc =[accuracy_score(y_true, (y_predict_proba >= thr).astype(int)) for thr in thr_set1]

    import matplotlib as mpl
    import seaborn as sns
    sns.set_style("ticks")
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] = 2
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:95% !important; }</style>"))
    import warnings 
    warnings.filterwarnings("ignore")
        
    # plotting
    plt.figure(figsize = (12, 12))
    plt.subplot(2,2,1)
    plt.plot(fpr, tpr, color = "red", label = F"AUC = {auc_s:.3f}")
    plt.plot(fpr[youden_idx], tpr[youden_idx], marker = "o", color = "navy", ms =10 )
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Specificity' , fontsize=12)
    plt.ylabel('Sensitivity' , fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend( prop={'size':12} , loc = 0)
    plt.title("ROC Curve", fontsize = 12)
    plt.annotate(F"Threshold = {youden_threshold:.3f}",
                 xy=(fpr[youden_idx], tpr[youden_idx]), xycoords='data',
                 xytext=(fpr[youden_idx]+0.4, tpr[youden_idx]-0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='right',
                 verticalalignment='bottom')


    plt.subplot(2,2,2)
    plt.plot(thresholds, 1-fpr, label = "Specificity")
    plt.plot(thresholds, tpr, label = "Sensitivity")
    plt.plot(thr_set1, acc, label = "Accuracy")
    plt.xlabel("Threshold", fontsize = 12)
    plt.ylabel("Score", fontsize = 12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(bbox_to_anchor=(1.2, 0.5), loc='center', ncol=1)
    plt.axvline(thresholds[np.argmin(abs(tpr + fpr - 1))], color="k", ls = "--")
    plt.title(F"Threshold = {sens_spec_threshold:.3f}", fontsize = 12)
    plt.title("Pref. Scores vs Threshold", fontsize = 12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    if sens_spec_threshold <= 0.5:
        plt.annotate(F"Threshold = {sens_spec_threshold:.3f}",
                     xy=(sens_spec_threshold, 0.05), xycoords='data',
                     xytext=(sens_spec_threshold + 0.1, 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='left',
                     verticalalignment='bottom')
    else:
        plt.annotate(F"Threshold = {sens_spec_threshold:.3f}",
                     xy=(sens_spec_threshold, 0.05), xycoords='data',
                     xytext=(sens_spec_threshold - 0.1, 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='left',
                     verticalalignment='bottom')

    plt.subplot(2,2,3)
    plt.plot(recall, precision, color = "red", label = F"PR AUC = {aucpr:.3f}")
    plt.plot(recall[pr_re_index], precision[pr_re_index], marker = "o", color = "navy", ms =10 )
    plt.axvline(x = recall[pr_re_index], ymin = recall[pr_re_index], ymax = precision[pr_re_index], color = "navy", ls = "--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall' , fontsize=12)
    plt.ylabel('Precision' , fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend( prop={'size':12} , loc = 0)
    plt.title("Precision-Recall Curve", fontsize = 12)
    plt.annotate(F"Threshold = {prec_rec_threshold:.3f}",
                 xy=(recall[pr_re_index], precision[pr_re_index]), xycoords='data',
                 xytext=(recall[pr_re_index]-0.4, precision[pr_re_index]-0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='left',
                 verticalalignment='bottom')
    
    plt.subplot(2,2,4)
    plt.plot(thresholds2, precision[1:], label = "Precision")
    plt.plot(thresholds2, recall[1:], label = "Recall")
    plt.plot(thresholds2, f1[1:], label = "F1-Score")
    plt.plot(thr_set2, queue_rate, label = "Queue Rate")
    plt.xlabel("Threshold", fontsize = 12)
    plt.ylabel("Score", fontsize = 12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.axvline(thresholds2[np.argmin(abs(precision-recall))], color="k", ls = "--")
    plt.title("Pref. Scores vs Threshold", fontsize = 12)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(bbox_to_anchor=(1.2, 0.5), loc='center', ncol=1)
    if prec_rec_threshold <= 0.5:
        plt.annotate(F"Threshold = {prec_rec_threshold:.3f}",
                     xy=(prec_rec_threshold, 0.03), xycoords='data',
                     xytext=(prec_rec_threshold + 0.1, 0.03),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='left',
                     verticalalignment='bottom')
    else:
        plt.annotate(F"Threshold = {prec_rec_threshold:.3f}",
                     xy=(prec_rec_threshold, 0.03), xycoords='data',
                     xytext=(prec_rec_threshold - 0.1, 0.03),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='left',
                     verticalalignment='bottom')        
    
    plt.show()
    

    thshld_dict = {"Youden" : youden_threshold,
                   "Sensitivity-Specificity" : sens_spec_threshold,
                   "Precision-Recall-F1" : prec_rec_threshold
                  }
    df_frames = []
    for th in thshld_dict:
        for m in average_methods:
            df_frames.append(_binary_classification_metrics(y_true = y_true, y_pred_proba = y_predict_proba, threshold = thshld_dict[th], average_method = m))

    df_to_show = pd.concat(df_frames)
    
    # Set CSS properties
    th_props = [("font-size", "12px"),
                ("text-align", "left"),
                ("font-weight", "bold")]

    td_props = [("font-size", "12px"),
               ("text-align", "center")]

    # Set table styles
    styles = [dict(selector = "th", props = th_props),
              dict(selector = "td", props = td_props)]
    cm = sns.light_palette("blue", as_cmap = True)
    display(df_to_show.style.background_gradient(cmap = cm) \
                            .set_table_styles(styles))            

# --------------------------------------------------------------------------------------
def _clf_train(X_train, y_train, X_test, y_test,
                 learning_rate = 0.05,
                 n_estimators = 100,
                 max_depth = 3,
                 min_child_weight = 5.0,
                 gamma = 1,
                 reg_alpha = 0.0,
                 reg_lambda = 1.0,
                 subsample = 0.9,
                 colsample_bytree = 0.9,
                 objective = "binary:logistic",
                 nthread = 4,
                 scale_pos_weight = 1.0,
                 seed = 1367,
                 random_state = 1367):
    """
    an xgboost model for training
    """

    clf = XGBClassifier(learning_rate = learning_rate,
                        n_estimators = n_estimators,
                        max_depth = max_depth,
                        min_child_weight = min_child_weight,
                        gamma = gamma,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda,
                        subsample = subsample,
                        colsample_bytree = colsample_bytree,
                        objective = objective,
                        nthread = nthread,
                        scale_pos_weight = scale_pos_weight,
                        seed = seed,
                        random_state = random_state)
    
    clf.fit(X_train, y_train, eval_metric = "auc", early_stopping_rounds = 20, verbose = True, eval_set = [(X_test, y_test)])

    return clf

# --------------------------------------------------------------------------------------

def _feature_importance_to_dataframe(bst, importance_type = "total_gain"):
    """
    Function to build a dataframe out of feature importance dictionary of xgboost
    --------------------
    Parameters:
               - bst: xgboost trained model
               - importance_type: can be gain, total_gain,....
    """
    data = {"feature" : [], F"{importance_type}": []}
    cols = []
    importance = []
    features_gain = bst.get_score(importance_type = importance_type)
    for key, val in  features_gain.items():
        data["feature"].append(key)
        data[F"{importance_type}"].append(val)
        
    return pd.DataFrame(data).sort_values(by = F"{importance_type}", ascending = False).reset_index(drop = True)
# --------------------------------------------------------------------------------------

def _plot_shap_importance(model, X_, explainer_type = "tree", max_display = 20, plot_type = "bar"):
    """
    Function to plot SHAP summary plot
    -------------------------
    Parameters:
               - model: trained model (can be tree-based, linear, and deep neural network)
               - X_: features matrix (pandas dataframe or numpy array)
               - explainer_type: shap explainer type (it should match the type of model ["tree", "linear", "deep"])
               - max_display: maximum number of features to display (default = 40)
               - plot_type: ["bar", "dot", "violin"] (default = "bar")
               - color: bar plot color (default = "fuschia")
            
    """
    # scaling features
    scaled_X = scale(X_)
    X = pd.DataFrame(scaled_X, columns = X_.columns.tolist())
        
    import shap
    
    if explainer_type == "tree":
        shap.summary_plot(shap.TreeExplainer(model).shap_values(X), X, plot_type = plot_type, max_display = max_display)
        plt.show()
        
    elif explainer_type == "linear":
        shap.summary_plot(shap.LinearExplainer((model.coef_, model.intercept_), X, feature_dependence = "independent").shap_values(X), X, plot_type = plot_type, max_display = max_display)
        plt.show()
        
    elif explainer_type == "deep":
        shap.summary_plot(shap.DeepExplainer(model).shap_values(X), X, plot_type = plot_type, max_display = max_display)
        plt.show()
        
# --------------------------------------------------------------------------------------       
def _plot_shap_comparison(glmnet_model, xgboost_model, X_,
                          xgb_importance_type = "total_gain",
                          max_display = 20,
                          output = False,
                          figsize = (10,10)):
    """
    Function to compare SHAP values for a linear model versus xgboost model
    ---------------------
    Parameters:
               - glmnet_model: a trained glmnet model
               - xgboost_model: a trained xgboost model
               - X_: features set (pandas datafarame or numpy array)
               - xgb_importance_type: importance type for xgboost (default = "total_gain")
               - max_display: maximum number of features to display
               - output: flag to return the final dataframe
               - figsize: xgboost importance figure size (default = (10,10))

    """
    # scaling features
    scaled_X = scale(X_)
    X = pd.DataFrame(scaled_X, columns = X_.columns.tolist())
    
    # shap values for xgboost
    tree_exp = shap.TreeExplainer(xgboost_model)
    shap_tree_vals = tree_exp.shap_values(X)

    # buidling dataframe for xgboost gain
    xgb_gain_colname = F"""{xgb_importance_type.replace("_", " ").title()}"""
    gain_dict = xgboost_model.get_score(importance_type = xgb_importance_type)
    df_gain = pd.DataFrame(data = gain_dict.items() , columns = ["Feature" , xgb_gain_colname])\
                            .sort_values(by = [xgb_gain_colname], ascending = False).reset_index(drop = True)

    # shap dataframe for xgboost
    df_tree_shap = pd.DataFrame(data = {"Feature" :  X.columns.tolist(),
                                        "Normalized Tree-Shap (%)" : np.mean(np.abs(shap_tree_vals), axis = 0)/max(np.mean(np.abs(shap_tree_vals), axis = 0))})

    # shap values for glmnet
    linear_exp = shap.LinearExplainer((glmnet_model.coef_, glmnet_model.intercept_), X, feature_dependence = "independent")
    shap_linear_vals = np.array(linear_exp.shap_values(X))
    shap_linear_vals_reshaped = shap_linear_vals.reshape(shap_linear_vals.shape[1], shap_linear_vals.shape[2])


    # shap dataframe for glmnet
    df_linear_shap = pd.DataFrame(data = {"Feature" :  X.columns.tolist(),
                                          "Normalized Linear-Shap (%)" : np.mean(np.abs(shap_linear_vals_reshaped), axis = 0)/max(np.mean(np.abs(shap_linear_vals_reshaped), axis = 0))})
    # dataframe for glmnet coeff
    df_glmnet_coeff = pd.DataFrame(data = dict(zip(X.columns.tolist(), np.abs(glmnet_model.coef_.reshape(glmnet_model.coef_.shape[1],)))).items(), columns = ["Feature" , "GLM-Net Coeff"])

    # merging
    df_final = df_tree_shap.merge(df_gain, how = "inner" , on = "Feature") \
                           .merge(df_linear_shap, how = "inner" , on = "Feature") \
                           .merge(df_glmnet_coeff, how = "inner", on = "Feature")

    norm_gain_colname = "Normalized "+xgb_gain_colname+" (%)"
    df_final[norm_gain_colname] = df_final[xgb_gain_colname]/df_final[xgb_gain_colname].max() 

    df_to_show = df_final.loc[:, ["Feature", "GLM-Net Coeff", norm_gain_colname, "Normalized Tree-Shap (%)", "Normalized Linear-Shap (%)"]] \
                         .sort_values(by = ["GLM-Net Coeff",norm_gain_colname, "Normalized Tree-Shap (%)", "Normalized Linear-Shap (%)" ], ascending = False).reset_index(drop = True)
    
    print(F"-*-*-*-*-*-*-*-*-*-* Linear SHAP -*-*-*-*-*-*-*-*-*-*")
    _plot_shap_importance(glmnet_model, X, explainer_type = "linear", max_display = max_display, plot_type = "bar")
    print(F"-*-*-*-*-*-*-*-*-*-* Tree SHAP -*-*-*-*-*-*-*-*-*-*")
    _plot_shap_importance(bst_xgb, X, explainer_type = "tree", max_display = max_display, plot_type = "bar")
    print(F"-*-*-*-*-*-*-*-*-*-* XGBoost Importance -*-*-*-*-*-*-*-*-*-*")
    _plot_xgboost_importance(bst_xgb, importance_type=xgb_importance_type, figsize = figsize, color = "#008BFB", max_num_features = max_display)
    # Set CSS properties
    th_props = [("font-size", "12px"),
                ("text-align", "center"),
                ("font-weight", "bold")]

    td_props = [("font-size", "12px")]

    # Set table styles
    styles = [dict(selector = "th", props = th_props),
              dict(selector = "td", props = td_props)]
    cm = sns.light_palette("blue", as_cmap = True)
    display(df_to_show.head(max_display).style.background_gradient(cmap = cm, subset = ["GLM-Net Coeff", norm_gain_colname, "Normalized Tree-Shap (%)", "Normalized Linear-Shap (%)"]) \
                            .highlight_max(subset=["GLM-Net Coeff", norm_gain_colname, "Normalized Tree-Shap (%)", "Normalized Linear-Shap (%)"], color = "fuchsia") \
                            .format({"GLM-Net Coeff" : "{:.3}",
                                     norm_gain_colname : "{:.3%}",
                                     "Normalized Tree-Shap (%)" : "{:.3%}",
                                     "Normalized Linear-Shap (%)" : "{:.3%}"
                                    }) \
                             .set_table_styles(styles))

    if output:
        return df_to_show
    
# --------------------------------------------------------------------------------------  
def _sparsity_level(df, fillna = 0.):
    """
    Function to calculate the sparsity level of a matrix in percent.
    -------------------
    Parameters:
               - df: matrix
               - fillna: value to fill the null values (default = 0.)
    """
    if isinstance(df, pd.DataFrame):
        return round((1. - np.count_nonzero(df.fillna(value = fillna).values)/df.size) * 100, 2)

    elif isinstance(df, np.ndarray):
        return round((1. - np.count_nonzero(np.nan_to_num(df, nan = fillna))/df.size) * 100, 2)
    
    else:
        raise ValueError("Bad input data type! The input data type should either be Pandas DataFrame or Numpy ndarray")
        
# -------------------------------------------------------------------------------------- 
def _plot_calibration_curve(model, X, y_true, normalize = False, n_bins = 10, strategy = "uniform", grid = False, n_folds = 10, calibration_method = "sigmoid", return_model = False):
    """
    Function to plot the calibaration curve for a trained model
    ----------------------
    Parameters:
               - model: Trained model (i.e. glmnet or xgboost)
               - X: Feature Set
               - y_true: Ground truth class labels
               - normalize:  Whether y_prob needs to be normalized in [0, 1] (default = False)
               - n_bins: Number of bins (default = 10)
               - strategy: The method to define the width of bins. Options are "uniform" or "quantile" (default = "uniform")
               - grid: Grid for plot (default = False)
               - n_folds: Number of folds for cross-validation in calibration (default = 10)
               - calibration_method: Method of calibration. Options are "sigmoid" and "isotonic" (default = "sigmoid")
               - return_model: A flag to return the calibrated model (default = False)
    
    """
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    import matplotlib as mpl
    import seaborn as sns
    sns.set_style("ticks")
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] = 2
    
    clf = CalibratedClassifierCV(base_estimator = model, cv = n_folds, method = calibration_method)
    clf.fit(X, y_true)
    y_pred_pos = clf.predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true = y_true, y_prob = y_pred_pos, n_bins = n_bins, normalize = normalize, strategy = strategy)
    
    plt.figure(figsize = (8, 8))
    plt.subplot(2,1,1)
    plt.plot(mean_predicted_value, fraction_of_positives, marker = "o", color = "navy", ms =10, label= F"""Brier Loss = {brier_score_loss(y_true, y_pred_pos, pos_label= 1):.3f}""")
    plt.plot([0, 1], [0, 1], color = "k", ls = "--", label = "Ideal Calibration")
    plt.ylabel("Fraction of Positives", fontsize = 12)
    plt.title("Calibration Curve", fontsize = 12)
    plt.legend(prop={'size':12} , loc = 0)
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(grid)
    
    plt.subplot(2,1,2)
    plt.hist(y_pred_pos, range = (0, 1), bins = n_bins, histtype = "stepfilled", lw=2, color = "cyan", ec = "navy")
    plt.xlabel("Mean Predicted Value", fontsize = 12)
    plt.ylabel("Frequency", fontsize = 12)
    plt.grid(grid)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()
    
    if return_model:
        return clf
# --------------------------------------------------------------------------------------     
def _train_test_split(X, y, test_size=0.20, shuffle=True, scale = True, with_mean=True, with_std=True, random_state=1367):
    """
    Function to build stratified train/test sets with scale flag
    ---------------------
    Parameters:
               - X: Feature set in pandas dataframe or numpy array
               - y: Targets in list of values
               - test_size: Test size in float (deafult = 0.20)
               - shuffle: Shuffle flag (default = True)
               - scale: Scale flag to fit on train set and transform it on the test set (default = True)
               - with_mean: Flag to remove the mean from the feature to have 0 mean (default = True)
               - with_std: Scale the feature to have unit variance (default = True)
               - random_state: Random state (default = 1367)
    """
    if scale == True:
        if isinstance(X, pd.DataFrame):
            cols = X.columns.tolist()
            Xtr, Xte, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=y)
            _scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
            SXtr = _scaler.fit_transform(Xtr)
            SXte = _scaler.transform(Xte)
            X_train = pd.DataFrame(data=SXtr, columns=cols)
            X_test = pd.DataFrame(data=SXte, columns=cols)
            return X_train, X_test, y_train, y_test
        else:
            Xtr, Xte, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=y)
            _scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
            X_train = _scaler.fit_transform(Xtr)
            X_test = _scaler.transform(Xte)
            return X_train, X_test, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=y)
        return X_train, X_test, y_train, y_test
    
# --------------------------------------------------------------------------------------             
def _cdS(x, y, gridsize=512, adjust=1, bw="scott", cut=0, clip=None, kernel="gau",
        hist=True, hist_bins=20, hist_frac=0.75, hist_color='red', verbose=False):
    """Univariate conditional density plot for binary outcome
    
    Parameters
    __________
    x: 1d array-like
        continuous variable
    y: 1d array-like
        binary outcome
    gridsize : int, optional
        Number of discrete points in the evaluation grid.
    adjust: float
        An adjustment factor for the bw. Bandwidth becomes bw * adjust.
    bw : {'scott' | 'silverman' | scalar | pair of scalars }, optional
        Name of reference method to determine kernel size, scalar factor.
    cut : scalar, optional
        Draw the estimate to cut * bw from the extreme data points.
    clip : pair of scalars, or pair of pair of scalars, optional
        Lower and upper bounds for datapoints used to fit KDE.
    kernel : {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }, optional
        Code for shape of kernel to fit with.
    """
    #remove missing
    x = x.copy()*1.
    y = y.copy()*1.
    if verbose: print(stats.describe(x))
    sel = ~(np.isnan(x) | np.isnan(y))
    x = x[sel]
    y = y[sel]
    if verbose: print(stats.describe(x))
    
    if clip is None:
        clip = (-np.inf, np.inf)
        #clip = (np.min(x), np.max(x))
    
    fft = (kernel == "gau")
    #fft = False
    kde = smnp.KDEUnivariate(x)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip, adjust=adjust)
    xd, yd = kde.support, kde.density
    if verbose: print(stats.describe(xd))
    
    kde1 = smnp.KDEUnivariate(x[y==y[1]])
    kde1.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip, adjust=adjust)
    #xd1, yd1 = kde1.support, kde1.density
    #yd1i = np.interp(xd, xd1, yd1)
    yd1 = np.array([kde1.evaluate(x) for x in xd]).flatten()
    
    yvals, ycounts = np.unique(y, return_counts=True)
    yprop = np.cumsum(ycounts)/np.sum(ycounts)
    #print((yvals, ycounts, yprop))   
    yprob = yd1/yd*yprop[0]
    
    sns.set_style("whitegrid")
    
    bot = np.maximum(np.minimum(yprob, 1), 0)
    top = 1 - bot
    pal = sns.color_palette("Greys",n_colors=2).as_hex()
    pal.reverse()
    ax = plt.stackplot(xd, bot, top, edgecolor='black', colors=pal, alpha=0.5)
    #plt.legend(loc='upper right')
    #plt.show()
    #sns.kdeplot(x)
    
    if hist:
        #plt.hist(x, density=True, bins=hist_bins, histtype='step', color='green')
        hst, edges = np.histogram(x, bins=hist_bins, range=(np.min(xd),np.max(xd)))
        weights = [hist_frac / np.max(hst)]*len(x)
        plt.hist(x, bins=edges, weights=weights, histtype="step", color=hist_color)

    return xd, yd, yd1, yprob, plt
# --------------------------------------------------------------------------------------     

def _cond_plot(df, x, y=["re", "re6m"], agg_func='mean',
              color=None, ls="-", lw=1,
              show_shape=True, shape_equal_bins=True, shape_bin_width=None, 
              shape_ls="steps-mid", shape_lw=3, shape_alpha=0.25, shape_color='k',
              shape_factor=0.8, shape_shift=None,
              impute=True, miss_val=None, miss_marker='s', miss_markersize=25,
              logodds=False, logodds_ylim=(-10,10),
              xlabel=None, ylabel="probability", xlim=None, ylim=None, ax=None,
              verbose=False):
    """Plot the conditional aggregated outcome(s) vs discrete values of `x`
    
    E.g., when the outcome is binary 0-1 outcome, it's a conditional outcome probability plot
    WRT discrete values of x.
    
    Parameters
    ----------
    df : a DataFrame
    x : name of a column in df with discrete values to condition the outcome probability on
    y : name (or a list of names) of the outcome column (or of several outcomes)
    agg_func : aggregation function for the outcome
    color : color of the conditional outcome lines
    ls : linestyle of the conditional outcome lines
    ls : linewidth of the conditional outcome lines
    show_shape : whether to plot the shape of the distribution of x's discrete values
    shape_equal_bins : whether to assume that the discrete values of x are representing
        equal width bins. In this case, the distribution shape is assigned to be zero in the bins
        where x has no values.
    shape_bin_width : this parameter allows to manually set the bin width when the automatic bin width
        computation fails when assume_equal_bins=True
    shape_ls : linestyle of the distribution shape
    shape_lw : linewidth of the distribution shape
    shape_alpha : alpha transparency of the distribution shape
    shape_color : color of the distribution shape
    shape_factor : the factor of the relative size of the shape distribution
        WRT the max of conditional outcome values
    shape_shift : the amount to move the shape distribution vertically WRT zero.
        When it is None and logodds=True, the shift is automatically set to a minimal y logodds value.
    impute : whether missing x values should be imputed, with their x-location displayed as markers
    miss_val : numeric or string x value to "impute" any missing x values, or a function taking
        a single 1D numpy array as an input. When None, the missing values are shown on the left side.
        The special string values are 'min','max','mean','median' which call the corresponding pd.Series
        methods on df[x].
    miss_marker : the marker shape to represent the location of missing values
    miss_markersize : the missing values marker size
    logodds : whether instead of conditional probability, conditional logodds 
        of a binary outcome should be displayed on y axis
    logodds_ylim : the minimum and maximum logodds to draw when probability is either 0 or 1
    xlabel : the label to show on the x axis
    ylabel : the label to show on the y axis
    ylim : a tuple of the limits for the x axis
    ylim : a tuple of the limits for the y axis
    verbose : turn on debugging printouts
    """
    import re
    
    if logodds and ylabel=='probability':
        ylabel = 'logodds'
    if type(y) != list:
        y = [y]
    
    df = df[[x] + y].copy()
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x])
    if pd.api.types.is_bool_dtype(df[x]):
        df[x] = df[x].astype("float")
    
    if impute and not df[x].isnull().any():
        impute = False
    
    if impute:
        if miss_val is None:
            if x_is_numeric:
                shift = np.nanmin(np.diff(np.sort(df[x].unique())))
                if shift == 0:
                    shift = 0.05 * (df[x].max() - df[x].min())
                miss_val = df[x].min() - shift
            else:
                miss_val = " NA"
        if callable(miss_val):
            miss_val = miss_val(df[x].values)
        elif miss_val=='mean':
            miss_val = df[x].mean()
        elif miss_val=='median':
            miss_val = df[x].median()
        elif miss_val=='min':
            miss_val = df[x].min()
        elif miss_val=='max':
            miss_val = df[x].max()
        df[x] = df[x].fillna(miss_val)
        if verbose:
            print(f"[miss_val] = [{miss_val}]")


    #agg_dict = {**dict(zip(y, ["mean"]*len(y))), **{y[0]:"size"}}
    agg_dict = [agg_func, 'count']
    tmp = df[[x] + y].groupby(x).agg(agg_dict).reset_index()
    tmp.columns = [re.sub(r"_mean$", "", '_'.join(col).rstrip('_')) for col in tmp.columns.values]
    tmp.rename(columns={f'{y[0]}_count':'__count__'}, inplace = True)
    if verbose:
        print(tmp)
        print(tmp.info())
    
    if logodds:
        for v in y:
            p = tmp[v]
            p_ylim = (1./(1 + np.exp(-logodds_ylim[0])),
                      1./(1 + np.exp(-logodds_ylim[1])))
            tmp[v] = np.select([(p > p_ylim[0]) & (p < p_ylim[1]),
                                (p >= 0.) & (p < p_ylim[0]),
                                (p <= 1.) & (p > p_ylim[1])],
                               [np.log(np.divide(p.values,
                                                 (1. - p.values),
                                                 where=(p > p_ylim[0]) & (p < p_ylim[1]))),
                                logodds_ylim[0],
                                logodds_ylim[1]],
                               np.NaN)
        if verbose: print(tmp)
    
    if x_is_numeric and xlim is None:
        min_x, max_x = (tmp[x].min(), tmp[x].max())
        nudge = 0.03 * (max_x - min_x)
        xlim = [min_x - nudge, max_x + nudge]

    min_y = np.min([tmp[r].min() for r in y])
    max_y = np.max([tmp[r].max() for r in y])
    if ylim is None:
        nudge = 0.03 * (max_y - min_y)
        ylim = [min_y - nudge, max_y + nudge]
        if min_y >= 0 and min_y <= 1:
            ylim[0] = 0 - nudge

    ax = tmp.plot(x, y, ls=ls, lw=lw, color=color, ax=ax)
    
    if show_shape:
        # calculate the distribution shape
        nmax_shape = tmp['__count__'].max()
        if shape_shift is None:
            shape_shift = 0.
            if logodds:
                shape_shift = min_y
        shape_scale = shape_factor / nmax_shape * (max_y - shape_shift)
        tmp = tmp.assign(distr_shape = lambda x: x['__count__'] * shape_scale + shape_shift)
        
        if verbose:
            print(tmp)
            print((xlim, ylim, min_y, max_y))
            print((nmax_shape, shape_scale, shape_shift))
        
        # when the assumption is that the discrete bins are of equal width, 
        # infer the bin width, and fill the empty inner bins with zeros for the shape
        if shape_equal_bins and x_is_numeric:
            if shape_bin_width is None:
                shape_bin_width = np.min(np.diff(tmp[x]))
            if shape_bin_width is None or shape_bin_width == 0:
                print(f"{x} has invalid range")
                return None
            tmp = tmp.assign(__x_idx = lambda c: np.round((c[x] - np.min(tmp[x]))/shape_bin_width))
            n_x_values = int(np.round(tmp[x].values.ptp() / shape_bin_width))
            tmp_fill = pd.DataFrame({"__x_idx":range(n_x_values + 1)})
            tmp = pd.merge(tmp_fill, tmp, 'left', on="__x_idx")
            tmp["distr_shape"] = np.where(tmp[x].isnull(),
                                          shape_shift,
                                          tmp["distr_shape"])
            tmp[x] = np.where(tmp[x].isnull(),
                              tmp['__x_idx'] * shape_bin_width + np.min(tmp[x]),
                              tmp[x])
        
        if verbose: print(tmp)
        tmp.plot(x, "distr_shape", alpha=shape_alpha,
                 lw=shape_lw, ls=shape_ls, color=shape_color, ax=ax)
        if logodds:
            x1, x2 = plt.xlim()
            ax.hlines(min_y, x1, x2, color=ax.get_lines()[-1].get_color(),
                      alpha=shape_alpha, linestyles="dashed")
    
    if impute:
        miss_colors = [ln.get_color() for ln in ax.get_lines()]
        miss_iloc = pd.Index(tmp[x]).get_loc(miss_val)
        for i in range(len(y)):
            miss_yval = tmp[y[i]].iloc[miss_iloc]
            ax.scatter(miss_iloc, miss_yval, marker=miss_marker,# clip_on=False,
                       color=miss_colors[i], s=miss_markersize)
        if show_shape:
            miss_yval = tmp["distr_shape"].iloc[miss_iloc]
            ax.scatter(miss_iloc, miss_yval, marker=miss_marker,# clip_on=False,
                       color=miss_colors[i+1], alpha=shape_alpha, s=miss_markersize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fancybox=True, framealpha=0.3)
    return ax

# --------------------------------------------------------------------------------------   
def _glmnet_validation(X, Y, alpha=0.5, n_splits=10, alpha_ini=0.0, penalty_gamma=2.0, random_state=1367, shuffle=True, precision_digits=3, with_mean=False, with_std=False, figsize=(8, 8), color="slateblue", output_features=False):
    """
    Function to validate adaptive glmnet models with nfolds internal/external cross-validation
    ---------------------
    Parameters:
               - X: Features matrix in Pandas DataFrame format
               - Y: Targets in numpy array/list format
               - alpha: Stability Parameter (default=0.5)
               - n_splits: Number of folds for cross-validation (default=10)
               - alpha_ini : Stability parameter initial point (default = 0.0)
               - penalty_gamma : Gamma penalty value which can be lower for more features to keep (default = 2.0)
               - random_state: Random state seed (default = 1367)
               - shuffle: Flag to shuffle the data
               - precision_digits: Number of floating point digits (default=3)
               - figsize: Figure size (default=(8,8))
               - color: Color for bar chart (default="cyan")
               - output_features: Flag to output features frequency across nfolds cross validation (default=False)
    """
    
    
    X.reset_index(drop=True, inplace=True)
    # main loop
    fold = 1
    frames = []
    pruned_features = []
    cols = X.columns.tolist()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for train_index, test_index in cv.split(X, Y):
        _scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        X_train = pd.DataFrame(data = _scaler.fit_transform(X.values[train_index]), columns = cols)
        X_test = pd.DataFrame(data = _scaler.transform(X.values[test_index]), columns = cols)
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        model = _A_glmnet(X_train, Y_train, alpha=alpha, n_splits=n_splits, alpha_ini=alpha_ini, penalty_gamma=penalty_gamma, random_state=random_state, scale_X=False)
        auc_test = roc_auc_score(y_true = Y_test, y_score = model.predict_proba(X_test)[:, 1]) 
        
        
        performance = {"Train-Mean-AUC" : round(np.mean(model.cv_mean_score_), precision_digits),
                       "Train-Std-AUC" : round(np.std(model.cv_mean_score_), precision_digits),
                       "Train-Median-AUC" : round(np.median(model.cv_mean_score_), precision_digits),
                       "Test-AUC" : round(auc_test, precision_digits),
                       "Nonzero Coeff." : np.count_nonzero(model.coef_)
                      }
        df_show = pd.DataFrame(data = performance, index = [F"""Fold = {fold}"""])
        frames.append(df_show)
        for col in _df_glmnet_coeff_path(model, X_train)["Features"].values.tolist():
            pruned_features.append(col)
        fold += 1
        del train_index, test_index, _scaler, X_train, X_test, Y_train, Y_test, model, auc_test, performance, df_show
    
    
    df_to_show = pd.concat(frames)
    # Set CSS properties
    th_props = [("font-size", "12px"),
                ("text-align", "center"),
                ("font-weight", "bold")]

    td_props = [("font-size", "12px")]

    # Set table styles
    styles = [dict(selector = "th", props = th_props),
              dict(selector = "td", props = td_props)]
    cm = sns.light_palette("blue", as_cmap = True)
    display(df_to_show.style.background_gradient(cmap = cm) \
                            .highlight_max(color = "fuchsia") \
                             .set_table_styles(styles))    
    unique_elements, counts_elements = np.unique(pruned_features, return_counts=True)
    counts_elements = [float(i) for i in list(counts_elements)]
    df_features = pd.DataFrame(data = {"Feature" : list(unique_elements) , "Count" : counts_elements})
    df_features.sort_values(by = ["Count"], ascending = False, inplace = True)
    # Plotting
    import matplotlib as mpl
    mpl.rcParams["axes.linewidth"] = 3 
    mpl.rcParams["lines.linewidth"] = 7
    plt.figure()
    df_features.sort_values(by = ["Count"]).plot.barh(x ="Feature", y="Count", color=color, figsize=figsize)
    plt.show()
    
    if output_features==True:
        return df_features
    
# --------------------------------------------------------------------------------------   
        
def _plot_threat_score_vs_threshold(model, X, y_true, num_thresholds=100, output=False):
    """
    Function to plot threat score vs different threshold at return the maximum
    ----------------------------
    Parameters:
               -model: Trained model
               -X: Feature Set
               -y_true: targets
               -num_thresholds: Number of points as thresholds (default=100)
               -output: Flag to return the best threshold to maximize the threat score
    """
    def __calculate_threat_score(tp, fp, fn):
        """
        Helper function to calculate threat score based on confusion matrix in each iteration
        """
        return tp/(tp+fp+fn)
    
    # main loop
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    ts = []
    for th in thresholds:
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true = y_true, y_pred = y_pred).ravel()
        ts.append(__calculate_threat_score(tp, fp, fn))
    
    # plotting    
    import matplotlib as mpl
    import seaborn as sns
    sns.set_style("ticks")
    mpl.rcParams['axes.linewidth'] = 3 
    mpl.rcParams['lines.linewidth'] = 2
    
    plt.figure(figsize=(10,6))
    plt.plot(thresholds, ts, ls="-", marker="o", ms=8, color="cyan")
    
    plt.axvline(thresholds[np.argmax(ts)], color="navy", ls = "--", label=F"Threshold = {thresholds[np.argmax(ts)]:.3f}\nThreat Score = {ts[np.argmax(ts)]:.3f}")
    plt.legend(prop={'size':12} , loc = 0)
    plt.xlim([-0.01, 1.01])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel("Threshold", fontsize = 15)
    plt.ylabel("Threat Score", fontsize = 15)
    plt.show()
    
    if output:
        return thresholds[np.argmax(ts)]
    
# -------------------------------------------------------------------------------------- 

def _iqr_clipper(df, low=0.25, high=0.75, factor=1.5, exclude=None):
    """
    Function to clip continuous features based on their interquantile range
    ------------------------------
    Parameters:
              -df: Pandas DataFrame
              -low: Lower bound of IQR (default=0.25)
              -high: Upper bound of IQR (default=0.75)
              -factor: IQR factor (default=1.5)
              -exclude: List of columns (i.e. ["col1", "col2", "col3"]) to be excluded from IQR clipping (default=None)
    ------------------------------    
    IQR Algorithm:
    Q1 = 0.25 Quantile
    Q3 = 0.75 Quantile
    IQR = Q3 - Q1
    Low = Q1 - 1.5 * IQR
    High = Q3 + 1.5 * IQR
    """
    
    # finding continouos/binary columns
    bin_cols = _binary_col_finder(df)
    cont_cols = _continuous_col_finder(df)
    
    # checkpoint for any exclusion
    if exclude == None:
        df_cont = df.copy().loc[:, cont_cols]
        df_bin = df.copy().loc[:, bin_cols]
    else:
        if isinstance(exclude, list):
            ex_cont_cols = [c for c in cont_cols if c not in exclude]
            df_ex = df.copy().loc[:, exclude]
            df_cont = df.copy().loc[:, ex_cont_cols]
            df_bin = df.copy().loc[:, bin_cols]
        else:
            lst = []
            lst.append(exclude)
            ex_cont_cols = [c for c in cont_cols if c not in lst]
            df_ex = df.copy().loc[:, lst]
            df_cont = df.copy().loc[:, ex_cont_cols]
            df_bin = df.copy().loc[:, bin_cols]
        
    # main loop for IQR
    for col in df_cont.columns.tolist():
        q1 = df_cont[col].quantile(low)
        q3 = df_cont[col].quantile(high)
        iqr = q3 - q1 
        low_range  = q1 - factor * iqr
        high_range = q3 + factor * iqr
        df_cont[col].clip(lower=low_range, upper=high_range, inplace=True)
        
    if exclude == None:
        return pd.concat([df_bin, df_cont], axis = 1)
    else:
        return pd.concat([df_bin, df_ex, df_cont], axis = 1)
# -------------------------------------------------------------------------------------- 

def _percentiles_clipper(df, low=0.05, high=0.95, exclude=None):
    """
    Function to clip continuous features based on [low, high] percentiles
    ------------------------------
    Parameters:
              -df: Pandas DataFrame
              -low: nth percentile as the low range (default=0.05)
              -high: nth percentile as the high range (default=0.95)
              -exclude: List of columns (i.e. ["col1", "col2", "col3"]) to be excluded from IQR clipping (default=None)
    """
    
    # finding continouos/binary columns
    bin_cols = _binary_col_finder(df)
    cont_cols = _continuous_col_finder(df)

    # checkpoint for any exclusion
    if exclude == None:
        df_cont = df.copy().loc[:, cont_cols]
        df_bin = df.copy().loc[:, bin_cols]
    else:
        if isinstance(exclude, list):
            ex_cont_cols = [c for c in cont_cols if c not in exclude]
            df_ex = df.copy().loc[:, exclude]
            df_cont = df.copy().loc[:, ex_cont_cols]
            df_bin = df.copy().loc[:, bin_cols]
        else:
            lst = []
            lst.append(exclude)
            ex_cont_cols = [c for c in cont_cols if c not in lst]
            df_ex = df.copy().loc[:, lst]
            df_cont = df.copy().loc[:, ex_cont_cols]
            df_bin = df.copy().loc[:, bin_cols]
        
    # main loop for [low, high] percentiles
    df_quant = df_cont.quantile([low, high])
    for col in df_cont.columns.tolist():
        low_range = df_cont[col].quantile(low)
        high_range = df_cont[col].quantile(high)
        df_cont[col].clip(lower=low_range, upper=high_range, inplace=True)        
        
    if exclude == None:
        return pd.concat([df_bin, df_cont], axis = 1)
    else:
        return pd.concat([df_bin, df_ex, df_cont], axis = 1)
# --------------------------------------------------------------------------------------

def _clipping_dict(df):
    """
    Function to return a dictionary of low/high range of clipping
    Can be used after applying any clipping function including _iqr_clipper or _percentiles_clipper
    """
    cont_cols = _continuous_col_finder(df)
    clipping_range = dict()
    
    df_min = df.loc[:, cont_cols].min()
    df_max = df.loc[:, cont_cols].max()
    for col in cont_cols:
        clipping_range[col] = [df_min[col], df_max[col]]
    
    return clipping_range
# --------------------------------------------------------------------------------------
    
def _join_dictionaries(dictionary1, dictionary2):
    '''
    Takes two dictionaries and combines values and keys into a single dictionary (outputed values may not be distinct,
    if same key in both dictionaries have the same value).
    
    Required Parameter(s): dictionary1 (dictionary), dictionary2 (dictionary)
    Optional Parameter(s): None
    '''
    dictionary={}
    d1Keys = list(dictionary1.keys())
    d2Keys = list(dictionary2.keys())
    combinedKeys = list(set(d1Keys + d2Keys))

 

    for key in combinedKeys:
        d1_vals = []
        d2_vals = []
        if key in d1Keys:
            d1_vals = dictionary1[key]
            if isinstance(d1_vals, (int, float, str)):
                d1_vals = [d1_vals]
            
        if key in d2Keys:
            d2_vals = dictionary2[key]
            if isinstance(d2_vals, (int, float, str)):
                d2_vals = [d2_vals]
        
        dictionary[key] = list(set(d1_vals + d2_vals))
    return dictionary
# --------------------------------------------------------------------------------------
def _slick_glmnet(X_, y, n_iter=2, alpha=0.1, n_splits=10, alpha_ini=0.0, penalty_gamma=2, seed=1367, scale_X=True):
    """
    Function to use adaptive glmnet for feature selection.
    ---------------------------------
    Parameters:
               - X_ : features set which can be pandas dataframe and numpy array
               - Y : target/response values
               - n_iter: number of iterations (default=2)
               - alpha : stability parameter, 0.0 for ridge and 1.0 for lasso (default = 0.1)
               - n_splits : number of folds CV for finding penalty parameters lambda (default = 10)
               - alpha_ini : (default = 0.0)
               - penalty_gamma : gamma penalty value which can be lower for more features to keep (default = 2.0)
               - seed: Random seed (default = 1367)
               - scale_X: flag for scaling the feautures (default = True)
              
    """     
    
    
    if scale_X == True:
        # Defining the data + scaling
        if isinstance(X_, pd.DataFrame):
            X = pd.DataFrame(scale(X_), columns = X_.columns.tolist())
        else:
            X = scale(X_)
    else:
        X = X_
    
    for iteration in range(n_iter):
        # update random state 
        random_state = seed * iteration
        X_permuted = _permute(df = X, random_state = random_state)
        model = _A_glmnet(X_permuted, y, alpha=alpha, n_splits=n_splits, alpha_ini=alpha_ini, penalty_gamma=penalty_gamma, random_state=random_state, scale_X=scale_X)
        df_coeff = _df_glmnet_coeff_path(model, X_permuted)
        display(df_coeff)
        pruned_features = [col for col in df_coeff.Features.values if "noisy" not in col]
        print(F"Iteration {iteration+1}: {len(pruned_features)} Pruned Features Were Selected!")
        X = X.loc[:, pruned_features]
        del df_coeff, model, X_permuted
        gc.collect()
        
    return pruned_features