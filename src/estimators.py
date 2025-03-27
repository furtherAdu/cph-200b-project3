from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np
from operator import itemgetter

base_regressor = GradientBoostingRegressor
base_classifier = GradientBoostingClassifier

def propensity_estimator(X, T, multiclass=False): # propensity scores
    # this now outputs scores for each probability class
    propensity_model = base_classifier(random_state=40)
    pi = propensity_model.fit(X, T).predict_proba(X)
    score = propensity_model.score(X, T)
    pi = pi if multiclass else pi[:, 1]
    return {'pi': pi, 'score': score}

def unadjusted_DM_estimator(data, treatment_var, outcome_var, **kwargs):
    # changed this to give us multi-class difference in means

    # get avg treatement effect for each treatment
    tx_classes = np.sort(data[treatment_var].unique()).astype(int)
    estimates = {k: data.loc[data[treatment_var] == k, outcome_var].mean()
                for k in tx_classes}

    # get difference in means between each tx group
    results = {}
    for i in range(len(tx_classes)):
        for j in range(i + 1, len(tx_classes)):
            diff = estimates[tx_classes[j]] - estimates[tx_classes[i]]
            results[f"{tx_classes[j]} vs {tx_classes[i]}"] = {
                'tau': diff,
                "variance": float('nan'),
                "propensity_model_score": float('nan'),
                "outcome_model_score": float('nan')
            }

    return results

def ipw_estimator(data, treatment_var, outcome_var, covariates, **kwargs):
    # modified to return dict of results with tau and var for each treatment group difference
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]
    
    tx_classes = np.sort(np.unique(T)).astype(int)
    multiclass = len(tx_classes) > 2
    pi, propensity_model_score = itemgetter('pi', 'score')(propensity_estimator(X,T, multiclass=multiclass))

    # get weighted values for each person and get mean
    outcomes = {}
    for k in tx_classes:
        treated_pts = (T==k).astype(float)
        weights = treated_pts / pi[:,k]
        weightedYs = (weights * Y)
        outcomes[k] = {
            'mean': weightedYs.mean(),
            'values': weightedYs
        }

    # pairwise ates for each tx group
    results = {}
    for i in range(len(tx_classes)):
        for j in range(i + 1, len(tx_classes)):
            tau_diff = outcomes[tx_classes[j]]['mean'] - outcomes[tx_classes[i]]['mean']
            var_diff = (outcomes[tx_classes[j]]['values'] - outcomes[tx_classes[i]]['values']).var()
            results[f"{tx_classes[j]} vs {tx_classes[i]}"] = {
                "tau": tau_diff,
                "variance": var_diff,
                "propensity_model_score": propensity_model_score,
                "outcome_model_score": float('nan')
            }

    return results

def t_learner(data, treatment_var, outcome_var, covariates, continuous_outcome=False, **kwargs):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]
    outcome_model_class = base_regressor if continuous_outcome else base_classifier

    tx_classes = np.sort(np.unique(T)).astype(int)

    # train one model on each tx group and store it
    models = {}
    outcome_model_scores = {}
    for k in tx_classes:
        model = outcome_model_class(random_state=40)
        model.fit(X[T==k], Y[T==k])
        outcome_model_scores[k] = model.score(X[T==k], Y[T==k])
        models[k]=model
    
    # predict outcome for all patients using each model
    mu={}
    for k in tx_classes:
        mu[k] = model.predict(X) if continuous_outcome else model.predict_proba(X)[:,1]

    # get ates 
    results={}
    for i in range(len(tx_classes)):
        for j in range(i + 1, len(tx_classes)):
            tau_diff = mu[tx_classes[j]] - mu[tx_classes[i]]
            results[f"{tx_classes[j]} vs {tx_classes[i]}"] = {
                "tau": tau_diff.mean(),
                "variance": tau_diff.var(),
                "propensity_model_score": float('nan'),
                "outcome_model_score": [outcome_model_scores],
            }

    return results

def s_learner(data, treatment_var, outcome_var, covariates, continuous_outcome=False, **kwargs):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    XT = data[[*covariates, treatment_var]]
    T = data[treatment_var]
    Y = data[outcome_var]
    outcome_model_class = base_regressor if continuous_outcome else base_classifier

    tx_classes = np.sort(np.unique(T)).astype(int)
    
    model = outcome_model_class(random_state=40)
    model.fit(XT, Y)
    outcome_model_score = model.score(XT, Y)

    mu = {}
    # get counterfactual predictions from outcome model for all 
    # patients over each treatment class
    for k in tx_classes:
        X_k = XT.copy()
        X_k[treatment_var] = k
        mu[k] = model.predict(X_k) if continuous_outcome else model.predict_proba(X_k)[:,1]

    # calculate pairwise ates for each tx group
    results = {}
    for i in range(len(tx_classes)):
        for j in range(i + 1, len(tx_classes)):
            mu_diff = mu[tx_classes[j]] - mu[tx_classes[i]]
            results[f"{tx_classes[j]} vs {tx_classes[i]}"] = {
                "tau": mu_diff.mean(),
                "variance": mu_diff.var(),
                "propensity_model_score": float('nan'),
                "outcome_model_score": outcome_model_score,
            }
    return results

def x_learner(data, treatment_var, outcome_var, covariates, continuous_outcome=False, **kwargs):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]
    outcome_model_class = base_regressor if continuous_outcome else base_classifier

    tx_classes = np.sort(np.unique(T)).astype(int)
    n_txclasses = len(tx_classes)
    multiclass = n_txclasses > 2

    # t-learner
    t_models={}
    outcome_model_scores = {}
    for k in tx_classes:
        model = outcome_model_class(random_state=40)
        model.fit(X[T==k], Y[T==k])
        outcome_model_scores[k] = model.score(X[T==k], Y[T==k])
        t_models[k] = model
    
    # build ITE estimator
    tau_models={}
    for k in tx_classes:
        X_k = X[T==k]
        outcomes=[]

        for j in tx_classes:
            if j==k:
                continue
            mu_j = t_models[j].predict(X_k) if continuous_outcome else t_models[j].predict_proba(X_k)[:,1]
            D_kj = Y[T==k] - mu_j
            outcomes.append(D_kj)
            
        D_mean = np.mean(outcomes, axis=0)
        model = base_regressor(random_state=40)
        model.fit(X_k, D_mean)
        tau_models[k] = model
    
    # get propensity score
    pi, propensity_model_score = itemgetter('pi', 'score')(propensity_estimator(X,T, multiclass=multiclass))

    # get individual treatment effects from tau models
    tau_i = {}
    for k in tx_classes:
        tau_i[k] = tau_models[k].predict(X) * pi[:, k]

    # get pairwise ATEs
    results = {}
    for i in range(n_txclasses):
        for j in range(i + 1, n_txclasses):
            tau_diff = tau_i[tx_classes[j]] - tau_i[tx_classes[i]]
            results[f"{tx_classes[j]} vs {tx_classes[i]}"] = {
                "tau": tau_diff.mean(),
                "variance": tau_diff.var(),
                "propensity_model_score": propensity_model_score,
                "outcome_model_score": outcome_model_scores,
            }
    return results

def aipw_estimator(data, treatment_var, outcome_var, covariates, continuous_outcome=False, **kwargs):
    X = data[covariates]
    XT = data[[*covariates, treatment_var]]
    T = data[treatment_var]
    Y = data[outcome_var]
    outcome_model_class = base_regressor if continuous_outcome else base_classifier

    tx_classes = np.sort(np.unique(T)).astype(int)
    n_txclasses = len(tx_classes)
    multiclass = n_txclasses > 2

    # propensity model
    pi, propensity_model_score = itemgetter('pi', 'score')(propensity_estimator(X,T, multiclass=multiclass))

    # s-learner outcome model
    outcome_model = outcome_model_class(random_state=40)
    outcome_model.fit(XT, Y)
    outcome_model_score = outcome_model.score(XT,Y)

    # get predictions from outcome model for each treatment class
    mu = np.zeros((len(data), n_txclasses))
    for k in range(n_txclasses):
        X_k = XT.copy()
        X_k[treatment_var] = k
        mu[:, k] = outcome_model.predict(X_k) if continuous_outcome else outcome_model.predict_proba(X_k)[:,1]

    # get ipw adjusted estimates
    tau_i = {}
    for k in range(n_txclasses):
        treated_pts = (T==k).astype(float)
        ipw_term = treated_pts / pi[:, k] * (Y - mu[:, k])
        tau_i[k] = mu[:, k] + ipw_term 

    # calculate pairwise ates for each tx group
    results = {}
    for i in range(n_txclasses):
        for j in range(i + 1, n_txclasses):
            tau_diff = tau_i[j] - tau_i[i]
            results[f"{tx_classes[j]} vs {tx_classes[i]}"] = {
                "tau": tau_diff.mean(),
                "variance": tau_diff.var(),
                "propensity_model_score": propensity_model_score,
                "outcome_model_score": outcome_model_score,
            }
    return results

