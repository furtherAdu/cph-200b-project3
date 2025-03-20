from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def propensity_estimator(X, T): # propensity scores
    propensity_model = GradientBoostingClassifier(random_state=40)
    pi = propensity_model.fit(X, T).predict_proba(X)[:, 1]
    return pi

def unadjusted_DM_estimator(data, treatment_var, outcome_var, **kwargs):
    EY1 = data.loc[data[treatment_var] == 1, outcome_var].mean()
    EY0 = data.loc[data[treatment_var] == 0, outcome_var].mean()

    tau = EY1 - EY0

    return tau, None

def ipw_estimator(data, treatment_var, outcome_var, covariates):
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]
    
    pi = propensity_estimator(X,T)

    tau_i = (T * Y/ pi) - ((1-T) * Y / (1 - pi))
    variance = (tau_i).var()
    tau = (tau_i).mean()

    return tau, variance

def t_learner(data, treatment_var, outcome_var, covariates):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]

    models = {
        0: GradientBoostingClassifier(random_state=40),
        1: GradientBoostingClassifier(random_state=40)
    }

    models[0].fit(X[T == 0], Y[T == 0])
    models[1].fit(X[T == 1], Y[T == 1])

    mu0 = models[0].predict(X)
    mu1 = models[1].predict(X)

    tau_i = mu1 - mu0

    variance = (tau_i).var()
    tau = (tau_i).mean()

    return tau, variance

def s_learner(data, treatment_var, outcome_var, covariates):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    XT = data[[*covariates, treatment_var]]
    Y = data[outcome_var]

    model = GradientBoostingClassifier(random_state=40)
    model.fit(XT, Y)

    X_treated = XT.copy()
    X_treated[treatment_var] = 1
    X_control = XT.copy()
    X_control[treatment_var] = 0
    
    mu1 = model.predict(X_treated).mean()
    mu0 = model.predict(X_control).mean()

    tau_i = mu1 - mu0

    variance = (tau_i).var()
    tau = (tau_i).mean()

    return tau, variance

def x_learner(data, treatment_var, outcome_var, covariates):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]

    # t-learner
    models = {
        0: GradientBoostingClassifier(random_state=40),
        1: GradientBoostingClassifier(random_state=40)
    }

    models[0].fit(X[T == 0], Y[T == 0])
    models[1].fit(X[T == 1], Y[T == 1])

    # build ITE estimator
    tau_models = {
      0: GradientBoostingRegressor(random_state=40),
      1: GradientBoostingRegressor(random_state=40)
    }
    
    tau_models[0].fit(X[T==0], models[1].predict(X[T==0]) - Y[T==0])
    tau_models[1].fit(X[T==1], Y[T==1] - models[0].predict(X[T==1]))

    # get propensity score
    pi = propensity_estimator(X,T)

    tau_i = pi * tau_models[0].predict(X) + (1-pi) * tau_models[1].predict(X)

    variance = (tau_i).var()
    tau = (tau_i).mean()

    return tau, variance

def aipw_estimator(data, treatment_var, outcome_var, covariates):
    X = data[covariates]
    XT = data[[*covariates, treatment_var]]
    T = data[treatment_var]
    Y = data[outcome_var]

    # propensity model
    pi = propensity_estimator(X,T)

    # s-learner outcome model
    outcome_model = GradientBoostingClassifier(random_state=40)
    outcome_model.fit(XT, Y)

    X_treated = XT.copy()
    X_treated[treatment_var] = 1
    X_control = XT.copy()
    X_control[treatment_var] = 0

    mu1 = outcome_model.predict(X_treated)
    mu0 = outcome_model.predict(X_control)
    
    # AIPW (doubly robust)
    ipw_term = T / pi * (Y - mu1) - (1 - T) / (1 - pi) * (Y - mu0)
    s_learner_term = mu1 - mu0

    tau_i = s_learner_term + ipw_term

    variance = (tau_i).var()
    tau = (tau_i).mean()

    return tau, variance

