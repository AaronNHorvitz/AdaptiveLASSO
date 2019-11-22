
def Adaptive_LASSO(X_train,y_train,max_iterations = 1000,lasso_iterations = 10, alpha = 0.1, tol = 0.001, max_error_up = 5, title = ''):
    
    # set checks
    higher  = float('inf')
    lower   = 0
    
    # set lists
    coefficients_list = []
    iterations_list   = []
    
    # set variables
    X_train  = X_train
    y_train  = y_train
    
    # set constants
    alpha    = alpha
    tol      = tol
    max_iter = max_iterations
    n_lasso_iterations = lasso_iterations
    
    g = lambda w: np.sqrt(np.abs(w))
    gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

    n_samples, n_features = X_train.shape
    p_obj = lambda w: 1. / (2 * n_samples) * np.sum((y_train - np.dot(X_train, w)) ** 2) \
                      + alpha * np.sum(g(w))

    weights = np.ones(n_features)

    X_w = X_train / weights[np.newaxis, :]
    X_w  = np.nan_to_num(X_w)
    X_w  = np.round(X_w,decimals = 3)

    y_train    = np.nan_to_num(y_train)

    adaptive_lasso = Lasso(alpha=alpha, fit_intercept=False)

    adaptive_lasso.fit(X_w, y_train)

    for k in range(n_lasso_iterations):
        X_w = X_train / weights[np.newaxis, :]
        adaptive_lasso = Lasso(alpha=alpha, fit_intercept=False)
        adaptive_lasso.fit(X_w, y_train)
        coef_ = adaptive_lasso.coef_ / weights
        weights = gprime(coef_)
        
        print ('Iteration #',k+1,':   ',p_obj(coef_))  # should go down
        
        iterations_list.append(k)
        coefficients_list.append(p_obj(coef_))
        
    print (np.mean((adaptive_lasso.coef_ != 0.0) == (coef_ != 0.0)))   
    
    coef = pd.Series(adaptive_lasso.coef_, index = X_train.columns)
    print('=============================================================================')
    print("Adaptive LASSO picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables.")
    print('=============================================================================')

    plt.rcParams["figure.figsize"] = (18,8)

    # subplot of the predicted vs. actual

    plt.plot(iterations_list,coefficients_list,color = 'orange')
    plt.scatter(iterations_list,coefficients_list,color = 'green')
    plt.title('Iterations vs. p_obj(coef_)')
    plt.show()

    # plot of the coefficients'

    imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
    imp_coef.plot(kind = "barh", color = 'green',  fontsize=14)
    plt.title("Top and Botton 10 Coefficients Selected by the Adaptive LASSO Model", fontsize = 14)
    plt.show()
    return adaptive_lasso

# variable selection with LASSO for the model

y_train = numbers_df['y']
X_train = numbers_df[[col for col in numbers_df.columns if col != 'y']]

model = Adaptive_LASSO(X_train,
                       y_train,
                       max_iterations = 1000,
                       lasso_iterations = 10, 
                       alpha = 0.1, 
                       tol = 0.001, 
                       max_error_up = 5, 
                       title = '')

# look at the coefficients in the model

coef = pd.Series(model.coef_, index = X_train.columns)
coef = pd.DataFrame(coef).reset_index()
coef_list = coef.loc[coef[0]!= 0.0]['index'].to_list()
new_X_train = X_train[coef_list]