# This Julia code is useful to reproduce synthetic data results for our paper "A Large Dimensional Analysis of Regularized Discriminant Analysis Classifiers", 
# submitted to the Journal of Machine Learning Research

# Copyright (c) Khalil Elkhalil, Abla Kammoun, Romain Couillet, Tareq Y. Al-NAffouri, and Mohamed-Slim Alouini

# Contact Persons: Khalil Elkhalil
# E-mail: khalil.elkhalil@kaust.edu.sa

# import useful functions
# Make sure that you include the right path of module.jl
@everywhere include("module.jl");

# Starting the main code
# General parameters

#global n_vect = linspace(1e-3, 5, 15); # Plotting with respect to the regularizer γ
n_vect = 10:10:50; # plotting with respect to p 
gamma = 1; # regularizer 
num_real = 100; # number of Monte Carlo realizations

# Type of results to be plotted 
msg = "RMS"; # "PLAIN": means the value of the error itself, "BIAS" or "RMS": with respect to the real error value 

# loop on the values of p or \gamma depending on plot: it is better to keep it unparallel when the inner loop is parallelized
Error_array = map(1:length(n_vect)) do i 


    # dimensions
    p = Int64(round(n_vect[i]));
    n0 = Int64(round(p));
    n1 = n0;
    n = n0 + n1;
    n0_test = Int64(1000); # testing size for class 0
    n1_test = Int64(round(n0_test * n1/n0)); # testing size for class 0
    n_test = n0_test + n1_test;

    # statistics

    # priors 
    pi0 = n0/n; 
    pi1 = n1/n;

    # covariance matrix design (Exponential model). Any other type can be used as long as norm(Sigma0) = O(1) as p --> ∞ 

    Sigma0 = zeros(p,p);
    for kk = 1:p
        for ll = 1:p
            Sigma0[kk, ll] = 0.6^(abs(kk-ll));
        end
    end
    k = Int64(round(sqrt(p))); # order of the trace(Sigma0 - Sigma1): refer to Assumption 8 in the paper 
    A = [eye(k) zeros(k, p-k); zeros(p-k, k) zeros(p-k, p-k)];
    Sigma1 = Sigma0 + 3 * A; # This model satisfies Assumption 8. in the paper 
    A0 = eig(Sigma0);
    A1 = eig(Sigma1); # eigen decomposition of \Sigma0 and \Sigma1
    ss0 = A0[1]; # Eigenvalues
    ss1 = A1[1]; # Eigenvalues
    UU0 = A0[2]; # Eigenvectors
    UU1 = A1[2]; # Eigenvectors
    Sigma_r0 = sqrtm(Sigma0);
    Sigma_r1 = sqrtm(Sigma1);
    mu0 = [1; zeros(p-1)];
    mu1 = mu0 + 0.8/sqrt(p) * ones(p); # satisfies Assumptions on the means mean(mu0-mu1) = O(1).
    mu = mu0 - mu1;
    
    # Error computation through Monte Carlo

    Error_real = pmap(1:num_real) do it
    # Error_real =  @parallel (+) for it = 1:num_real

        println(i, " ", it);

         # Training the system

        estimates = training_period(p, n0, n1, Sigma_r0, Sigma_r1, mu0, mu1, "synthetic", [], []); # training with synthetic data 

        # getting the empirical estimates

        x0_ = estimates[1];
        x1_ = estimates[2];
        C0 = estimates[3];
        C1 = estimates[4];
        C = estimates[5];
        E0 = eigfact(C0);
        E1 = eigfact(C1);
        s0 = E0[:values];
        U0 = E0[:vectors];
        s1 = E1[:values];
        U1 = E1[:vectors];
        Hr = inv(eye(p) + gamma*C);
        H0r = pinv(gamma*C0 + eye(p));
        H1r = pinv(gamma*C1 + eye(p));
        D0r = ones(p) ./ (ones(p) + gamma*s0);
        D1r = ones(p) ./ (ones(p) + gamma*s1);
        Cr0 = U0 * diagm(sqrt.(abs.(s0))) * U0'; # sqrtm
        Cr1 = U1 * diagm(sqrt.(abs.(s1))) * U1'; # sqrtm

        #  ............... Generate Gaussian data for testing ................
        Y0_test = mu0 * ones(n0_test)' + Sigma_r0 * randn(p,n0_test); # samples for class 0
        Y1_test = mu1 * ones(n1_test)' + Sigma_r1 * randn(p,n1_test); # samples for class 1
        
        # true LDA error based on testing data 
        err_lda = empirical_error(x0_, x1_, Hr, pi1, pi0, [], [], [], [], Y0_test, Y1_test, "lda");

        # RLDA G estimator
        err_g_lda = LDA_General_estim(n0, n1, x0_, x1_, C0, C1, gamma);
 
        # training sets as output of the training_period
        Y0_train = estimates[6];
        Y1_train = estimates[7];

        # Error on the training
        err_train_lda = empirical_error(x0_, x1_, Hr, pi1, pi0, [], [], [], [], Y0_train, Y1_train, "lda");

        # cross validation for RLDA
        error_cv = cross_valid(5, 5, p, gamma, n0, n1, Y0_train, Y1_train, "test"); # cross validation is always performed on a partial testin set
        err_cv_lda = error_cv[1];

        # Plugin estimator LDA
        err_plug_lda = DE_lda(C0, C1, x0_-x1_, p, n0, n1, gamma); # substitue with the sample estimated in the deterministic equivalents

        # True error RQDA: based on testing data
        err_qda = qda_true(p, n0_test, n1_test, Sigma_r0, Sigma_r1, mu0, mu1, D0r, D1r, H0r, H1r, Hr, x0_, x1_, pi0, pi1, "synthetic", [], []);

        # RQDA G estimator
        err_g_qda = QDA_General_estim(p, n0, n1, x0_, x1_, s0, s1, U0, U1, C0, C1, gamma); 

        # Error on the training for RQDA
        err_train_qda = empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, Y0_train, Y1_train, "qda");

        # cross validation for RQDA
        err_cv_qda = error_cv[2] ;

        # Bootstrap
        err_boot = Bootstrap_lda_qda(5, p, n0, n1, gamma, Y0_train, Y1_train, x0_, x1_, C0, C1, H0r, H1r, D0r, D1r) ;

        # Bootstrap 0.632
        err_boot_lda = 0.632 * err_boot[1] + (1-0.632) * err_train_lda;
        err_boot_qda = 0.632 * err_boot[2] + (1-0.632) * err_train_qda;

        # No information error rate
        err_no_info_lda = no_info(Y0_train, Y1_train, n0, n1, x0_, x1_, Hr, H0r, H1r, D0r, D1r, pi1, pi0, "lda");
        err_no_info_qda = no_info(Y0_train, Y1_train, n0, n1, x0_, x1_, Hr, H0r, H1r, D0r, D1r, pi1, pi0, "qda");

        # 0.632+ Bootstrap
        R_lda = (err_boot[1] - err_train_lda) / (err_no_info_lda - err_train_lda); # relative overfitting LDA
        R_qda = (err_boot[2] - err_train_qda) / (err_no_info_qda - err_train_qda); # relative overfitting QDA

        w_lda = 0.632 / (1 - 0.368 * R_lda);
        w_qda = 0.632 / (1 - 0.368 * R_qda);

        err_boot_plus_lda = (1-w_lda) * err_train_lda + w_lda * err_boot[1];
        err_boot_plus_qda = (1-w_qda) * err_train_qda + w_qda * err_boot[2];

        # Plugin RQDA

        err_plug_qda = DE_qda(gamma, n0, n1, p, C0, C1, U0, s0, U1, s1, x0_-x1_, Cr0, Cr1);

        # Type of the results depending on the value of msg.
        if msg == "PLAIN"

            return  err_lda, err_g_lda, err_train_lda, err_cv_lda, err_boot_lda, err_boot_plus_lda, err_plug_lda,
                    err_qda, err_g_qda, err_train_qda, err_cv_qda, err_boot_qda, err_boot_plus_qda, err_plug_qda;

            elseif msg == "BIAS" || msg == "RMS"
            return  real(err_lda-err_lda), real(err_g_lda - err_lda), real(err_train_lda - err_lda), real(err_cv_lda - err_lda), real(err_boot_lda - err_lda), real(err_boot_plus_lda-err_lda), real(err_plug_lda-err_lda),
                    real(err_qda - err_qda), real(err_g_qda - err_qda), real(err_train_qda - err_qda), real(err_cv_qda - err_qda), real(err_boot_qda - err_qda), real(err_boot_plus_qda - err_qda), real(err_plug_qda - err_qda);

        end

    end

    Error_real = tuple_to_array(Error_real, 14) ;

    # R-LDA
    error_lda = Error_real[:, 1] ;
    error_g_lda = Error_real[:, 2] ;
    error_train_lda = Error_real[:, 3] ;
    error_cv_lda = Error_real[:, 4] ;
    error_boot_lda = Error_real[:, 5] ;
    error_boot_plus_lda = Error_real[:, 6] ;
    error_plugin_lda = Error_real[:, 7];

    # R-QDA
    error_qda = Error_real[:, 8] ;
    error_g_qda = Error_real[:, 9] ;
    error_train_qda = Error_real[:, 10] ;
    error_cv_qda = Error_real[:, 11] ;
    error_boot_qda = Error_real[:, 12] ;
    error_boot_plus_qda = Error_real[:, 13] ;
    error_plugin_qda = Error_real[:, 14] ;

      # Type of the results depending on the value of msg.
    if msg == "PLAIN" || msg == "BIAS"
        return    abs(mean(error_lda)), abs(mean(error_g_lda)), abs(mean(error_train_lda)), abs(mean(error_cv_lda)), abs(mean(error_boot_lda)), abs(mean(error_boot_plus_lda)), abs(mean(error_plugin_lda)),
                  abs(mean(error_qda)), abs(mean(error_g_qda)), abs(mean(error_train_qda)), abs(mean(error_cv_qda)), abs(mean(error_boot_qda)), abs(mean(error_boot_plus_qda)), abs(mean(error_plugin_qda));
    elseif msg == "RMS"
        return    rms(error_lda), rms(error_g_lda), rms(error_train_lda), rms(error_cv_lda), rms(error_boot_lda), rms(error_boot_plus_lda), rms(error_plugin_lda),
                  rms(error_qda), rms(error_g_qda), rms(error_train_qda), rms(error_cv_qda), rms(error_boot_qda), rms(error_boot_plus_qda), rms(error_plugin_qda);
    end


end

# R-LDA
Error_array = tuple_to_array(Error_array, 14) ;
Error_array = ["true lda" "lda g estim" "lda train" "lda cv" "lda boot" "lda boot +" "lda Plugin" "true qda" "qda g estim" "qda train" "qda cv" "qda boot" "qda boot +" "qda Plugin"; Error_array];
lda_true_error = Error_array[2:end, 1] ;
lda_g_error = Error_array[2:end, 2] ;
lda_train_error = Error_array[2:end, 3] ;
lda_cv_error = Error_array[2:end, 4] ;
lda_boot_error = Error_array[2:end, 5] ;
lda_boot_plus_error = Error_array[2:end, 6] ;
lda_plugin_error = Error_array[2:end, 7] ;

# R-QDA
qda_true_error = Error_array[2:end, 8] ;
qda_g_error = Error_array[2:end, 9] ;
qda_train_error = Error_array[2:end, 10] ;
qda_cv_error = Error_array[2:end, 11] ;
qda_boot_error = Error_array[2:end, 12] ;
qda_boot_plus_error = Error_array[2:end, 13] ;
qda_plugin_error = Error_array[2:end, 14] ;

# Exporting the results to mat files so that you will be able to plot then on Matlab.
# Pkg.add("MAT"); # if you don't alrady have the package
using MAT

array = Error_array[2:end, :];
matwrite("results.mat", Dict("array" => array));
