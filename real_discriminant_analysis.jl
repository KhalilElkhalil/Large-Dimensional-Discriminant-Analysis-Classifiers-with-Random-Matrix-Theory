# This Julia code is useful to reproduce real data results for our paper "A Large Dimensional Analysis of Regularized Discriminant Analysis Classifiers", 
# submitted to the Journal of Machine Learning Research

# Copyright (c) Khalil Elkhalil, Abla Kammoun, Romain Couillet, Tareq Y. Al-NAffouri, and Mohamed-Slim Alouini

# Contact Persons: Khalil Elkhalil
# E-mail: khalil.elkhalil@kaust.edu.sa

# import useful functions
# Make sure that you include the right path of module.jl
@everywhere include("module.jl") ;
# Starting the main code

#tic();

# General parameters
gamma_vect = linspace(1e-2, 10, 2); # regularizer
num_real = 12; # number of Monte Carlo realizations

# Type of results to be plotted 
msg = "rms"; # "plain": means the value of the error itself, "bias" or "rms": with respect to the real error value 

#data = MNIST_data(1, 7); # MNIST data for the digits 5 and 6
data = USPS_Data(5, 6); # USPS data for the digits 5 and 6
train0 = data[1]; # all training data for class 0
test0 = data[2]; # all testing data for class 0
train1 = data[3]; # all training data for class 1
test1 = data[4]; # all testing data for class 1
n0_test = size(test0)[2];
n1_test = size(test1)[2];
n_test = n0_test + n1_test;


Error_array = map(1:length(gamma_vect)) do i # it is better to keep it unparallel when the inside loop is parallelized

    gamma = gamma_vect[i];
    #gamma = 1;
    #n0 = Int64(round(gamma_vect[i]));
    #n1 = Int64(round(n0));
    n0 = 400; 
    n1 = n0;
    n = n0 + n1; 

    # statistics
    pi0 = n0 / n;
    pi1 = n1 / n;

    # dimensions
    p = size(train0)[1];
    N0 = size(train0)[2];
    N1 = size(train1)[2];
    N = N0 + N1;
    n0_test = size(test0)[2];
    n1_test = size(test1)[2];
    n_test = n0_test + n1_test;

    # Error computation through Monte Carlo
    Error_real = pmap(1:num_real) do it
    # error =  @parallel (+) for it = 1:num_real

        println(i, " ", it);

        # Training the system
        train_pattern0 = sample(1:N0, n0;replace=false); # random patter of size n0 out of N0 
        train_pattern1 = sample(1:N1, n1;replace=false); # random patter of size n1 out of N1 

        train_set0 = train0[:, train_pattern0]; # randomly selected set from train_set0
        train_set1 = train1[:, train_pattern1]; # randomly selected set from train_set1

        estimates = training_period(p, n0, n1, [], [], [], [], "real", train_set0, train_set1) # training with synthetic data 

        # getting the empirical estimates
        x0_ = estimates[1];
        x1_ = estimates[2];
        C0 = estimates[3];
        C1 = estimates[4];
        C = estimates[5];
        E = eig(C);
        s = E[1];
        U = E[2];
        Hr = pinv(eye(p) + gamma*C);
        E0 = eigfact(eye(p) + gamma*C0);
        E1 = eigfact(eye(p) + gamma*C1);
        s0 = (E0[:values] - ones(p)) / gamma;
        s1 = (E1[:values] - ones(p)) / gamma;
        U0 = E0[:vectors];
        U1 = E1[:vectors];
        H0r = pinv(eye(p) + gamma*C0);
        H1r = pinv(eye(p) + gamma*C1);
        D0r = ones(p) ./ E0[:values];
        D1r = ones(p) ./ E1[:values];
        Cr0 = U0 * diagm(sqrt.(abs.(s0))) * U0'; # sqrtm
        Cr1 = U1 * diagm(sqrt.(abs.(s1))) * U1'; # sqrtm

        # RLDA error estimators
        test_size = min(size(test0)[2], size(test1)[2]); # testing size for both classes: chosen to be equal 

        # True error based on the testing data     
        err_lda = empirical_error(x0_, x1_, Hr, pi1, pi0, [], [], [], [], test0[:, 1:test_size], test1[:, 1:test_size], "lda");

        # RLDA G estimator
        err_g_lda = LDA_General_estim(n0, n1, x0_, x1_, C0, C1, gamma); 

        # training sets
        Y0_train = estimates[6];
        Y1_train = estimates[7];

        # Error on the training
        err_train_lda = empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, train_set0, train_set1, "lda");

        # cross validation for RLDA
        error_cv = cross_valid(5, 5, p, gamma, n0, n1, train_set0, train_set1, "test"); # cross validation is always performed on a partial testing set
        err_cv_lda = error_cv[1];

        # Plugin estimator LDA
        err_plug_lda = DE_lda(C0, C1, x0_-x1_, p, n0, n1, gamma);

        # True error RQDA: based on testing data
        err_qda = qda_true(p, n0_test, n1_test, [], [], [], [], D0r, D1r, H0r, H1r, Hr, x0_, x1_, pi0, pi1, "real", test0[:, 1:test_size], test1[:, 1:test_size]);

        # RQDA G estimator   
        err_g_qda = QDA_General_estim(p, n0, n1, x0_, x1_, s0, s1, U0, U1, C0, C1, gamma);

        # Error on the training for RQDA
        err_train_qda = empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, train_set0, train_set1, "qda");

        # cross validation for RQDAin
        err_cv_qda = error_cv[2] ;

        # Bootstrap
        err_boot = Bootstrap_lda_qda(5, p, n0, n1, gamma, train_set0, train_set1, x0_, x1_, C0, C1, H0r, H1r, D0r, D1r) ;

        # Bootstrap 0.632
        err_boot_lda = 0.632 * err_boot[1] + (1-0.632) * err_train_lda;
        err_boot_qda = 0.632 * err_boot[2] + (1-0.632) * err_train_qda;

        # No information error rate
        err_no_info_lda = no_info(train_set0, train_set1, n0, n1, x0_, x1_, Hr, H0r, H1r, D0r, D1r, pi1, pi0, "lda");
        err_no_info_qda = no_info(train_set0, train_set1, n0, n1, x0_, x1_, Hr, H0r, H1r, D0r, D1r, pi1, pi0, "qda");

        # 0.632+ Bootstrap
        R_lda = (err_boot[1] - err_train_lda) / (err_no_info_lda - err_train_lda); # relative overfitting LDA
        R_qda = (err_boot[2] - err_train_qda) / (err_no_info_qda - err_train_qda); # relative overfitting QDA
        w_lda = 0.632 / (1 - 0.368 * R_lda);
        w_qda = 0.632 / (1 - 0.368 * R_qda);
        err_boot_plus_lda = (1-w_lda) * err_train_lda + w_lda * err_boot[1];
        err_boot_plus_qda = (1-w_qda) * err_train_qda + w_qda * err_boot[2];

        # Plugin RQDA
        err_plug_qda = DE_qda(gamma, n0, n1, p, C0, C1, U0, s0, U1, s1, x0_-x1_, Cr0, Cr1);

        #= ................................ Gamma optimization ......................................
        # Optimizing the gamma 
        # First stage optimization using the G estimator
        gamma_object = optimize_DA(p, n0, n1, x0_, x1_, s0, s1, U0, U1, s, U, C0, C1);
        gamma_g_lda = gamma_object[1]; # optimal regularizer for R-LDA based on its G estimator
        gamma_g_qda = gamma_object[3]; # optimal regularizer for R-QDA based on its G estimator

        println("The optimal LDA G value is ", gamma_g_lda);
        println("The optimal QDA G value is ", gamma_g_qda);

        # Second stage optimization using testing data or cross validation 
        gamma_opt = two_stage_test(p, n0_test, n1_test, x0_, x1_, pi0, pi1, s, U, s0, U0, s1, U1, test0, test1, gamma_g_lda, gamma_g_qda, 50);
        err_lda = gamma_opt[1]; 
        err_g_lda = gamma_opt[2]
        err_qda = gamma_opt[3];
        err_g_qda = gamma_opt[4];          
        println("The optimal LDA value is ", gamma_opt[1]);
        println("The optimal QDA value is ", gamma_opt[3]);
        # ...........................................................................................
        =# 
        # returing the required statistics based on the value of msg.
        if msg == "plain"

            return  err_lda, err_g_lda, err_cv_lda, err_boot_lda, err_boot_plus_lda, err_plug_lda,
                    err_qda, err_g_qda, err_cv_qda, err_boot_qda, err_boot_plus_qda, err_plug_qda;

            elseif msg == "bias" || msg == "rms"
            return  err_lda-err_lda, err_g_lda - err_lda, err_cv_lda - err_lda, err_boot_lda - err_lda, err_boot_plus_lda - err_lda, err_plug_lda - err_lda,
                    err_qda - err_qda, err_g_qda - err_qda, err_cv_qda - err_qda, err_boot_qda - err_qda, err_boot_plus_qda - err_qda, err_plug_qda - err_qda;

        end

    end

    Error_real = tuple_to_array(Error_real, 12) ;

    # R-LDA
    error_lda = Error_real[:, 1] ;
    error_g_lda = Error_real[:, 2] ;
    #error_train_lda = Error_real[:, 3] ;
    error_cv_lda = Error_real[:, 3] ;
    error_boot_lda = Error_real[:, 4] ;
    error_boot_plus_lda = Error_real[:, 5] ;
    error_plug_lda = Error_real[:, 6] ;

    # R-QDA
    error_qda = Error_real[:, 7] ;
    error_g_qda = Error_real[:, 8] ;
    #error_train_qda = Error_real[:, 10] ;
    error_cv_qda = Error_real[:, 9] ;
    error_boot_qda = Error_real[:, 10] ;
    error_boot_plus_qda = Error_real[:, 11] ;
    error_plug_qda = Error_real[:, 12] ;

      # taking the average
    if msg == "plain" || msg == "bias"
        return    mean(error_lda), mean(error_g_lda), mean(error_cv_lda), mean(error_boot_lda), mean(error_boot_plus_lda), mean(error_plug_lda),
                  mean(error_qda), mean(error_g_qda), mean(error_cv_qda), mean(error_boot_qda), mean(error_boot_plus_qda), mean(error_plug_qda);
    elseif msg == "rms"
        return    rms(error_lda), rms(error_g_lda), rms(error_cv_lda), rms(error_boot_lda), rms(error_boot_plus_lda), rms(error_plug_lda),
                  rms(error_qda), rms(error_g_qda), rms(error_cv_qda), rms(error_boot_qda), rms(error_boot_plus_qda), rms(error_plug_qda);
    end


end

Error_array = tuple_to_array(Error_array, 12) ;

# Exporting the results to mat files so that you will be able to plot then on Matlab.
# Pkg.add("MAT"); # if you don't alrady have the package
using MAT
array = Error_array;
matwrite("results.mat", Dict(
                     "array" => array
             ));
