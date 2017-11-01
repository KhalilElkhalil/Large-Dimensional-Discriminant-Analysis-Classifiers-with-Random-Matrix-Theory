# Module containing all the needed function to run "synthetic_discriminant_analysis.jl" and "real_discriminant_analysis.jl"

# adding some SpecialFunctions

#Pkg.add("SpecialFunctions");
using SpecialFunctions;

# adding the MLBase package for machine learning use

#Pkg.add("MLBase");
using MLBase;

# solve second order equation

function quadratic(a, b, c)
  discr = b^2 - 4*a*c;
  sq = (discr > 0) ? sqrt(discr) : sqrt(discr + 0im);

  return (-b - sq)/(2a), (-b + sq)/(2a);
end

# From tuple to array

function tuple_to_array(v,num_out)

  w = zeros(1,num_out);
  for i =1:length(v)
    temp = collect(v[i])' ;
    w = vcat(w,temp)
  end
  w = w[2:end,:];
  return w;
end

function rms(v)
    return sqrt(abs(mean(v)^2 + var(v)));
end

# Defining the Q function for error computations

function qfunc(x)

    1/2 * erfc(x/sqrt(2));
end

# Training the system (Gaussian Synthetic data)

function training_period(p, n0, n1, Sigma_r0, Sigma_r1, mu0, mu1, typedata, data0, data1)

    u0 = ones(n0)/n0;
    u1 = ones(n1)/n1;

    if typedata == "synthetic"

        Z0_train = randn(p,n0); # training data for C0
        Z1_train = randn(p,n1); # training data for C1
        Y0 = Sigma_r0 * Z0_train;
        Y1 = Sigma_r1 * Z1_train;
        Y0_train = mu0 * ones(n0)' + Sigma_r0 * Z0_train; # All training data as columns of C0
        Y1_train = mu1 * ones(n1)' + Sigma_r1 * Z1_train; # All training data as columns of C0

    elseif typedata == "real"
        Y0_train = data0;
        Y1_train = data1;
    end

    # Compute the statistics

    x0_ = Y0_train * u0;
    x1_ = Y1_train * u1; # sample means
    C0 = 1/(n0-1)*(Y0_train - x0_ * ones(n0)')*(Y0_train - x0_ * ones(n0)')'; # SCM for C0
    C1 = 1/(n1-1)*(Y1_train - x1_ * ones(n1)')*(Y1_train - x1_ * ones(n1)')'; # SCM for C1
    C = ((n0-1) * C0 + (n1-1) * C1) / (n0 + n1 - 2) ; # common (pooled) SCM for LDA

    return x0_, x1_, C0, C1, C, Y0_train, Y1_train;
end

# RLDA or RQDA empirical error

function empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, Y0_test, Y1_test, discriminant)

    error_lda = 0;
    error_qda = 0;
    n0_test = size(Y0_test)[2];
    n1_test = size(Y1_test)[2];
    Y_test = [Y0_test Y1_test];

    for ii = 1:n0_test
        x = Y0_test[:, ii];

        # LDA

        if discriminant == "lda"
            WLDA = (x-0.5*(x0_+x1_))' * Hr * (x0_-x1_) - log(pi1/pi0); # linear discriminant using the pooled SCM

            if real(WLDA) < 0
                error_lda +=  1;
            end

        elseif discriminant == "qda"

        # QDA

            qda0 = 0.5 * sum(log.(abs.(D0r))) - 0.5 * (x-x0_)' * H0r * (x-x0_) + log(pi0); # quadratic disc. for C0
            qda1 = 0.5 * sum(log.(abs.(D1r))) - 0.5 * (x-x1_)' * H1r * (x-x1_) + log(pi1); # quadratic disc. for C1

            if real(qda0) < real(qda1)
                error_qda +=  1;
            end
        end

    end

    for ii = 1:n1_test
        x = Y1_test[:, ii];

        # LDA

        if discriminant == "lda"
            WLDA = (x-0.5*(x0_+x1_))' * Hr * (x0_-x1_) - log(pi1/pi0); # linear discriminant

            if real(WLDA) > 0
                error_lda += 1;
            end

        elseif discriminant == "qda"


        # QDA

            qda0 = 0.5 * sum(log.(abs.(D0r))) - 0.5 * (x-x0_)' * H0r * (x-x0_) + log(pi0); # quadratic disc. for C0
            qda1 = 0.5 * sum(log.(abs.(D1r))) - 0.5 * (x-x1_)' * H1r * (x-x1_) + log(pi1); # quadratic disc. for C1

            if real(qda0) > real(qda1)
                error_qda += 1;
            end
        end
    end

    error_lda =  error_lda / (n0_test + n1_test);
    error_qda = error_qda / (n0_test + n1_test);

        if discriminant == "lda"
            return error_lda;
        elseif discriminant == "qda"
            return error_qda;
        end

end

#=
function empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, Y0_test, Y1_test, discriminant)

    error_lda = 0;
    error_qda = 0;
    n0_test = size(Y0_test)[2];
    n1_test = size(Y1_test)[2];
    (p, ~) = size(H0r);
    n = n0_test + n1_test; # total # of training samples
    Y_test = [Y0_test Y1_test];

    # vectorization of the discriminant computation

    #dr0 = 0.5 * sum(log.(abs.(D0r)));
    #dr1 = 0.5 * sum(log.(abs.(D1r)));
    #WLDA_mat = (Y_test - 0.5 * (x0_+x1_) * ones(1, n))' * Hr * (x0_+x1_) - log(pi1/pi0) * ones(n);
    #WQDA_mat0 = dr0  * eye(n) - 0.5 * (Y_test - x0_ * ones(1, n))' * H0r * (Y_test - x0_ * ones(1, n)) + log(pi0) * eye(n);
    #WQDA_mat1 = dr1  * eye(n) - 0.5 * (Y_test - x1_ * ones(1, n))' * H1r * (Y_test - x1_ * ones(1, n)) + log(pi1) * eye(n);

    # Error computation
     if discriminant == "lda"
         WLDA_mat = (Y_test - 0.5 * (x0_+x1_) * ones(1, n))' * Hr * (x0_+x1_) - log(pi1/pi0) * ones(n);
         sign_w = isless.(WLDA_mat, 0) #
         indic = ! isless.(1:n, n0_test+1) # indices < n0
         er_v = xor(sign_w, indic);
         error_lda = real(sum(complex(er_v).^2));
         error_lda =  error_lda / (n0_test + n1_test);
         return error_lda;

     elseif discriminant == "qda"
         err_qda = 0;
         return err_qda;
     end



     #=
    for ii = 1:n0_test
        #x = Y0_test[:, ii];

        # LDA

        if discriminant == "lda"
            #WLDA = (x-0.5*(x0_+x1_))' * Hr * (x0_-x1_) - log(pi1/pi0); # linear discriminant using the pooled SCM
            WLDA = WLDA_mat[ii];
            if WLDA < 0
                error_lda +=  1;
            end

        elseif discriminant == "qda"

        # QDA

            #qda0 = 0.5 * sum(log.(abs.(D0r))) - 0.5 * (x-x0_)' * H0r * (x-x0_) + log(pi0); # quadratic disc. for C0
            #qda1 = 0.5 * sum(log.(abs.(D1r))) - 0.5 * (x-x1_)' * H1r * (x-x1_) + log(pi1); # quadratic disc. for C1
            qda0 = WQDA_mat0[ii, ii];
            qda1 = WQDA_mat1[ii, ii];
            if real(qda0) < real(qda1)
                error_qda +=  1;
            end
        end

    end

    for ii = 1:n1_test
        #x = Y1_test[:, ii];

        # LDA

        if discriminant == "lda"
            #WLDA = (x-0.5*(x0_+x1_))' * Hr * (x0_-x1_) - log(pi1/pi0); # linear discriminant
            WLDA = WLDA_mat[ii+n0_test];
            if WLDA > 0
                error_lda += 1;
            end

        elseif discriminant == "qda"


        # QDA

            #qda0 = 0.5 * sum(log.(abs.(D0r))) - 0.5 * (x-x0_)' * H0r * (x-x0_) + log(pi0); # quadratic disc. for C0
            #qda1 = 0.5 * sum(log.(abs.(D1r))) - 0.5 * (x-x1_)' * H1r * (x-x1_) + log(pi1); # quadratic disc. for C1
            qda0 = WQDA_mat0[ii+n0_test, ii+n0_test];
            qda1 = WQDA_mat1[ii+n0_test, ii+n0_test];
            if real(qda0) > real(qda1)
                error_qda += 1;
            end
        end
    end

    error_lda =  error_lda / (n0_test + n1_test);
    error_qda = error_qda / (n0_test + n1_test);

        if discriminant == "lda"
            return error_lda;
        elseif discriminant == "qda"
            return error_qda;
        end
    =#
end
=#
# RLDA true error

function lDA_true(mu0, mu1, Sigma0, Sigma1, pi0, pi1, x0_, x1_, Hr, type_eval, Y0_test, Y1_test)
    if type_eval == "synthetic" # we have a closed from expression in this case
        GG = [-(mu0-0.5*(x0_+x1_))' * Hr * (x0_-x1_) + log(pi1/pi0) , (mu1-0.5*(x0_+x1_))' * Hr * (x0_-x1_) - log(pi1/pi0)];
        DD = [(x0_-x1_)' * Hr * Sigma0 *Hr* (x0_-x1_) , (x0_-x1_)' * Hr * Sigma1 *Hr* (x0_-x1_)];
        err_lda = pi0 * (1-qfunc(GG[1]/sqrt(DD[1]))) + pi1 * (1-qfunc(GG[2]/sqrt(DD[2])));

    elseif type_eval == "real"
        err_lda = empirical_error(x0_, x1_, Hr, pi1, pi0, [], [], [], [], Y0_test, Y1_test, "lda");
    end

    return err_lda;
end

# RLDA G estimator

function LDA_General_estim(n0, n1, x0_, x1_, C0, C1, gamma)
#function LDA_General_estim(p, n0, n1, x0_, x1_, C0, C1, s, U, gamma)

    #Hr = U * diagm(ones(p) ./ (ones(p) + gamma * s)) * U'; # Hr with the RLDA regularizeri
    p = size(C0)[1]; 
    Hr = pinv(eye(p) + gamma * ((n0-1)*C0 + (n1-1)*C1) / (n0+n1-2));
    n = n0+n1;
    pi0 = n0/n;
    pi1 = n1/n;
    delta0_hat = 1/n0*trace(C0*Hr) / (1-gamma/n * trace(C0*Hr));
    delta1_hat = 1/n1*trace(C1*Hr) / (1-gamma/n * trace(C1*Hr));
    G0_hat = (x0_ - (x0_+x1_)/2)' * Hr * (x0_-x1_);
    G1_hat = (x1_ - (x0_+x1_)/2)' * Hr * (x0_-x1_);
    psi0 = (1-gamma/(n-2)*trace(C0*Hr))^2;
    psi1 = (1-gamma/(n-2)*trace(C1*Hr))^2;
    D0_hat = 1/psi0 * (x0_-x1_)' * Hr * C0 * Hr * (x0_-x1_);
    D1_hat = 1/psi1 * (x0_-x1_)' * Hr * C1 * Hr * (x0_-x1_);
    arg0 = (-G0_hat + delta0_hat + log(pi1/pi0)) / sqrt(D0_hat);
    arg1 = (G1_hat + delta1_hat - log(pi1/pi0)) / sqrt(D1_hat);
    LDA_estim = real(pi0 *(1-qfunc(arg0)) + pi1 *(1-qfunc(arg1)));

    return LDA_estim;
end

# RQDA empirical error

function qda_true(p, n0_test, n1_test, Sigma_r0, Sigma_r1, mu0, mu1, D0r, D1r, H0r, H1r, Hr, x0_, x1_, pi0, pi1, type_eval, Y0_test, Y1_test)
    if type_eval == "synthetic"
         Z0_test = randn(p,n0_test) ; # testing data for C0
         Z1_test = randn(p,n1_test) ; # testing data for C1
         Y0_test = mu0 * ones(n0_test)' + Sigma_r0 * Z0_test ; # All testing data as columns of C0
         Y1_test = mu1 * ones(n1_test)' + Sigma_r1 * Z1_test ; # All testing data as columns of C1
         err_qda = empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, Y0_test, Y1_test, "qda");
    elseif type_eval == "real"
        err_qda = empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, Y0_test, Y1_test, "qda");
    end

    return err_qda;
end

# RQDA G estimator

#function QDA_General_estim(p, n0, n1, x0_, x1_, H0r, H1r, D0r, D1r, C0, C1, gamma)
function QDA_General_estim(p, n0, n1, x0_, x1_, s0, s1, U0, U1, C0, C1, gamma)

    D0r = ones(p) ./ (ones(p) + gamma*s0);
    D1r = ones(p) ./ (ones(p) + gamma*s1);
    #H0r = U0 * diagm(D0r) * U0';
    #H1r = U1 * diagm(D1r) * U1';
    H1r = pinv(gamma*C1 + eye(p));
    H0r = pinv(gamma*C0 + eye(p));
    x_ = x0_-x1_ ;

    GE_epsilon = 0 ;

    # Defining some useful estimates

    delta0_hat = 1/gamma * (p/n0-1/n0*trace(H0r)) / (1-p/n0+1/n0*trace(H0r)) ;
    delta1_hat = 1/gamma * (p/n1-1/n1*trace(H1r)) / (1-p/n1+1/n1*trace(H1r)) ;
    #=delta0_hat_prime = -delta0_hat / gamma + 1/gamma * (1/n0 * trace(H0r*C0*H0r)) / (1-p/n0 + trace(H0r)/n0)^2 ;
    delta1_hat_prime = -delta1_hat / gamma + 1/gamma * (1/n1 * trace(H1r*C1*H1r)) / (1-p/n1 + trace(H1r)/n1)^2 ;
    delta_tilde0_hat = 1/(1+gamma*delta0_hat) ;
    delta_tilde1_hat = 1/(1+gamma*delta1_hat) ;
    delta_tilde0_hat_prime = - (delta0_hat + gamma*delta0_hat_prime) * delta_tilde0_hat^2 ;
    delta_tilde1_hat_prime = - (delta1_hat + gamma*delta1_hat_prime) * delta_tilde1_hat^2 ;
    phi0_hat = -delta0_hat_prime / (delta_tilde0_hat+gamma*delta_tilde0_hat_prime) ;
    phi1_hat = -delta1_hat_prime / (delta_tilde1_hat+gamma*delta_tilde1_hat_prime);
    phi_tilde0_hat = delta_tilde0_hat^2 ;
    phi_tilde1_hat = delta_tilde1_hat^2 ;
    =#

    # Estimates of main quantities

    #= 1/n1 * tr T1^2

    lamda1 = 1 / (1-gamma^2*phi1_hat*phi_tilde1_hat) ;
    a1 = 1/n1 * trace(H1r) ;
    H1 = 1/n0 * trace(H1r^2) ;
    sol1 = quadratic(real(lamda1), real(1-2*a1*lamda1), real(lamda1*a1^2-H1)) ;
    tr_T12 = sol1[2] ; # take the positive solution
    =#

    #= 1/n1 * tr T1^2

    lamda0 = 1 / (1-gamma^2*phi0_hat*phi_tilde0_hat) ;
    a0 = 1/n0 * trace(H0r) ;
    H0 = 1/n0 * trace(H0r^2) ;
    sol0 = quadratic(real(lamda0), real(1-2*a0*lamda0), real(lamda0*a0^2-H0)) ;
    tr_T02 = sol0[2] ; # take the positive solution
    =#
    # Other quantities

    #= 1/n0 * trace(Sigma0*T1^2)
    GE_trS0T12 = (1/n1 * trace(C0*H1r^2) - 1/(1-gamma^2*phi1_hat*phi_tilde1_hat) * 1/n1 * trace(C0*H1r) * (1/n1 * trace(H1r) - tr_T12)) / (1 -1/(1-gamma^2*phi1_hat*phi_tilde1_hat)*(1/n1 * trace(H1r) - tr_T12)) ;
    GE_S0S1T12 = 1/n1/gamma/delta_tilde1_hat * trace(C0*H1r) - 1/gamma/delta_tilde1_hat * GE_trS0T12 ;

    # 1/n0 * trace(Sigma0*T1^2)
    GE_trS1T02 = (1/n1 * trace(C1*H0r^2) - 1/(1-gamma^2*phi0_hat*phi_tilde0_hat) * 1/n1 * trace(C1*H0r) * (1/n1 * trace(H0r) - tr_T02)) / (1 -1/(1-gamma^2*phi0_hat*phi_tilde0_hat)*(1/n1 * trace(H0r) - tr_T02)) ;
    GE_S1S0T02 = 1/n1/gamma/delta_tilde0_hat * trace(C1*H0r) - 1/gamma/delta_tilde0_hat * GE_trS1T02 ;
    A0 = 1/n1 * trace(H1r) -  tr_T12 ;
    A1 = 1/n0 * trace(H0r) -  tr_T02 ;

    # DE of 1/n * trace(Sigma0^2*T1^2)
    B0 = C0^2-1/n0*trace(C0)*C0 ;
    tr_S02T12 = (1/n0 *  trace(B0*H1r^2) - A0 * 1/n0*trace(B0*H1r)/ (1-gamma^2*phi1_hat*phi_tilde1_hat) ) / (1-A0 / (1-gamma^2*phi1_hat*phi_tilde1_hat)) ;
    B1 = C1^2-1/n1*trace(C1)*C1 ;
    tr_S12T02 = (1/n0 *  trace(B1*H0r^2) - A1 * 1/n0*trace(B1*H0r)/ (1-gamma^2*phi0_hat*phi_tilde0_hat) ) / (1-A1 / (1-gamma^2*phi0_hat*phi_tilde0_hat)) ;
    =#
    #= GEs tr(B^2)
    GE_tr_B02 =    (phi0_hat / (1-gamma^2 * phi0_hat*phi_tilde0_hat) * n0/p
        + (tr_S02T12 + gamma^2 / (1-gamma^2*phi1_hat*phi_tilde1_hat) * GE_S0S1T12 * phi_tilde1_hat * GE_S0S1T12) * n1/p
        -2/p/gamma/delta_tilde0_hat * trace(C0*H1r) + 2/p/(gamma*delta_tilde0_hat)^2 * trace(H1r-H0r*H1r) )  ;

    GE_tr_B12 = ( phi1_hat / (1-gamma^2 * phi1_hat*phi_tilde1_hat) * n1/p
        + (tr_S12T02 + gamma^2 / (1-gamma^2*phi0_hat*phi_tilde0_hat) * GE_S1S0T02 * phi_tilde0_hat * GE_S1S0T02) * n0/p
        -2/p/gamma/delta_tilde1_hat * trace(C1*H0r) + 2/p/(gamma*delta_tilde1_hat)^2 * trace(H0r-H1r*H0r) )  ;
    =#

    GE_tr_B02 = ( (1+gamma*delta0_hat)^4 * 1/p * trace(C0*H0r*C0*H0r) - n0/p * delta0_hat^2*(1+gamma*delta0_hat)^2
          + 1/p*trace(C0*H1r*C0*H1r)- n0/p * (1/n0*trace(C0*H1r))^2
          - 2*(1+gamma*delta0_hat)^2*1/p*trace(C0*H0r*C0*H1r) + 2*n0/p * 1/n0*trace(C0*H1r) * delta0_hat * (1+gamma*delta0_hat)  );

    GE_tr_B12 = ( (1+gamma*delta1_hat)^4 * 1/p * trace(C1*H1r*C1*H1r) - n1/p * (delta1_hat^2)*(1+gamma*delta1_hat)^2
          + 1/p*trace(C1*H0r*C1*H0r) - n1/p * (1/n1*trace(C1*H0r))^2
          - 2*(1+gamma*delta1_hat)^2 * 1/p * trace(C1*H1r*C1*H0r) + 2*n1/p * 1/n1*trace(C1*H0r) * delta1_hat * (1+gamma*delta1_hat) );

    #GG1 = (1+gamma*delta1_hat)^4 * 1/p * trace(C1*H1r*C1*H1r) - n1/p * (delta1_hat^2)*(1+gamma*delta1_hat)^2;
    #GG2 =  1/p*trace(C1*H0r*C1*H0r) - n1/p * (1/n1*trace(C1*H0r))^2;
    #GG3 = (1+gamma*delta1_hat)^2 * 1/p * trace(C1*H1r*C1*H0r) - n1/p * 1/n1*trace(C1*H0r) * delta1_hat * (1+gamma*delta1_hat);

    # GE y0
    GE_y0 = ((x0_-x1_)' * H1r* C0 * H1r * (x0_-x1_)) / p ;
    GE_y1 = ((x0_-x1_)' * H0r* C1 * H0r * (x0_-x1_)) / p ;
    GE_tr_B0 = 1/sqrt(p) * trace(C0*H1r) - n0/sqrt(p) * delta0_hat ;
    GE_tr_B1 = -1/sqrt(p) * trace(C1*H0r) + n1/sqrt(p) * delta1_hat ;
    #GE_c0 = (-(sum(log.(abs.(D0r))) - sum(log.(abs.(D1r)))) -  ((x_'*H1r*x_)-delta1_hat-1/n0*trace(C0*H1r)) ) /sqrt(p) ;
    #GE_c1 = (-(sum(log.(abs.(D0r))) - sum(log.(abs.(D1r)))) +  (x_'*H0r*x_)-delta0_hat-1/n1*trace(C1*H0r) ) /sqrt(p) ;
    GE_c0 = (-(log(complex(det(H0r))) - log(complex(det(H1r)))) -  ((x_'*H1r*x_)-delta1_hat-1/n0*trace(C0*H1r)) ) /sqrt(p) ;
    GE_c1 = (-(log(complex(det(H0r))) - log(complex(det(H1r)))) +  (x_'*H0r*x_)-delta0_hat-1/n1*trace(C1*H0r) ) /sqrt(p) ;
    # Final error estimate
    GE_arg0 = real(GE_c0  - GE_tr_B0) / sqrt(abs(2 * GE_tr_B02 + 4*GE_y0));
    GE_arg1 = real(GE_c1  - GE_tr_B1) / sqrt(abs(2 * GE_tr_B12 + 4*GE_y1));
    GE_eps0 = 1-qfunc(GE_arg0);
    GE_eps1 = qfunc(GE_arg1);
    GE_epsilon = n0/(n0+n1) * GE_eps0 + n1/(n0+n1) * GE_eps1;

    return GE_epsilon;
end

# Cross validation estimators for both classifiers

function cross_valid(k, repeat, p, gamma, n0, n1, data0, data1, str)

    pi0 = n0 / (n0 + n1);
    pi1 = n1 / (n0 + n1);

    error_cv_lda = 0;
    error_cv_qda = 0;

    for r = 1:repeat
        if k == 1
            train_pattern0 = [collect(1:n0)];
            train_pattern1 = [collect(1:n1)];
        else
            train_pattern0 = collect(MLBase.Kfold(n0, k));
            train_pattern1 = collect(MLBase.Kfold(n1, k));
        end

        for fold = 1:k

            # Training pattern

            train_index0 = train_pattern0[fold];
            train_index1 = train_pattern1[fold];

            # Testing pattern

            test_index0 = setdiff(1:n0, train_index0);
            test_index1 = setdiff(1:n1, train_index1);

            # Estimation

            u0 = ones(length(train_index0)) / length(train_index0);
            u1 = ones(length(train_index1)) / length(train_index1);

            Y0_train = data0[:, train_index0]; # All training data as columns of C0
            Y1_train = data1[:, train_index1]; # All training data as columns of C1

            x0_ = Y0_train * u0; # sample means
            x1_ = Y1_train * u1; # sample means

            C0 = 1/(length(train_index0)-1)*(Y0_train - x0_ * ones(length(train_index0))')*(Y0_train - x0_ * ones(length(train_index0))')'; # SCM for C0
            C1 = 1/(length(train_index1)-1)*(Y1_train - x1_ * ones(length(train_index1))')*(Y1_train - x1_ * ones(length(train_index1))')'; # SCM for C1
            C = ((length(train_index0)-1) * C0 + (length(train_index1)-1) * C1) / (length(train_index0) + length(train_index1)-2); # common (pooled) SCM for LDA
            Hr = pinv(eye(p) + gamma*C); # regularized inverse SCM
            E0 = eig(eye(p) + gamma*C0);
            E1 = eig(eye(p) + gamma*C1);
            H0r = E0[2] * diagm(ones(p) ./ E0[1]) * E0[2]';
            H1r = E1[2] * diagm(ones(p) ./ E1[1]) * E1[2]';
            D0r = ones(p) ./ E0[1];
            D1r = ones(p) ./ E1[1];

            # Error on testing data or training depending on the value of str

            if str == "test"

                Y0_test = data0[:, test_index0] ; # All testing data as columns of C0
                Y1_test = data1[:, test_index1] ; # All testing data as columns of C1

            elseif str == "train"

                Y0_test = Y0_train ; # All testing data as columns of C0
                Y1_test = Y1_train ; # All testing data as columns of C1
                test_index0 = train_index0 ;
                test_index1 = train_index1 ;

            end

            # LDA and QDA true error

            error_lda = empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, Y0_test, Y1_test, "lda");
            error_qda = empirical_error(x0_, x1_, Hr, pi1, pi0, D0r, D1r, H0r, H1r, Y0_test, Y1_test, "qda");

            error_cv_lda +=  error_lda;
            error_cv_qda += error_qda;
        end
    end

    error_cv_lda = error_cv_lda / k / repeat;
    error_cv_qda = error_cv_qda / k / repeat;
    return error_cv_lda, error_cv_qda ;
end

# Error on the training


# Bootstrap revisited

function Bootstrap_lda_qda(B, p, n0, n1, gamma, data0, data1, x0_, x1_, C0, C1, H0r, H1r, D0r, D1r)

    pi0 = n0 / (n0+n1);
    pi1 = n1 / (n0+n1);

    error_boot_lda = 0;
    error_boot_qda = 0;
    #error_boot_lda1 = 0;
    #error_boot_qda1 = 0;
    for b = 1:B
        # Error on class0
        check = "no";
        while check == "no"
            global bstrp_sample0 = wsample(1:n0,1/n0 * ones(n0), n0); # Bootstrap sample
            global l_out_0 = filter(x -> !(x in bstrp_sample0), 1:n0); # testing set
            if length(l_out_0) >= 1
                check = "yes";
            end
        end

        train_index0 = bstrp_sample0;
         u0 = ones(length(train_index0)) / length(train_index0);
         Y0_train = data0[:, train_index0]; # All training data as columns of C0
         xx0_ = Y0_train * u0; # sample means
         CC0 = 1/(length(train_index0)-1)*(Y0_train - xx0_ * ones(length(train_index0))')*(Y0_train - xx0_ * ones(length(train_index0))')'; # SCM for C0
         #C = ((length(train_index0)-1) * CC0 + (n1-1) * C1) / (length(train_index0) + n1-2); # common (pooled) SCM for LDA
         #Hr = inv(eye(p)+gamma*C); # regularized inverse SCM
         EE0 = eig(eye(p) + gamma*CC0);
         HH0r = EE0[2] * diagm(ones(p)./EE0[1]) * EE0[2]';
         DD0r = ones(p)./EE0[1];
         #HH0r = inv(eye(p)+gamma*CC0);
         #DD0r = eig(HH0r)[1];

        #=
        for i = 1:length(l_out_0)
            x = data0[:, l_out_0[i]];
            # RLDA

            WLDA = (x-0.5*(xx0_+x1_))' * Hr * (xx0_-x1_) - log(pi1/pi0); # linear discriminant using the pooled SCM
            if real(WLDA) < 0
                error_boot_lda0 += 1;
            end

            # RQDA

            qda0 = 0.5 * sum(log.(DD0r)) - 0.5 * (x-xx0_)' * HH0r * (x-xx0_) + log(pi0); # quadratic disc. for C0
            qda1 = 0.5 * sum(log.(D1r)) - 0.5 * (x-x1_)' * H1r * (x-x1_) + log(pi1); # quadratic disc. for C1

            if real(qda0) < real(qda1)
                error_boot_qda0 += 1;
            end
        end
        error_boot_lda0 = error_boot_lda0 / length(l_out_0);
        error_boot_qda0 = error_boot_qda0 / length(l_out_0);
        =#
        # Error on class1

        check = "no";
        while check == "no"
            global bstrp_sample1 = wsample(1:n1, 1/n1 * ones(n1), n1); # Bootstrap sample
            global l_out_1 = filter(x -> !(x in bstrp_sample1), 1:n1); # testing set
            if length(l_out_1) >= 1
                check = "yes";
            end
        end

        train_index1 = bstrp_sample1;
         u1 = ones(length(train_index1)) / length(train_index1);
         Y1_train = data1[:, train_index1]; # All training data as columns of C0
         xx1_ = Y1_train * u1; # sample means
         CC1 = 1/(length(train_index1)-1)*(Y1_train - xx1_ * ones(length(train_index1))')*(Y1_train - xx1_ * ones(length(train_index1))')'; # SCM for C0
         #C = ((length(train_index1)-1) * CC1 + (n0-1) * C0) / (length(train_index1) + n0-2); # common (pooled) SCM for LDA
         #Hr = inv(eye(p)+gamma*C); # regularized inverse SCM
         EE1 = eig(eye(p) + gamma*CC1);
         HH1r = EE1[2] * diagm(ones(p)./EE1[1]) * EE1[2]';
         DD1r = ones(p)./EE1[1];
         #HH1r = inv(eye(p)+gamma*CC1);
         #DD1r = eig(HH1r)[1];

         # Common statistics
         C = ((length(train_index1)-1) * CC1 + (length(train_index0)-1) * CC0) / (length(train_index1) + length(train_index0)-2); # common (pooled) SCM for LDA
         Hr = inv(eye(p)+gamma*C); # regularized inverse SCM

        error_boot_lda += empirical_error(xx0_, xx1_, Hr, pi1, pi0, DD0r, DD1r, HH0r, HH1r, data0[:, l_out_0], data1[:, l_out_1], "lda");
        error_boot_qda += empirical_error(xx0_, xx1_, Hr, pi1, pi0, DD0r, DD1r, HH0r, HH1r, data0[:, l_out_0], data1[:, l_out_1], "qda");
        #=
        for i = 1:length(l_out_1)
            x = data1[:, l_out_1[i]];
            # RLDA

            WLDA = (x-0.5*(x0_+xx1_))' * Hr * (x0_-xx1_) - log(pi1/pi0); # linear discriminant using the pooled SCM
            if real(WLDA) > 0
                error_boot_lda1 += 1;
            end

            # RQDA

            qda0 = 0.5 * sum(log.(D0r)) - 0.5 * (x-x0_)' * H0r * (x-x0_) + log(pi0); # quadratic disc. for C0
            qda1 = 0.5 * sum(log.(DD1r)) - 0.5 * (x-xx1_)' * HH1r * (x-xx1_) + log(pi1); # quadratic disc. for C1

            if real(qda0) > real(qda1)
                error_boot_qda1 += 1;
            end
        end
        =#

        #error_boot_lda1 = error_boot_lda1 / length(l_out_1);
        #error_boot_qda1 = error_boot_qda1 / length(l_out_1);

    end
    # Total error

    error_boot_lda = error_boot_lda / B;
    error_boot_qda = error_boot_qda / B;

    return error_boot_lda, error_boot_qda;
end

# No information error rate

function no_info(Y0_train, Y1_train, n0, n1, x0_, x1_, Hr, H0r, H1r, D0r, D1r, pi1, pi0, discriminant)

    Y_train = [Y0_train Y1_train]; # all training set
    n = n0 + n1;

    aux = 0;
    for i = 1:n
        for j = 1:n
            x = Y_train[:, j];

            # LDA
            if discriminant == "lda"
                WLDA = real((x-0.5*(x0_+x1_))' * Hr * (x0_-x1_) - log(pi1/pi0)); # linear discriminant using the pooled SCM
                if WLDA < 0 && i <= n0
                    aux += 1;
                elseif WLDA >=0 && i > n0;
                    aux += 1;
                end

            #aux += xor(real(complex(isless(WLDA, 0))^2), real(complex(isless(n0, i))^2));

            # QDA
            elseif discriminant == "qda"
                qda0 = 0.5 * sum(log.(abs.(D0r))) - 0.5 * (x-x0_)' * H0r * (x-x0_) + log(pi0); # quadratic disc. for C0
                qda1 = 0.5 * sum(log.(abs.(D1r))) - 0.5 * (x-x1_)' * H1r * (x-x1_) + log(pi1); # quadratic disc. for C1
                if real(qda1) > real(qda0) && i <= n0
                    aux += 1;
                elseif real(qda1) <= real(qda0) && i > n0
                    aux += 1;
                end

                #aux += xor(real(complex(isless(qda1, qda0))^2), real(complex(isless(n0, i))^2));
            end
        end
    end

    aux = aux / n^2;
    return aux;
end

# ......................... Determininstic equivalents ............................

# RLDA

function DE_lda(Sigma0, Sigma1, mu, p, n0, n1, gamma)

    # Computing some statistics

    n = n0 + n1;
    pi0 = n0 / n;
    pi1 = n1 / n;
    a = p/n;
    c0 = n0/n;
    c1 = n1/n;
    n_s = [c0 c1];
    g0 = 1;
    g1 = 1;
    g_tilde0 = 1;
    g_tilde1 = 1;
    erreur0 = 1;
    erreur1 = 1;
    z = -n/p/gamma ;
    while erreur0 > 1e-9 && erreur1 > 1e-9
        a0 = g0;
        a1 = g1;
        g_tilde0 = -1/z * 1/p * trace(Sigma0 * inv(eye(p) + c0*a0*Sigma0 + c1*a1*Sigma1));
        g_tilde1 = -1/z * 1/p * trace(Sigma1 * inv(eye(p) + c0*a0*Sigma0 + c1*a1*Sigma1));
        g0 = -1/z / a / (1+g_tilde0);
        g1 = -1/z / a / (1+g_tilde1);
        erreur0 = abs(a0-g0)^2;
        erreur1 = abs(a1-g1)^2;
    end
    Q = -1/z * pinv(eye(p) + c0*g0*Sigma0 + c1*g1*Sigma1);
    R = zeros(2,2);
    g = [g0 g1];
    for i = 1:2
        for j = 1:2
            R[i,j] = n_s[i]*z^2*g[i]^2 * 1/n * trace(Sigma0*Q*Sigma1*Q) / (1 - n_s[1]*z^2*g[1]^2 * 1/n * trace(Sigma0*Q*Sigma1*Q)- n_s[2]*z^2*g[2]^2 * 1/n * trace(Sigma0*Q*Sigma1*Q) );
        end
    end
    QQ0 = Q*Sigma0*Q + R[1,1]*Q*Sigma0*Q + R[2,1]*Q*Sigma1*Q;
    QQ1 = Q*Sigma1*Q + R[1,2]*Q*Sigma0*Q + R[2,2]*Q*Sigma1*Q;

    # Computing the error

    GG0 = -1/2 * z * mu' * Q * mu + z/2/n0 * trace(Sigma0 * Q) - z/2/n1 * trace(Sigma1 * Q);
    GG1 = 1/2 * z * mu' * Q * mu + z/2/n0 * trace(Sigma0 * Q)  - z/2/n1 * trace(Sigma1 * Q);
    DD0 = z^2 * mu' * QQ0 * mu + z^2 /n0 * trace(Sigma0 * QQ0)+ z^2/n1 * trace(Sigma1 * QQ0);
    DD1 = z^2 * mu' * QQ1 * mu + z^2/n0 * trace(Sigma0 * QQ1)  + z^2/n1 * trace(Sigma1 * QQ1);
    arg0 = (-GG0 + log(pi1/pi0))/sqrt(DD0);
    arg1 = (GG1 - log(pi1/pi0))/sqrt(DD1);
    DE_lda = pi0 *(1-qfunc(arg0)) + pi1 *(1-qfunc(arg1)) ;

    return DE_lda;
end

# Computing δ for RQDA as in the paper δ = 1/n tr ΣT

function solve_delta(n, gamma, s) # s is the vector of eigenvalues of Σ

    delta = 1;
    erreur = 1;
    while erreur > 1e-9
        a = delta;
        delta = 1/n * sum(s./(1+gamma*s/(1+gamma*a)));
        erreur = abs(a-delta)^2;
    end
    delta_tilde = 1 / (1+gamma*delta);

    return delta, delta_tilde;
end

# RQDA

function DE_qda(gamma, n0, n1, p, Sigma0, Sigma1, U0, s0, U1, s1, mu, Sigma_r0, Sigma_r1)

    del0 = solve_delta(n0, gamma, s0);
    del1 = solve_delta(n1, gamma, s1);
    delta0 = del0[1];
    delta_tilde0 = del0[2];
    delta1 = del1[1];
    delta_tilde1 = del1[2];
    T0 = U0 * diagm(ones(p)./(ones(p)+gamma*delta_tilde0*s0)) * U0';
    T_tilde0 = 1 / (1+gamma*delta0) * eye(n0);
    T1 = U1 * diagm(ones(p)./(ones(p)+gamma*delta_tilde1*s1)) * U1';
    T_tilde1 = 1 / (1+gamma*delta1) * eye(n1);

    phi0 = 1/n0 * trace(Sigma0^2 * T0^2);
    phi_tilde0 = 1/n0 * trace(T_tilde0^2);
    phi1 = 1/n1 * trace(Sigma1^2 * T1^2);
    phi_tilde1 = 1/n1 * trace(T_tilde1^2);
    #B0 = Sigma_r0 * (T1-T0) * Sigma_r0;
    #B1 = Sigma_r1 * (T1-T0) * Sigma_r1;

    # DEs class 0


    DE_tr_B02 = (phi0 / (1-gamma^2 * phi0*phi_tilde0) * n0/p
    + (1/n1 * trace(T1^2 * Sigma0^2) + gamma^2 / (1-gamma^2*phi1*phi_tilde1) * 1/n1 * trace(Sigma0*Sigma1*T1^2) * phi_tilde1 * 1/n1 * trace(Sigma0*Sigma1*T1^2)) * n1/p
    -2/p * trace(Sigma0 * T0 * Sigma0 * T1 ) );


    DE_tr_B0 = 1 / sqrt(p) * trace(Sigma0 * (T1-T0));
    DE_c0 = (1 / sqrt(p) * log((1+gamma*delta0)^n0/(1+gamma*delta1)^n1) + 1 /sqrt(p) * (sum(log.(1+gamma*delta_tilde0*s0))-sum(log.(1+gamma*delta_tilde1*s1)))
    + 1/sqrt(p) * gamma * (n1*delta1 * delta_tilde1 - n0*delta0 * delta_tilde0) );
    DE_y0 = 1/p * mu.'*Sigma1*T1^2*mu  / (1-gamma^2*phi1*phi_tilde1);
    DE_arg0 = (DE_c0 - (mu.'*T1*mu)/sqrt(p) - DE_tr_B0) / sqrt(2 * DE_tr_B02  + 4*DE_y0);
    DE_eps0 = 1-qfunc(DE_arg0);

    # DEs class 1

    DE_tr_B12 = ( phi1 / (1-gamma^2 * phi1*phi_tilde1) * n1/p
    + (1/n0 * trace(T0^2 * Sigma1^2) + gamma^2 / (1-gamma^2*phi0*phi_tilde0) * 1/n0 * trace(Sigma1*Sigma0*T0^2) * phi_tilde0 * 1/n0 * trace(Sigma1*Sigma0*T0^2)) * n0/p
    -2/p * trace(Sigma1 * T0 * Sigma1 * T1 ) );

    DE_tr_B1 = 1 / sqrt(p) * trace(Sigma1 * (T1-T0));
    DE_c1 = DE_c0;
    DE_y1 = 1/p * mu.'*Sigma0*T0^2*mu  / (1-gamma^2*phi0*phi_tilde0);
    DE_arg1 = (DE_c1 + (mu.'*T0*mu)/sqrt(p) - DE_tr_B1) / sqrt(2 * DE_tr_B12 + 4*DE_y1);
    DE_eps1 = qfunc(DE_arg1);

    DE_epsilon = n0/(n0+n1) * DE_eps0 + n1/(n0+n1) * DE_eps1;

    return DE_epsilon;

end

#=

Pkg.add("Convex")
Pkg.add("SCS")
using SCS
using Convex

function optimize_RLDA(n0, n1, x0_, x1_, Hr, C0, C1)

    x = Variable();
    expr = LDA_General_estim(n0, n1, x0_, x1_, Hr, C0, C1, x);
    problem = minimize(expr, x >=0);
    solve!(problem);
    gamma_opt = evaluate(x);
    return gamma_opt, LDA_General_estim(n0, n1, x0_, x1_, Hr, C0, C1, x);

end
=#

# Optimizing the G estimators

#Pkg.add("Optim");
using Optim;

function optimize_DA(p, n0, n1, x0_, x1_, s0, s1, U0, U1, s, U, C0, C1)

    optim_fun_lda(x) = LDA_General_estim(n0, n1, x0_, x1_, C0, C1, x[1]);
                        
    optim_fun_qda(x) = QDA_General_estim(p, n0, n1, x0_, x1_, s0, s1, U0, U1, C0, C1, x[1]);

    output_lda = optimize(optim_fun_lda, [2.5,], GradientDescent()); # or GradientDescent()
    gamma_opt_lda = output_lda.minimizer;
    gamma_opt_lda = gamma_opt_lda[1];
    min_val_lda = output_lda.minimum;

    output_qda = optimize(optim_fun_qda, [2.5,], GradientDescent());
    gamma_opt_qda = output_qda.minimizer;
    gamma_opt_qda = gamma_opt_qda[1];
    min_val_qda = output_qda.minimum;
    return gamma_opt_lda[1], min_val_lda, gamma_opt_qda[1], min_val_qda;
end

#= Constrained optimization

Pkg.add("JuMP")
Pkg.add("NLopt")
using JuMP
using NLopt
function optimize_RQDA(p, n0, n1, x0_, x1_, s0, s1, U0, U1, C0, C1)

    optim_fun_qda(x) = QDA_General_estim(p, n0, n1, x0_, x1_, s0, s1, U0, U1, C0, C1, x)
    m = Model(solver=NLoptSolver(algorithm=:LD_MMA));
    JuMP.register(m, :optim_fun, 1, optim_fun, autodiff=true); # register optim_fun
    @variable(m, x >= 0);
    #@constraint(m, gamma>=0);
    @NLobjective(m, Min, optim_fun(x));
    solve(m);
    gamma_opt = getvalue(gamma);
    obj = getobjectivevalue(m);
    return gamma_opt, obj;
end
=#
#=
optim_fun(gamma) = (gamma-5)^2 * (gamma+1);
m = Model(solver=NLoptSolver(algorithm=:LD_MMA));
JuMP.register(m, :optim_fun, 1, optim_fun, autodiff=true); # register optim_fun
@variable(m, gamma >=0);
#@constraint(m, gamma>=0);
@NLobjective(m, Min, optim_fun(gamma));
solve(m);
gamma_opt = getvalue(gamma)
obj = getobjectivevalue(m)
=#

# Convex optimization

#Pkg.add("Convex")
#Pkg.add("SCS")

using Convex
using SCS
#=
function optimize_RLDA(n0, n1, x0_, x1_, Hr, C0, C1)

    optim_fun(gamma) = LDA_General_estim(n0, n1, x0_, x1_, Hr, C0, C1, gamma);
    x = Variable(1);
    problem = minimize(optim_fun(x), [x >= 0])
    solve!(problem);

    return problem.optval;
end
=#




# Preparing MNIST data

#Pkg.add("MNIST");
using MNIST;

train_mat = traindata(); # first element of the tuple 784*6e4, second 6e4 labels
test_mat = testdata(); # .......................................................

# Define a function on MNIST that will recover train and test data for two given digits

function MNIST_data(class0, class1)

    # class0 training and testing

    label0_train = find(x->(x == class0), train_mat[2]); # lables of class0 in training
    label0_test = find(x->(x == class0), test_mat[2]); # lables of class0 in testing
    #train0 = (train_mat[1])[:, label0_train];
    #test0 = (test_mat[1])[:, label0_test];

    # class1 training and testing

    label1_train = find(x->(x == class1), train_mat[2]); # lables of class1 in training
    label1_test = find(x->(x == class1), test_mat[2]); # lables of class1 in testing
    #train1 = (train_mat[1])[:, label1_train];
    #test1 = (test_mat[1])[:, label1_test];

    # Normalizing the data to have finite norms for sample covariances and means

    p = size(train_mat[1])[1];
    #train0 = train0 / sqrt(sum(mean(train0.^2, 2)));
    #train1 = train1 / sqrt(sum(mean(train1.^2, 2)));
    training = train_mat[1];
    testing = test_mat[1];
    training = training * sqrt(p) / sqrt(sum(mean(training.^2, 2)));
    testing = testing  * sqrt(p) / sqrt(sum(mean(testing.^2, 2)));

    train0 = (training)[:, label0_train];
    test0 = (testing)[:, label0_test];

    train1 = (training)[:, label1_train];
    test1 = (testing)[:, label1_test];

    #tes0 = test0 / sqrt(sum(mean(test0.^2, 2)));
    #test1 = test1 / sqrt(sum(mean(test1.^2, 2)));
    #return train0[:, 1:1200], test0, train1[:, 1:1200], test1;
    return train0, test0, train1, test1;
end

# Preparing the USPS data (US postal data)
using MAT

# training data

function USPS_Data(class0, class1)

    file_train = matopen("/home/elkhalk/Dropbox/BLUE_1/Big_Data_Regression/Classification/Julia\ codes/USPS_train.mat");
    file_test = matopen("/home/elkhalk/Dropbox/BLUE_1/Big_Data_Regression/Classification/Julia\ codes/USPS_test.mat");
    train_data = read(file_train, "USPS_train");
    test_data = read(file_test, "USPS_test");

    # Pre-processing
    p = size(train_data')[1];
    A_train = train_data[:, 2:end];
    A_test = test_data[:, 2:end];
    A_train = A_train* sqrt(p) / sqrt(sum(mean(A_train'.^2, 2)));
    A_test = A_test * sqrt(p) / sqrt(sum(mean(A_test'.^2, 2)));

    # training data
    label0_train = find(x->(x == class0), train_data[:,1]);
    label1_train = find(x->(x == class1), train_data[:,1]);
    train0 = A_train[label0_train, :];
    train1 = A_train[label1_train, :];

    # testing data
    label0_test = find(x->(x == class0), test_data[:,1]);
    label1_test = find(x->(x == class1), test_data[:,1]);
    test0 = A_test[label0_test, :];
    test1 = A_test[label1_test, :];

    return train0.', test0.', train1.', test1.';
end
 # Preparing the sonar data

 # every time this function is called will randomly generate a pattern for training and testing
 using MAT
function Sonar_data(n0, n1)

     file = matopen("/home/elkhalk/Dropbox/BLUE_1/Big_Data_Regression/Classification/Julia\ codes/sonar_data.mat");
     sonar_data = read(file, "data"); # a matrix 208x61
     (n, ~) = size(sonar_data);
    rock = zeros(97, 60);
    mines = zeros(111, 60);
    for ii = 1:97
        rock[ii, :] = sonar_data[ii, 1:60];
    end
    for ii = 1:111
        mines[ii, :] = sonar_data[ii+97, 1:60];
    end
    p = 60;
    rock = rock.';
    mines = mines.';

    # training pattern
    train_pattern0 = sample(1:97, n0;replace=false);
    train_pattern1 = sample(1:111, n1;replace=false);

    # testing pattern
    test_pattern0 = setdiff(1:97, train_pattern0);
    test_pattern1 = setdiff(1:111, train_pattern1);

    train0 = rock[:, train_pattern0];
    test0 = rock[:, test_pattern0];

    train1 = mines[:, train_pattern1];
    test1 = mines[:, test_pattern1];

    return train0, test0, train1, test1;
end

# Preparing the Breast cancer data wdbc.mat

function wdbc()
    file = matopen("/home/elkhalk/Dropbox/BLUE_1/Big_Data_Regression/Classification/Julia\ codes/wdbc.mat");
    wdbc_data = read(file, "wdbc");
    (N, ncol) = size(wdbc_data);
    p = ncol-2;
    class0 = [Array{Float64}(0, p);]
    class1 = Array{Float64}(0, p);
    for i = 1:N
        if wdbc_data[i, 2] == 1
            class0 = [class0, wdbc_data[i, 3:ncol]];
        else
            class1 = [class1, wdbc_data[i, 3:ncol]];
        end
    end
    return class0, class1;
end

# Two-stage optimization: 1) get the optimal \gamma from G estimation, 2) perform a grid search using 5-fold cross validation

function two_stage_cv(p, n0, n1, train_set0, train_set1, gamma_g_lda, gamma_g_qda, points)

    # gamma_g is the optimal gamma from G estimation
    # points is the # of search points on the grid

    range_lda = linspace(max(0, gamma_g_lda - 2/sqrt(p)), gamma_g_lda + 2/sqrt(p), points); # the grid on which we are going to perform the search
    range_qda = linspace(max(0, gamma_g_qda - 2/sqrt(p)), gamma_g_qda + 2/sqrt(p), points); # the grid on which we are going to perform the search

    error_search = map(1:points) do i
        e_lda = cross_valid(5, 1, p, range_lda[i], n0, n1, train_set0, train_set1, "test");
        e_qda = cross_valid(5, 1, p, range_qda[i], n0, n1, train_set0, train_set1, "test");
        return e_lda[1], e_qda[2];
    end

    error_search = tuple_to_array(error_search, 2);
    gamma_index_lda = findmin(error_search[:, 1])[2];
    gamma_index_qda = findmin(error_search[:, 2])[2];
    gamma_opt_lda = range_lda[gamma_index_lda];
    gamma_opt_qda = range_qda[gamma_index_qda];

    return gamma_opt_lda, gamma_opt_qda;
end

# Two-stage optimization: 1) get the optimal \gamma from G estimation, 2) perform a grid search using testing data

function two_stage_test(p, n0_test, n1_test, x0_, x1_, pi0, pi1, s, U, s0, U0, s1, U1, test0, test1, gamma_g_lda, gamma_g_qda, points)
    range_lda = linspace(gamma_g_lda - 2/sqrt(p), gamma_g_lda + 2/sqrt(p), points); # the grid on which we are going to perform the search
    range_qda = linspace(gamma_g_qda - 2/sqrt(p), gamma_g_qda + 2/sqrt(p), points); # the grid on which we are going to perform the search

    error_search = map(1:points) do i
        Hr = U * diagm(ones(p) ./ (ones(p) + range_lda[i] * s)) * U'; # Hr with the RLDA regularizer
        e_lda = empirical_error(x0_, x1_, Hr, pi1, pi0, [], [], [], [], test0, test1, "lda");

        D0r = ones(p) ./ (ones(p) + range_qda[i] * s0);
        D1r = ones(p) ./ (ones(p) + range_qda[i] * s1);
        H0r = U0 * diagm(D0r) * U0';
        H1r = U1 * diagm(D1r) * U1';
        Hr = U * diagm(ones(p) ./ (ones(p) + range_qda[i] * s)) * U'; # Hr with RQDA regularizer
        e_qda = qda_true(p, n0_test, n1_test, [], [], [], [], D0r, D1r, H0r, H1r, Hr, x0_, x1_, pi0, pi1, "real", test0, test1);
        return e_lda, e_qda;
    end
    error_search = tuple_to_array(error_search, 2);
    gamma_index_lda = findmin(error_search[:, 1])[2];
    gamma_index_qda = findmin(error_search[:, 2])[2];
    gamma_opt_lda = range_lda[gamma_index_lda];
    gamma_opt_qda = range_qda[gamma_index_qda];

    err_opt_lda = findmin(error_search[:, 1])[1];
    err_opt_qda = findmin(error_search[:, 2])[1];
    #err_opt_lda = 0;
    #err_opt_qda = 0;
    return gamma_opt_lda, err_opt_lda, gamma_opt_qda, err_opt_qda;
end

# Least Squares support vector machines (LS-SVM)
using LIBSVM;

function ls_svm(mu0, mu1, Sigma_r0, Sigma_r1, train0, train1, n0_test, n1_test, test0, test1, type_eval)

    # training the LS-SVM classifier
    (p, n0) = size(train0);
    (p, n1) = size(train1);

    # Generating some testing data

    if type_eval == "synthetic"
        Z0 = randn(p, n0_test); # training data for C0
        Z1 = randn(p, n1_test); # training data for C1
        test0 = mu0 * ones(n0_test)' + Sigma_r0 * Z0; # All training data as columns of C0
        test1 = mu1 * ones(n1_test)' + Sigma_r1 * Z1; # All training data as columns of C0
    end

    (p, n0_test) = size(test0);
    (p, n1_test) = size(test1);

    labels_train = [zeros(n0); ones(n1)];
    train = [real(train0) real(train1)]; # the whole training set
    test = [real(test0) real(test1)]; # the whole testing setss
    model = svmtrain(train, labels_train; 0, 1, 3, 1, 1); # SVM classifier
    (predicted_labels, decision_values) = svmpredict(model, test); # SVM predicted labels
    labels_test = [zeros(n0_test); ones(n1_test)]; # testing labels
    svm_error =  1 - mean((predicted_labels .== labels_test)); # SVM error rate

    return svm_error;

end

# PCA procedure
