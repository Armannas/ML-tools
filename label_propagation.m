function Yu_labeled = label_propagation(Xtrn, Xtst, Ytrn, Nclass, sigma)
% LABEL_PROPAGATION  Labels unlabeled samples using label propagation
%
%   Yu_labeled = label_propagation(Xtrn, Xtst, Ytrn, Nclass sigma, iters)
%
%  INPUT
%    Xtrn           Labeled train set. Nxd dataset(train samples x features)
%    Xtst           Unlabeled test set. Nxd dataset(test samples x features)
%    Ytrn           Labels of train set. Nx1 vector with arbitrary amount
%                   of labels
%    Nclass         Number of classes
%    sigma          Sigma hyperparameter. Sigma=0 causes algorithm to only
%                   take into account nearest neighbor  for labeling 
%                   unlabeled sample and for sigma << 1, take into account
%                   a large amount of surrounding labeled samples
%                   (default:25)
%   
%  OUTPUT
%    Yu_labeled     labels of test set estimated by LP Mxd(test samples x features)

% ref: X. Zhu, Z. Ghahramani, "Learning from Labeled and Unlabeled data
% with Label Propagation", Technical Report CMU-CALD-02-107",2002,vol.951.

% Author: Arman Nassiri, https://github.com/Armannas,
% arman.a.nassiri@gmail.com

    if (nargin < 5), sigma = 25;  end

    % Construct labeled part of Y probability matrix.
    % The probability of a certain sample(row) belonging to a certain
    % class(column). For Y_L, we know the class label, so the corresponding
    % column is probability 1 and the rest of the row is 0. In other words,
    % the Y matrix is one-hot encoded.
    
    % Labels cannot be smaller than 1 as the label number is also used to
    % set the corresponding column. 
    
    % Could be solved as following. If class numbers start lower than 1
    % offset all labels by the smallest class number and add one to start
    % at class number 1
    % if (min(Ytrn) < 1)
    %    Ytrn = Ytrn+abs(min(Ytrn))+1;
    %
    Y_L = zeros(length(Ytrn),Nclass);

    for i = 1:length(Ytrn)
        Y_L(i,Ytrn(i)) = 1;
    end

    %% Construct W matrix
    disp('Constructing W matrix')

    tic

    % Pair-wise distance between all samples.
    W = pdist2([Xtrn;Xtst],[Xtrn;Xtst]);
    W = exp(-W.^2/sigma); % RBF kernel
    disp('Finished W matrix')
    toc
    %% Construct transition matrix
    disp('Constructing transition matrix')
    % Construct probabilistic transition matrix
    tic
    T = zeros(length(W),length(W));
    for j = 1:length(T)
        % The normalization by the sum over all rows is very expensive.
        % Move row wise instead of column wise
        % (change rows each iteration and then move a column step)
        % so that the sum of row elements can be recycled over many
        % iterations
        t_sum = sum(W(:,j));
        % No need to calculate entire transition matrix,only Tul and Tuu
        % i.e. unlabeled rows(starting at n_labeled+1) and labeled+unlabeled columns(entire length)
        for i = length(Xtrn)+1:length(T)
    %           Slow version
    %             t_sum = 0;
    %             for k = 1:(n_labeled+n_unlabeled)
    %                 t_sum = t_sum + W(k,j);
    %             end
            T(i,j) = W(i,j)/t_sum;
        end
    end
    disp('Finished transition matrix')
    disp('')
    toc

    %% Normalize transition matrix
    disp('Constructing normalized transition matrix')
    tic
    T_bar = zeros(length(W),length(W));
     % No need to normalize entire trans. matrix, only Tul and Tuu
     % i.e. unlabeled rows (starting at n_labeled+1) and labeled+unlabeled columns(entire length)
    for i = length(Xtrn)+1:length(T_bar)
        % The normalization by the sum over all columns is also 
        % very expensive.
        % No need to calculate the column normalization each time.
        % Only when the row changes a new column sum is encountered.

        t_bar_sum = sum(T(i,:));
        for j = 1:length(T_bar)

    %             t_bar_sum = 0;
    %             for k = 1:(n_labeled+n_unlabeled)
    %                 t_bar_sum = t_bar_sum + T(i,k);
    %             end
            T_bar(i,j) = T(i,j)/t_bar_sum;
        end
    end

    % Normalized Tul and Tuu
    T_bar_UU = T_bar(length(Xtrn)+1:length(Xtrn)+length(Xtst),length(Xtrn)+1:length(Xtrn)+length(Xtst));
    T_bar_UL = T_bar(length(Xtrn)+1:length(Xtrn)+length(Xtst),1:length(Xtrn));
    disp('Finished normalized transition matrix')
    disp('')
    toc

    %% Estimate unlabeled data
    disp('Estimate unlabeled data')
    tic
    
    % Initialisation unlabeled part Y matrix arbitrary
    Y_U = ones(length(Xtst),Nclass);
    % Keeps track of difference of labels between iterations.
    % Iterate until no difference in labels between two iterations.
    label_diff = 1;
    % Iteratively estimate unlabeled data
    while label_diff
        Y_U_old = Y_U;
        Y_U = T_bar_UU*Y_U + T_bar_UL*Y_L;
        label_diff = sum(Y_U ~= Y_U_old);
    end
    toc
    %% Decode probabilities to class labels
    % Vector with estimated labels from Label Propagation
    Yu_labeled = zeros(length(Y_U),1);
    % Decode unlabeled data
    for i = 1:length(Y_U)
        Yu_labeled(i) = find(Y_U(i,:)==max(Y_U(i,:)), 1, 'last' );
    end
end