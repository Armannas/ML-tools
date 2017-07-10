function [ labels_estimate ] = least_squares(Xtrn_LS, Ytrn,Xtst_LS, Nclass)
%   Labels unlabeled samples using label propagation
%
%   Yu_labeled = label_propagation(Xtrn, Xtst, Ytrn, Nclass sigma, iters)
%
%  INPUT
%    Xrn_LS           Labeled train set. Nxd dataset(train samples x features)
%    Ytrn             Labels of train set. Nx1 vector with arbitrary amount
%                     of labels
%    Xtst             Unlabeled test set. Mxd dataset(test samples x features)
%    Nclass           Number of classes
%   
%  OUTPUT
%    labels_estimate  labels of test set estimated Mx1

% Author: Arman Nassiri, https://github.com/Armannas,
% arman.a.nassiri@gmail.com
X_all = [Xtrn_LS;Xtst_LS];

% Normalize dataset
Xtrn_norm = ( Xtrn_LS - mean(X_all(:)) ) / std(X_all(:));
Xtst_norm = ( Xtst_LS - mean(X_all(:)) ) / std(X_all(:));

%Regularize to prevent singularity
%Xtrn_norm = Xtrn_norm + (eye(size(Xtrn_norm,1),size(Xtrn_norm,2))*1e-4);

% Initialize self-learning labels to be Ytrain and start fitting on train
% set. 
Y_SL = Ytrn;
X = Xtrn_norm;
label_diff = 1;
labels_estimate = zeros(length(Xtst_norm),1);
% Repeat until labels do not change anymore
while label_diff    
    Y_SL_class = [];
    theta = [];
    y_new = [];
    Ytrn_OvA = [];
    for i = 1:Nclass
        % To prevent Ytrn from changing. Put in buffer variable that
        % assigns 1 for the label to be classified and -1 for others (one vs
        % all).
        Ytrn_OvA = Y_SL;
        
        % Remap labels for every classifier. 1 if part of class that classifier is
        % trying to predict, otherwise -1(One-vs-All).
        Ytrn_OvA(Y_SL==i)=1;
        Ytrn_OvA(Y_SL~=i)=-1;
        Y_SL_class(:,i) = Ytrn_OvA;
        
        % Find least squares fitting
        theta(:,i) = X \ Y_SL_class(:,i);
        % Apply fitting on test set
        y_new(:,i) = theta(:,i)'*Xtst_norm';
    end
    
    labels_estimate_old = labels_estimate;
    % Choose label from classifier with maximum confidence.
    for i = 1:length(y_new)
        labels_estimate(i) = find(y_new(i,:)==max(y_new(i,:)), 1, 'last' );
    end
    
    % Add estimated labels to known ones. Now that all samples have labels,
    % we can fit on the entire dataset starting next iteration.
    Y_SL = [Ytrn;labels_estimate];
    X = [Xtrn_norm;Xtst_norm];
    label_diff = sum(labels_estimate_old ~= labels_estimate);
    fprintf('Difference: %i\n', label_diff)
end

end
