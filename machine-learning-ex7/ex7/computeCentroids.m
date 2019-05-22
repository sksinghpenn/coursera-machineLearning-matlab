function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

idx_size = size(idx,1);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% matrix for points associated to each centroid

cell = {};

for i = 1:K
    cell{i} = zeros(1,n);
end;



for i = 1:idx_size
    centroid_number = idx(i,1);
    
    for j = 1:K
        if (centroid_number == j)
            cell{j} = [cell{j}; X(i,:)]
            break;
        end;
    end;
    
end;

disp(cell);

for i = 1:K
    %for j = 1:n
     %   disp(mean(cell{i},j));
      %  disp(centroids(i,j)) ;
        centroids(i,:) = (sum(cell{i})/(size(cell{i},1)-1));
   
end;

end

