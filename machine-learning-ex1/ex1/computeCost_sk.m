clear
data = load('landprice_data.txt')

X = data(:,1);
y= data(:,2);
theta = [0; 0];
m = length(X);
ones = ones(m,1);
X = [ones,X];


%disp(X);
%disp(theta);

size(X);
size(theta);
h = X*theta;
%disp(h);
hDiffActual = h - y;
hDiffActualSquare = hDiffActual.^2;
hDiffActualSquareSum = sum(hDiffActualSquare);
J = hDiffActualSquareSum/(2*m);
disp(J)