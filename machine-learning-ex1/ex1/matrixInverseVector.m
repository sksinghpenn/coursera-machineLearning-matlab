function A_inv_b = matrixInverseVector(A,b,x_init, alpha)

x = x_init
disp('shashi')
while(true),
  	Ax = A*x
  	sqlErrors = (Ax - b).^2; % sqaured error

	cost =  sum(sqlErrors);
	if cost < 10^-6,
           break
        else
	  AxMinusB = Ax - b
          twoA = 2 * A
          derivativeOfFx = twoA .* AxMinusB
	  sumOfDerivative = sum(derivativeOfFx)
          alphaDerivative = alpha * sumOfDerivative 
          disp(alphaDerivative)
          x = x - alphaDerivative 
        endif
endwhile
disp(x)
endfunction
