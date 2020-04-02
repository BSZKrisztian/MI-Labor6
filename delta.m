
function B = DeltaRule(X,Y,LearningRate,MinimumWeightChange,MaximumPasses,B0)

[n m] = size(X);

if ( (nargin < 6) || (isnan(B0(1))) )
    B = 0.01 * randn(m+1,1);
else
    B = B0;
end;


Pass = 1;
OldB = B + 1e10;


while ( (Pass <= MaximumPasses) && (norm(B - OldB) >= MinimumWeightChange) )
    
    OldB = B;
    
    R = randperm(n);
    
    for Exemplar = 1:n
        
        ShuffledExemplar = R(Exemplar);

        ModelOutput = Logistic(B(1) + X(ShuffledExemplar,:) * B(2:end));
        
        B = B + LearningRate * (Y(ShuffledExemplar) - ModelOutput) * [1 X(ShuffledExemplar,:)]';
    end
        
    Pass = Pass + 1;
end



