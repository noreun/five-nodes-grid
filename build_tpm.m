function [tpm, H] = build_tpm(cm, thresh, t)

    N = size(cm, 2);
    nstates = 2^N;
    H = zeros(nstates, N);
    tpm = zeros(nstates, N);
    for i = 0:nstates-1
        state = de2bi(i, N)';    
        for j = 1:size(cm, 2)
            e = sum(cm(:, j) .* state) - (thresh+1)/2;
            H(i+1, j) = e;
            tpm(i+1, j) = prob(e, t);
        end
    end
end

function p = prob(e, t)
    if t == 0
        if e > 0
            p = ones(size(e));
        elseif e < 0
            p = zeros(size(e));
        else
            p = .5 * ones(size(e));
        end
    else
        p = 1 ./ ( 1 + exp( -(1/t) .* e) );
    end
end