function tpm = build_tpm_step(cm, b)

    N = size(cm, 2);
    nstates = 2^N;
    tpm = zeros(nstates, N);
    t = ceil(sum(cm)/2);
    for i = 0:nstates-1
        state = de2bi(i, N)';    
        for j = 1:size(cm, 2)
            e = sum(cm(:, j) .* state);
            if e > t(j)
                tpm(i+1, j) = (1/b);
            elseif e < t(j)
                tpm(i+1, j) = 1 - (1/b);
            else
                tpm(i+1, j) = .5;
            end
%             if e > t(j)
%                 tpm(i+1, j) = (1/b);
%             else
%                 tpm(i+1, j) = 1 - (1/b);
%             end
        end
    end
end
