function sm = vcircshift(m, n, wrap)

    N = length(m);
    sm = circshift(m, [n, 0]);
    
    if ~wrap
        if n > 0
            sm = sm .* [zeros(n,N); ones(N-n,N)];
        else
            sm = sm .* [ones(N+n,N); zeros(-n,N)];
        end
    end