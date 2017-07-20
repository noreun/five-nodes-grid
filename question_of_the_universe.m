
%% 1

N=6;
all = zeros(2^N, N);
for s=0:2^N-1
    all(s+1,:) = de2bi(s, N, 'left-msb');
end

sym=false(1,2^N);
for i=size(all,1):-1:(2^(N/2)+1)
    s = all(i,:);
    for j=i-1:-1:1
        is = fliplr(all(j,:));
        if sum(abs(s-is)) == 0 && ~sym(j) && sum(s) > 0 && sum(s) < N 
            sym(i) = true;
            fprintf('found %d (%d)\n', i-1, j-1)
            break
        end
    end
end

fprintf('\n')
    
nsym = find(sym);
for insym = nsym
    fprintf('%d : ', insym-1)
    fprintf('%d ', all(insym,:))
    fprintf('\n')
end

size(all(~sym,:))

%% # 2

TN = 16;
Ns = zeros(1,TN-1);

for N=2:TN

    fprintf('checking %d...\n', N)
    
    all = [];
    for i = 0:2^N-1
        n = de2bi(i, N, 'left-msb');
        rn = fliplr(n);
        ri = bi2de(rn, 'left-msb');
        if any(all == ri)
            continue
        end
        all = [all i];
    end

    % for insym = 1:length(all)
    %     i = all(insym);
    %     s = de2bi(i, N, 'left-msb');
    %     fprintf('%d : ', i)
    %     fprintf('%d ', s)
    %     fprintf('\n')
    % end


    Ns(N-1) = length(all);
end

%%
figure;
plot(2:TN, 2:TN, '-b')
hold on
plot(2:TN, log2(Ns), '*-k')

