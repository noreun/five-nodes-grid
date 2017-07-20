% Build system

N = 12;
thresh = 2;

% cm5 = [0  0  0  0  0  1  0  0  0  0  0  0  0  0 ;...
%        0  0  0  0  0  0  1  0  0  0  0  0  0  0 ;...
%        0  0  0  0  0  0  0  1  0  0  0  0  0  0 ;...
%        0  0  0  0  0  0  0  0  1  0  0  0  0  0 ;...
%        0  0  0  0  0  0  0  0  0  1  0  0  0  0 ;...
%        
%        0  0  0  0  0  1  1  0  0  1  1  1  0  1 ;...
%        0  0  0  0  0  1  1  1  0  0  1  1  1  0 ;...
%        0  0  0  0  0  0  1  1  1  0  0  1  1  1 ;...
%        0  0  0  0  0  0  0  1  1  1  1  0  1  1 ;...
%        0  0  0  0  0  1  0  0  1  1  0  1  1  1 ;...
%        
%        0  0  0  0  0 -2 -2 -2 -2  0  0  0  0  0 ;...
%        0  0  0  0  0  0 -2 -2 -2 -2  0  0  0  0 ;...
%        0  0  0  0  0 -2 -2 -2  0 -2  0  0  0  0 ;...      
%        0  0  0  0  0  0  0  0  0  0  0  0  0  0 ];

% cm63 = [0  0  0  0  0  0  1  0  0  0  0  0  0  0  0 ;...
%        0  0  0  0  0  0  0  1  0  0  0  0  0  0  0 ;...
%        0  0  0  0  0  0  0  0  1  0  0  0  0  0  0 ;...
%        0  0  0  0  0  0  0  0  0  1  0  0  0  0  0 ;...
%        0  0  0  0  0  0  0  0  0  0  1  0  0  0  0 ;...
%        0  0  0  0  0  0  0  0  0  0  0  1  0  0  0 ;...
%        0  0  0  0  0  0  1  1  0  0  0  1  1  0  1 ;...
%        0  0  0  0  0  0  1  1  1  0  0  0  1  0  1 ;...
%        0  0  0  0  0  0  0  1  1  1  0  0  1  1  0 ;...
%        0  0  0  0  0  0  0  0  1  1  1  1  1  1  0 ;...
%        0  0  0  0  0  0  0  0  0  1  1  1  0  1  1 ;...
%        0  0  0  0  0  0  1  0  0  0  1  1  0  1  1 ;...
%        0  0  0  0  0  0 -2 -2 -2 -2  0  0 -2 -2 -2 ;...
%        0  0  0  0  0  0  0  0 -2 -2 -2 -2 -2 -2 -2 ;...
%        0  0  0  0  0  0 -2 -2  0  0 -2 -2 -2 -2 -2 ];

% cm51 = [0  0  0  0  0   1  0  0  0  0  0;...
%         0  0  0  0  0   0  1  0  0  0  0;...
%         0  0  0  0  0   0  0  1  0  0  0;...
%         0  0  0  0  0   0  0  0  1  0  0;...
%         0  0  0  0  0   0  0  0  0  1  0;...
%         0  0  0  0  0   1  1  0  0  1  .8;...
%         0  0  0  0  0   1  1  1  0  0  .8;...
%         0  0  0  0  0   0  1  1  1  0  .8;...
%         0  0  0  0  0   0  0  1  1  1  .8;...
%         0  0  0  0  0   1  0  0  1  1  .8;...
%         0  0  0  0  0  -4 -4 -4 -4 -4 -8];

% cm51 = [0  0  0  0  0   1  0  0  0  0  0;...
%         0  0  0  0  0   0  1  0  0  0  0;...
%         0  0  0  0  0   0  0  1  0  0  0;...
%         0  0  0  0  0   0  0  0  1  0  0;...
%         0  0  0  0  0   0  0  0  0  1  0;...
%         0  0  0  0  0   2  2  0  0  2  .8;...
%         0  0  0  0  0   2  2  2  0  0  .8;...
%         0  0  0  0  0   0  2  2  2  0  .8;...
%         0  0  0  0  0   0  0  2  2  2  .8;...
%         0  0  0  0  0   2  0  0  2  2  .8;...
%         0  0  0  0  0  -8 -8 -8 -8 -8 -4];

% cm50 = [0  0  0  0  0   0  0  0  0  0  ;...
%         0  0  0  0  0   0  0  0  0  0  ;...
%         0  0  0  0  0   0  0  0  0  0  ;...
%         0  0  0  0  0   0  0  0  0  0  ;...
%         0  0  0  0  0   0  0  0  0  0  ;...
%         0  0  0  0  0   1  1  0  0  1  ;...
%         0  0  0  0  0   1  1  1  0  0  ;...
%         0  0  0  0  0   0  1  1  1  0  ;...
%         0  0  0  0  0   0  0  1  1  1  ;...
%         0  0  0  0  0   1  0  0  1  1 ];

% cm60 = [1  1  0  0  0  0  ;...
%         1  1  1  0  0  0  ;...
%         0  1  1  1  0  0  ;...
%         0  0  1  1  1  0  ;...
%         0  0  0  1  1  1  ;...
%         0  0  0  0  1  1  ];
%     
% cm601 = [0  1  0  0  0  0  ;...
%          1  0  1  0  0  0  ;...
%          0  1  0  1  0  0  ;...
%          0  0  1  0  1  0  ;...
%          0  0  0  1  0  1  ;...
%          0  0  0  0  1  0  ];    
%     
% cm50 = [1  1  0  0  1  ;...
%         1  1  1  0  0  ;...
%         0  1  1  1  0  ;...
%         0  0  1  1  1  ;...
%         1  0  0  1  1 ];
% 
% cm500 = [1  0  0  0  0  ;...
%          0  1  0  0  0  ;...
%          0  0  1  0  0  ;...
%          0  0  0  1  0  ;...
%          0  0  0  0  1 ];
% 
% cm120 = [1  1  0  0  0  0  ;...
%         1  1  1  0  0  0  ;...
%         0  1  1  1  0  0  ;...
%         0  0  1  1  1  0  ;...
%         0  0  0  1  1  1  ;...
%         0  0  0  0  1  1  ];
    
cm1201 = create_cm(N, 1, 0);
     
cm = cm1201;
        
non = [0 0 0 0 0];
bar = [0 0 0 1 1];
shu = [0 0 1 0 1];

% input = non;
input = [];

% initial_state = [1 1 1 1 0];

% initial_state = non;
initial_state = [];
% initial_state = -1;

% temps = [.25:.25:3 4 5 7 10];
% temps = [.25:.25:3];
temps = linspace(0.25, 1.25, 20);

% figure; 
nlin = 3;
ncol = ceil(length(temps)/nlin);

ntemp = length(temps);
T = 10000;
nruns = 1;

diffs = zeros(T, ntemp, nruns);
for run = 1:nruns
    fprintf('\nrun %d of %d\n\n', run, nruns)
    
    % temp = 1.25;
    for itemp = 1:length(temps)

        temp = temps(itemp);

        fprintf('temp : %1.2f | ', temp)

        % cm = cm500; thresh = 0; 
        % cm = cm500; thresh = 0; temp = 1e16;


        % plot(digraph(cm))


        % build tpms

        tpm = build_tpm(cm, thresh, temp);
        % tpm = build_tpm_step(cm, b);

        % N = size(cm, 2);
        % nstates = 2^N;
        % t = ceil(sum(cm)/2);
        % states = zeros(nstates, N);
        % for i = 0:nstates-1
        %     states(i+1, :) = de2bi(i, N)';    
        % end
        % figure; imagesc(states)

        % figure; imagesc(tpm);
        caxis([0 1]);

        % run and plot

        has_input = ~isempty(input);
        
        if isempty(initial_state)
            % random initial state
            state = de2bi(randi(2^N)-1, N);
        else
            if sum(initial_state) < 0
                % go trhough all initial state
                state = de2bi(run-1, N);
            else
                state = initial_sate;
            end
        end
        fprintf('initial state : [ ')
        fprintf('%d ', state)
        fprintf(']\n')
        
        if has_input
            % state = [input rand(1, size(cm,1) - length(bar))>.5];
            state = [input state];
            is = length(input);
        else
            is = 0;
        end

        has_inhib = size(cm, 2)+is > N;

        maxstates = 30;
        states = [state];
        % the_pause = .5;
        the_pause = 0;
        for t = 1:T
            prob_on = tpm(bi2de(state)+1, :);
            nstate = rand(size(prob_on)) < prob_on;

            shift = is+randperm(N-is, ceil(.3*(N-is)) );
            state(shift) = nstate(shift);

            states = [states; state];
            if size(states ,1) > maxstates
                states = states(end-maxstates:end, :);
            end

            pstates = states;
            if is > 0
                pstates(:, 1:is) = pstates(:, 1:is) .* 2;
            end
            if has_inhib 
                pstates(:, is+is+1:end) = pstates(:, is+is+1:end) .* 3;
            end

        %     diffs = [diffs abs([diff(state(7:12), 1, 2) state(7)-state(12)])];
        %     histogram(diffs)
        %     xlim([-1 2])

            x = state(is+1:is+N);

            % 5 nodes
%             y = conv(x, [1 1]);
%             y = [y(2:end-1) x(end)+x(1)];
%             if any(y==2)
%                 if any(y==0)
%                     yx = 3;
%                 else
%                     if all(y==2)
%                         yx = 4;
%                     else
%                         yx = 2;
%                     end
%                 end
%             elseif any(y==1)
%                 yx = 1;
%             else
%                 yx=4;
%             end   
            
            % 6 nodes no wrap
            y = conv(x, [1 1]);
        %     y = [y(2:end-1) x(end)+x(1)];
            if any(y==2)
                if ~isempty(findstr(x, [0 1 0])) || ...
                   all(x(1:2) == [1 0]) || ...
                   all(x(end-1:end) == [0 1])
                    yx = 2;
                else
                    if all(y==2)
                        yx = 4;
                    else
                        yx = 3;
                    end
                end
            elseif any(y==1)
                yx = 1;
            else
                yx=4;
            end
            diffs(t,itemp,run) = yx;

        %     diffs = [diffs bi2de(x)];

            if the_pause
                subplot(3, 1, [1 2])
                imagesc(pstates);
                caxis([0 3])
                colormap(prism(4));
                colorbar()
                pause(the_pause)
                subplot(3, 1, 3)
                histogram(diffs)
                xlim([min(diffs)-1 max(diffs)+1])
            end
        end
    end
    
end
%%

% first get baseline

classes = {'Discontinuous', 'Mixed', 'Bars', 'Full'};

fprintf('\n\n\n')
adiffs = [];
for i = 0:(2^N-1)
    x = de2bi(i, N);

    y = conv(x, [1 1]);
%     y = [y(2:end-1) x(end)+x(1)];
    if any(y==2)
        if ~isempty(findstr(x, [0 1 0])) || ...
           all(x(1:2) == [1 0]) || ...
           all(x(end-1:end) == [0 1])
            yx = 2;
        else
            if all(y==2)
                yx = 4;
            else
                yx = 3;
            end
        end
    elseif any(y==1)
        yx = 1;
    else
        yx=4;
    end

%     y = abs([diff(x) x(end)-x(1)]);
%     yx = sum(y==0) - sum(y==1);

%     fprintf('%i: [ ', i)
%     fprintf('%i ', x)
%     fprintf('] | [ ')
%     fprintf('%i ', y)
%     fprintf(']: %i\n', yx)
    
    adiffs  = [adiffs yx];
end

chance_level = hist(adiffs(:),min(adiffs(:)):max(adiffs(:)) )./(2^N)

figure
histogram (adiffs)
xlim([min(adiffs(:))-1 max(adiffs(:))+1])

% now plot normalized values

figure

if ~the_pause

%         subplot(3, 1, [1 2])
%     imagesc(pstates);
%     caxis([0 3])
%     colormap(prism(4));
%     colorbar()
%     pause(the_pause)
%     subplot(3, 1, 3)
    
    dist_to_chance = zeros(length(chance_level), ntemp);
    for itemp = 1:ntemp
        temp = temps(itemp);
        this_classes = zeros(4, nruns);
        for run = 1:nruns
            this_classes(:,run) = hist(diffs(:,itemp, run), 1:4)./T;
%             this_classes(:,run) = this_classes(:,run)./chance_level';
        end
%         histogram(mean(this_classes,2), 'normalization', 'probability')
%         xlim([min(diffs)-1 max(diffs)+1])
        subplot(nlin, ncol, itemp)
        boxplot(this_classes')
%         m = mean(this_classes,2);
%         s = std(this_classes,[],2);
%         bar(1:4, m);
%         hold on
%         errorbar(1:4, m, s, 'r');
        
        xlim([0 5])
        title(sprintf('temp : %1.2f', temp))
        
        hold on
        plot(1:length(chance_level), chance_level, 'g*')
        dist_to_chance(:, itemp) = mean(this_classes,2)-chance_level';
    end
    figure; 
    plot(temps, dist_to_chance)
    title('distance to chance level per temperature');
    legend(classes)
end

%% 

