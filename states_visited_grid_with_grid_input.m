% Build system

% save_results = 0;
save_results = 1;

% results_folder = '/data/nsdm/pyphi/dynamics';
results_folder = '/data/nsdm/pyphi/dynamics_grid_grid';

pyphi_matrix = load('/data/nsdm/pyphi/C_matrix.mat');
cm_grid_fpp = circshift(double(pyphi_matrix.CG), [5 5]);
cm_rand_fpp = circshift(double(pyphi_matrix.CR), [5 5]);

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
%         0  0  0  0  0   1  1  0  0  1  1;...
%         0  0  0  0  0   1  1  1  0  0  1;...
%         0  0  0  0  0   0  1  1  1  0  1;...
%         0  0  0  0  0   0  0  1  1  1  1;...
%         0  0  0  0  0   1  0  0  1  1  1;...
%         0  0  0  0  0  -1 -1 -1 -1 -1 -5];

cmGG50 = [1  1  0  0  1   1  0  0  0  0  ;...
          1  1  1  0  0   0  1  0  0  0  ;...
          0  1  1  1  0   0  0  1  0  0  ;...
          0  0  1  1  1   0  0  0  1  0  ;...
          1  0  0  1  1   0  0  0  0  1  ;...
          0  0  0  0  0   1  1  0  0  1  ;...
          0  0  0  0  0   1  1  1  0  0  ;...
          0  0  0  0  0   0  1  1  1  0  ;...
          0  0  0  0  0   0  0  1  1  1  ;...
          0  0  0  0  0   1  0  0  1  1  ];

cmSG50 = [1  0  0  0  0   1  0  0  0  0  ;...
          0  1  0  0  0   0  1  0  0  0  ;...
          0  0  1  0  0   0  0  1  0  0  ;...
          0  0  0  1  0   0  0  0  1  0  ;...
          0  0  0  0  1   0  0  0  0  1  ;...
          0  0  0  0  0   1  1  0  0  1  ;...
          0  0  0  0  0   1  1  1  0  0  ;...
          0  0  0  0  0   0  1  1  1  0  ;...
          0  0  0  0  0   0  0  1  1  1  ;...
          0  0  0  0  0   1  0  0  1  1  ];
      
cmGR50 = [1  1  0  0  1   1  0  0  0  0  ;...
          1  1  1  0  0   0  1  0  0  0  ;...
          0  1  1  1  0   0  0  1  0  0  ;...
          0  0  1  1  1   0  0  0  1  0  ;...
          1  0  0  1  1   0  0  0  0  1  ;...
          0  0  0  0  0   1  1  0  0  1  ;...
          0  0  0  0  0   1  1  0  0  1  ;...
          0  0  0  0  0   0  1  1  1  0  ;...
          0  0  0  0  0   1  0  1  1  0  ;...
          0  0  0  0  0   0  0  1  1  1  ];

cmSR50 = [1  0  0  0  0   1  0  0  0  0  ;...
          0  1  0  0  0   0  1  0  0  0  ;...
          0  0  1  0  0   0  0  1  0  0  ;...
          0  0  0  1  0   0  0  0  1  0  ;...
          0  0  0  0  1   0  0  0  0  1  ;...
          0  0  0  0  0   1  1  0  0  1  ;...
          0  0  0  0  0   1  1  0  0  1  ;...
          0  0  0  0  0   0  1  1  1  0  ;...
          0  0  0  0  0   1  0  1  1  0  ;...
          0  0  0  0  0   0  0  1  1  1  ];
      
% cmR51 = [ 0  0  0  0  0   1  0  0  0  0  0;...
%           0  0  0  0  0   0  1  0  0  0  0;...
%           0  0  0  0  0   0  0  1  0  0  0;...
%           0  0  0  0  0   0  0  0  1  0  0;...
%           0  0  0  0  0   0  0  0  0  1  0;...
%           0  0  0  0  0   1  1  0  0  1  .25;...
%           0  0  0  0  0   1  1  0  0  1  .25;...
%           0  0  0  0  0   0  1  1  1  0  .25;...
%           0  0  0  0  0   1  0  1  1  0  .25;...
%           0  0  0  0  0   0  0  1  1  1  .25;...
%           0  0  0  0  0  -2 -2 -2 -2 -2  -1];
      
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

cm50 = [0  0  0  0  0   1  0  0  0  0  ;...
        0  0  0  0  0   0  1  0  0  0  ;...
        0  0  0  0  0   0  0  1  0  0  ;...
        0  0  0  0  0   0  0  0  1  0  ;...
        0  0  0  0  0   0  0  0  0  1  ;...
        0  0  0  0  0   1  1  0  0  1  ;...
        0  0  0  0  0   1  1  1  0  0  ;...
        0  0  0  0  0   0  1  1  1  0  ;...
        0  0  0  0  0   0  0  1  1  1  ;...
        0  0  0  0  0   1  0  0  1  1 ];

cm50noself = [  0  0  0  0  0   1  0  0  0  0  ;...
                0  0  0  0  0   0  1  0  0  0  ;...
                0  0  0  0  0   0  0  1  0  0  ;...
                0  0  0  0  0   0  0  0  1  0  ;...
                0  0  0  0  0   0  0  0  0  1  ;...
                0  0  0  0  0   0  1  0  0  1  ;...
                0  0  0  0  0   1  0  1  0  0  ;...
                0  0  0  0  0   0  1  0  1  0  ;...
                0  0  0  0  0   0  0  1  0  1  ;...
                0  0  0  0  0   1  0  0  1  0 ];
    
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

% Ns = [6, 12, 24, 48];

% Ns = 5; do_self_conn = 0; do_wrap = 1;
% Ns = 5; do_self_conn = 1; do_wrap = 1;
% Ns = 16; do_self_conn = 1; do_wrap = 0;
Ns = 5; do_self_conn = 1; do_wrap = 0;

% thresh = 1;
thresh = 2;
% thresh = 3;
     
non = [0 0 0 0 0];
bar = [0 0 0 1 1];
shu = [0 0 1 0 1];

% temps = 0:.25:2.9;
% temps = 0:.1:1.4;
temps = .5:.5:1.1;

% figure; 
nlin = 3;
ncol = ceil(length(temps)/nlin);

ntemp = length(temps);
T = 1000;
nruns = 30;
% nruns = 10;
the_pause = 0;
    
% all_inputs = {'bar', 'shu'};
% all_inputs = {'bar'};
% input = non;
input = [];

% % visual inpection
% temps = 1;
% nruns = 30;
% the_pause = .5;
% states = [];
% T = 100;

if save_results
    warning('Going to overwrite files!!!');
end

for iN = 1:length(Ns)
    
    N = Ns(iN);
    fprintf('N: %d\n\n', N);

%     cmGnoself = create_cm(N, do_self_conn, do_wrap);
%         cm = cm50; 
%         cm = cm51; warning('Using inhibitory!')
%         cm = cm_rand_fpp;
%     cm = cm_grid_fpp;

%     cms = {'cm_grid_fpp', 'cm_rand_fpp'};
%     cms = {'cmG51', 'cmR51'};
%     cms = {'cmGG50', 'cmSG50', 'cmGR50', 'cmSR50'};
%     cms = {'cm50noself'};    
    
    cmonthefly = create_cm(N, do_self_conn, do_wrap);
    cms = {'cmonthefly'};
    
    for icm = 1:length(cms)

        fprintf('cm: %s\n\n', cms{icm});
        cm = eval(cms{icm});

%         for iinput = 1:length(all_inputs)

%             fprintf('input: %s\n\n', all_inputs{iinput});

        % initial_state = [1 1 1 1 0];

        % initial_state = non;
        initial_state = [];
        % initial_state = -1;

        diffs = nan(length(Ns), T, ntemp, nruns);
        all_states = nan(length(Ns), T, ntemp, nruns);

        for run = 1:nruns
            fprintf('\nrun %d of %d\n\n', run, nruns)

            % temp = 1.25;
            for itemp = 1:length(temps)

                temp = temps(itemp);

                fprintf('temp : %1.2f | ', temp)

                % build tpms
                tpm = build_tpm(cm, thresh, temp);

                % select initial state
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

                % run
                
                % for now use input as a flag to select random starting state
                has_input = ~isempty(input);
                if has_input
                    % state = [input rand(1, size(cm,1) - length(bar))>.5];
%                     state = [input state];
                    state = [de2bi(randi(2^N)-1, N) state];
                    is = length(input);
                else
                    is = 0;
                end

                has_inhib = size(cm, 2) - (N + is);
                if has_inhib 
                    state = [state zeros(1,has_input)];
                end


                maxstates = 30;
                states = [state];

                if the_pause
                    figure
                end

                ttic = tic;
                for t = 1:T

                    prob_on = tpm(bi2de(state)+1, :);
                    nstate = rand(size(prob_on)) < prob_on;

%                     ratio_update = .3;
                    ratio_update = 1;

                    shift = randperm(is+N+has_inhib, ceil(ratio_update*(is+N+has_inhib)) );
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
                    if do_wrap
                        y = conv([x(end) x x(1)], [1 1]);
                    else
                        y = conv(x, [1 1]);
                    end
                    y = y(2:end-1);

                    if any(y==2)

                        mixed = 0;
                        if do_wrap 
                           if ~isempty(findstr(x, [0 1 0])) || ...
                              all([x(end) x(1:2)] == [0 1 0]) || all([x(end-1:end) x(1)] == [0 1 0])
                                mixed = 1;
                           end
                        else
                           if ~isempty(findstr(x, [0 1 0])) || ...
                              all(x(1:2) == [1 0]) || all(x(end-1:end) == [0 1])
                                mixed = 1;
                           end
                        end

                        if mixed
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

                    diffs(iN,t,itemp,run) = yx;
                    all_states(iN,t,itemp,run) = bi2de(x, 'left-msb');

                %     diffs = [diffs bi2de(x)];

                    if the_pause
                        subplot(3, 1, [1 2])
                        imagesc(pstates);
                        caxis([0 3])
                        colormap(prism(4));
                        colorbar()
                        pause(the_pause)
                        subplot(3, 1, 3)
                        histogram(diffs(iN, :, itemp, run))
%                         xlim([min(diffs(iN,:))-1 max(diffs(iN,:)+1)])
                        xlim([0 5])
                    end
                end
                ttoc = toc(ttic);
%                 fprintf('\n\ntook %5.2f min\n\n', ttoc/60);
            end

        end

            save(sprintf('%s/all_states_%d_nodes_%s_T_%1.1f_%1.1f_%d.mat', results_folder, N, cms{icm}, min(temps), max(temps), length(temps)), ...
                'all_states', 'diffs', 'Ns', 'temps', 'do_wrap', 'do_self_conn')

%         end
    end
end
%%

% files_to_load = {};

% files_to_load = {sprintf('%s/all_states_16_nodes_cm16selfwrap_T_0.0_1.4_15.mat', results_folder)};

% files_to_load = {sprintf('%s/all_states_16_nodes_cm16selfwrap_T_0.0_1.4_15.mat', results_folder)};

% files_to_load = {...
% sprintf('%s/all_states_5_nodes_bar_cm_grid_fpp_T_0.0_1.4_15.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_shu_cm_grid_fpp_T_0.0_1.4_15.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_bar_cm_rand_fpp_T_0.0_1.4_15.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_shu_cm_rand_fpp_T_0.0_1.4_15.mat', results_folder)};

% files_to_load = {...
% sprintf('%s/all_states_5_nodes_bar_cm_grid_fpp.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_shu_cm_grid_fpp.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_bar_cm_rand_fpp.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_shu_cm_rand_fpp.mat', results_folder)};

% files_to_load = {...
% sprintf('%s/all_states_5_nodes_bar_cmG51.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_shu_cmG51.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_bar_cmR51.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_shu_cmR51.mat', results_folder)};

% files_to_load = {...
% sprintf('%s/all_states_5_nodes_bar_cm50noself.mat', results_folder),...
% sprintf('%s/all_states_5_nodes_shu_cm50noself.mat', results_folder)};

% first get baseline

classes = {'Discontinuous', 'Mixed', 'Bars', 'Full'};

for ifile = 1:length(files_to_load)

    load(files_to_load{ifile});

    T = size(diffs,2);
    ntemp = size(diffs,3);
    nruns = size(diffs,4);
    
    for iN = 1:length(Ns)
        N = Ns(iN);

        fprintf('\n\n\n')
        adiffs = [];
        for i = 0:(2^N-1)
            x = de2bi(i, N, 'left-msb');

            if do_wrap
                y = conv([x(end) x x(1)], [1 1]);
            else
                y = conv(x, [1 1]);
            end
            y = y(2:end-1);

            if any(y==2)

                mixed = 0;
                if do_wrap 
                   if ~isempty(findstr(x, [0 1 0])) || ...
                      all([x(end) x(1:2)] == [0 1 0]) || all([x(end-1:end) x(1)] == [0 1 0])
                        mixed = 1;
                   end
                else
                   if ~isempty(findstr(x, [0 1 0])) || ...
                      all(x(1:2) == [1 0]) || all(x(end-1:end) == [0 1])
                        mixed = 1;
                   end
                end

                if mixed
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

    %     figure
    %     histogram (adiffs)
    %     xlim([min(adiffs(:))-1 max(adiffs(:))+1])

        % now plot normalized values

    %     figure

        prob_per_state = zeros(2^N-1, ntemp);
        dist_to_chance = zeros(length(chance_level), ntemp);
        dist_to_chance_std = zeros(length(chance_level), ntemp);
        ratio_occ = zeros(length(chance_level), ntemp);
        for itemp = 1:ntemp
            temp = temps(itemp);
            this_classes_raw = zeros(4, nruns);
            this_classes = zeros(4, nruns);
            this_states = zeros(2^N-1, nruns);
            for run = 1:nruns
                this_classes_raw(:,run) = hist(diffs(iN,100:end,itemp, run), 1:4)./T;
                this_classes(:,run) =  this_classes_raw(:,run) - chance_level';
                this_states(:,run) = hist(all_states(iN,:,itemp,run), 1:(2^N-1))./T;
            end
    %         histogram(mean(this_classes,2), 'normalization', 'probability')
    %         xlim([min(diffs)-1 max(diffs)+1])


    %             subplot(nlin, ncol, itemp)
    %             
    %             boxplot(this_classes')
    %             hold on
    %             xlim([0 5])
    %             plot(1:length(chance_level), chance_level, 'g*')
    %             ylim([0 1])

    %             boxplot(this_states')

    %         m = mean(this_classes,2);
    %         s = std(this_classes,[],2);
    %         bar(1:4, m);
    %         hold on
    %         errorbar(1:4, m, s, 'r');

    %         title(sprintf('temp : %1.2f', temp))

            ratio_occ(:, itemp) = mean(this_classes_raw,2);
            dist_to_chance(:, itemp) = mean(this_classes,2);
            dist_to_chance_std(:, itemp) = std(this_classes,[],2);
            prob_per_state (:, itemp) = mean(this_states, 2);
        end

    %         figure;
    %         plot(temps, ratio_occ)
    %         title('Ratio occurrence per class per temperature');
    %         legend(classes)


        figure
        c =  [  0    0.4470    0.7410;...
                0.8500    0.3250    0.0980;...
                0.9290    0.6940    0.1250;...
                0.4940    0.1840    0.5560;...
                0.4660    0.6740    0.1880;...
                0.3010    0.7450    0.9330;...
                0.6350    0.0780    0.1840;];
        for iclass = 1:size(dist_to_chance,1)
            e = errorbar(temps, dist_to_chance(iclass, :), dist_to_chance_std(iclass, :));
            e.Color = c(iclass,:);
            hold on
        end
        legend(classes)
        title(strrep(files_to_load{ifile}, '_', '\_'))
        ylim([ -.8 1.2])
    end
end


%%

% 
% % N=5
% 
% fprintf('\n\n\n')
% adiffs = [];
% for i = 0:(2^N-1)
%     x = de2bi(i, N, 'left-msb');
%     if do_wrap
%         y = conv([x(end) x x(1)], [1 1]);
%     else
%         y = conv(x, [1 1]);
%     end
%     y = y(2:end-1);
%     if any(y==2)
%         
%         mixed = 0;
%         if do_wrap 
%            if ~isempty(findstr(x, [0 1 0])) || ...
%               all([x(end) x(1:2)] == [0 1 0]) || all([x(end-1:end) x(1)] == [0 1 0])
%                 mixed = 1;
%            end
%         else
%            if ~isempty(findstr(x, [0 1 0])) || ...
%               all(x(1:2) == [1 0]) || all(x(end-1:end) == [0 1])
%                 mixed = 1;
%            end
%         end
%         
%         if mixed
%             yx = 2;
%         else
%             if all(y==2)
%                 yx = 4;
%             else
%                 yx = 3;
%             end
%         end
%     elseif any(y==1)
%         yx = 1;
%     else
%         yx=4;
%     end
% 
% %     y = abs([diff(x) x(end)-x(1)]);
% %     yx = sum(y==0) - sum(y==1);
% 
%     fprintf('%i: [ ', i)
%     fprintf('%i ', x)
%     fprintf('] | [ ')
%     fprintf('%i ', y)
%     fprintf(']: %i\n', yx)
% 
%     adiffs  = [adiffs yx];
% end
% 
% figure
% histogram (adiffs)
% xlim([min(adiffs(:))-1 max(adiffs(:))+1])
% 
% a = hist(adiffs, (min(adiffs(:))-1):(max(adiffs(:))+1))
