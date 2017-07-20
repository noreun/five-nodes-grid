% Build system

% save_results = 0;
save_results = 1;

results_folder = '/data/nsdm/pyphi/dynamics';

pyphi_matrix = load('/data/nsdm/pyphi/C_matrix.mat');
cm_grid_fpp = circshift(double(pyphi_matrix.CG), [5 5]);
cm_rand_fpp = circshift(double(pyphi_matrix.CR), [5 5]);

% Ns = 5; do_self_conn = 0; do_wrap = 1;
Ns = 9; do_self_conn = 0; do_wrap = 1;
% Ns = 16; do_self_conn = 1; do_wrap = 0;

% thresh = 1;
% thresh = 2;
thresh = 4;


voi = [];

non = [0 0 0 0 0];
bar = [0 0 0 1 1];
shu = [0 0 1 0 1];

% temps = [.25:.25:3 4 5 7 10];
temps = 0:.25:4;
% temps = 0:.1:1.4;
% temps = .5:.5:1.1;

% temps = linspace(0, 1, 11);
% temps = linspace(0, 2, 21);

% figure; 
nlin = 3;
ncol = ceil(length(temps)/nlin);

ntemp = length(temps);
T = 1000;
nruns = 32;
% nruns = 10;
the_pause = 0;
    
% all_inputs = {'bar', 'shu'};
% all_inputs = {'non'};
all_inputs = {'voi'};

% % visual inpection
% temps = 2;
% nruns = 30;
% the_pause = .5;
% states = [];
% T = 100;

if save_results
    warning('Going to overwrite files!!!');
end

ftic = tic;
for iN = 1:length(Ns)
    
    N = Ns(iN);
    fprintf('N: %d\n\n', N);

    cm9noinput = create_cm_2d(N, do_self_conn, do_wrap);
    cms = {'cm9noinput'};

%     cms = {'cm_grid_fpp', 'cm_rand_fpp'};
%     cms = {'cmG51', 'cmR51'};
    
    for icm = 1:length(cms)

        fprintf('cm: %s\n\n', cms{icm});
        cm = eval(cms{icm});

        for iinput = 1:length(all_inputs)

            fprintf('input: %s\n\n', all_inputs{iinput});
            input = eval(all_inputs{iinput});

            % initial_state = [1 1 1 1 0];

            % initial_state = non;
            initial_state = [];
            % initial_state = -1;

            diffs = nan(length(Ns), T, ntemp, nruns);

            all_states_buffer = cell(1,nruns);
            parfor run = 1:nruns
                this_state = nan(length(Ns), T, ntemp);
                
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
                    has_input = ~isempty(input);
                    if has_input
                        % state = [input rand(1, size(cm,1) - length(bar))>.5];
                        state = [input state];
                        is = length(input);
                    else
                        is = 0;
                    end

                    has_inhib = size(cm, 2) - (N + is);
                    if has_inhib 
                        state = [state zeros(1,has_input)];
                    end


%                     if the_pause
%                         figure
%                     end

                    ttic = tic;
                    for t = 1:T
                        prob_on = tpm(bi2de(state(:))+1, :);
                        nstate = rand(size(prob_on)) < prob_on;

                        ratio_update = .3;
%                         ratio_update = 1;

                        shift = is+randperm(N+has_inhib, ceil(ratio_update*(N+has_inhib)) );
                        state(shift) = nstate(shift);

                        x = state(is+1:is+N);

%                         diffs(iN,t,itemp,run) = yx;
                        this_state(iN,t,itemp) = bi2de(x, 'left-msb');

%                         if the_pause
%                             imagesc(reshape(state, [sqrt(N), sqrt(N)]));
%                             pause(the_pause)
%                         end
                    end
                    ttoc = toc(ttic);
%                     fprintf('\n\ntook %5.2f min\n\n', ttoc/60);
                end
                all_states_buffer{run} = this_state;
            end
            
            all_states = nan(length(Ns), T, ntemp, nruns);
            for run = 1:nruns
                all_states(:,:,:,run) = all_states_buffer{run};
            end
            
            save(sprintf('%s/all_states_2d_%d_nodes_%s_%s_T_%1.1f_%1.1f_%d.mat', results_folder, N, all_inputs{iinput}, cms{icm}, min(temps), max(temps), length(temps)), ...
                'all_states', 'diffs', 'Ns', 'temps', 'do_wrap', 'do_self_conn')

        end
    end
end

ftoc = toc(ftic);
fprintf('\n\nDone\n\nTook: %2.5f min\n\n', ftoc/60);
