function cm = create_cm_2d(N, selfcon, wrap)

    if nargin < 2, selfcon = 1;end
    if nargin < 3, wrap= 1;end

    sqN = sqrt(N);
    
    cm = vcircshift(eye(sqN), 1, wrap) + vcircshift(eye(sqN), -1, wrap);
    
    cm = kron(cm, eye(sqN)) + kron(eye(sqN), cm);

    if selfcon
        cm = cm + eye(N);
    end

