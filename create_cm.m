function cm = create_cm(N, selfcon, wrap)

    if nargin < 2, selfcon = 1;end
    if nargin < 3, wrap= 1;end
    
    cm = circshift(eye(N),[1 0])+circshift(eye(N),[-1,0]);
    
    if selfcon
        cm = cm + eye(N);
    end
    
    if ~wrap
        cm(1,end)=0;
        cm(end,1)=0;
    end
