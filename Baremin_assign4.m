alpha_values = [0.001 0.010 0.100];

x = linspace(0,1,50);
t = [1 5 10 50 100];

for k = 1:length(alpha_values)
    
    alpha = alpha_values(k);
    
    sol = pdepe(0,@(x,t,u,DuDx)pdefun(x,t,u,DuDx,alpha),@icfun,@bcfun,x,t);
                  
    T = sol(:,:,1);
    
    figure
    plot(x,T)
    xlabel('m')
    ylabel('K')
    legend('1s','5s','10s','50s','100s')
    grid on
    
end


% ---------------- PDE Function ----------------

function [c,f,s] = pdefun(x,t,u,DuDx,alpha)

c = 1;
f = alpha*DuDx;
s = 0;

end


% ---------------- Initial Condition ----------------

function u0 = icfun(x)

u0 = 350;

end


% ---------------- Boundary Conditions ----------------

function [pl,ql,pr,qr] = bcfun(xl,ul,xr,ur,t)

pl = ul - 300;
ql = 0;

pr = ur - 400;
qr = 0;

end