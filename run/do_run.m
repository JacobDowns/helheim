addpath /home/jake/trunk-jpl/bin/ /home/jake/trunk-jpl/lib/

load('model.mat');
md.timestepping.final_time = 2007.25;
md.cluster=generic('name', oshostname, 'np', 3);
md=solve(md,'Transient');
