addpath /home/jake/trunk-jpl/bin/ /home/jake/trunk-jpl/lib/

load('model2.mat');

md.timestepping.final_time = 2015;

md.cluster=generic('name', oshostname, 'np', 6);
md=solve(md,'Transient');
