addpath /home/jake/trunk-jpl/bin/ /home/jake/trunk-jpl/lib/

load model1;

plotmodel(md, 'data', md.friction.pressure_regions);

%Control general
md.inversion.iscontrol = 1;
md.inversion.nsteps = 40;
md.inversion.step_threshold = 0.99*ones(md.inversion.nsteps,1);
md.inversion.maxiter_per_step = 5*ones(md.inversion.nsteps,1);

%Cost functions
md.inversion.cost_functions=[101 103 501];
md.inversion.cost_functions_coefficients = ones(md.mesh.numberofvertices,3);
md.inversion.cost_functions_coefficients(:,1) = 5000;
md.inversion.cost_functions_coefficients(:,2) = 10;
md.inversion.cost_functions_coefficients(:,3) = 3*50^-3;
pos = find(md.mask.ice_levelset>0);
md.inversion.cost_functions_coefficients(pos,1:2) = 0;

%Controls
md.inversion.control_parameters = {'FrictionCoefficient'};
md.inversion.gradient_scaling(1:md.inversion.nsteps) = 50;
md.inversion.min_parameters = 1*ones(md.mesh.numberofvertices,1);
md.inversion.max_parameters = 300*ones(md.mesh.numberofvertices,1);

%Additional parameters
md.stressbalance.restol = 0.01;
md.stressbalance.reltol = 0.1; 
md.stressbalance.abstol = NaN;

md.cluster=generic('name',oshostname, 'np',  5);
md = solve(md, 'sb');

%Put results back into the model
md.friction.coefficient = md.results.StressbalanceSolution.FrictionCoefficient;
md.initialization.vx = md.results.StressbalanceSolution.Vx;
md.initialization.vy = md.results.StressbalanceSolution.Vy;
md.inversion.iscontrol = 0;

save model2 md;
