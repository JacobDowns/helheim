addpath /home/jake/trunk-jpl/bin/ /home/jake/trunk-jpl/lib/

% Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load model0;

file_name1 = '/media/jake/drive1/greenland_data/h5/geometry/geometry_data.h5';
file_name2 = '/media/jake/drive1/greenland_data/h5/velocity/velocity_data.h5';
file_name3 = '/media/jake/drive1/greenland_data/h5/smb/smb_data_2000_2009.h5';
file_name4 = '/media/jake/drive1/greenland_data/h5/smb/smb_data_2010_2019.h5';

x = double(h5read(file_name1, '/x'));
y = double(h5read(file_name1, '/y'));
x = x(: ,1);
y = y(1 ,:)';
H = double(h5read(file_name1, '/thickness'));
B = double(h5read(file_name1, '/bed'));
S = double(h5read(file_name1, '/surface'));
mask = double(h5read(file_name1, '/mask')); 
vx = double(h5read(file_name2, '/vx'));
vy = double(h5read(file_name2, '/vy'));


% Geometry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set geometry
mask = InterpFromGridToMesh(x, y, mask', md.mesh.x, md.mesh.y, 0);
md.geometry.thickness = InterpFromGridToMesh(x, y, H', md.mesh.x, md.mesh.y, 0);
md.geometry.surface = InterpFromGridToMesh(x, y, S', md.mesh.x, md.mesh.y, 0);
md.geometry.bed = InterpFromGridToMesh(x, y, B', md.mesh.x, md.mesh.y, 0);

% Ice levelset
ice_free_indexes = find(md.geometry.thickness == 0);
ice_indexes = find(md.geometry.thickness > 0);
md.mask.ice_levelset = -1*ones(md.mesh.numberofvertices,1);
md.mask.ice_levelset(ice_free_indexes) = 1;

% Min thickness
md.geometry.base = md.geometry.surface - md.geometry.thickness;
indexes = find(abs((md.geometry.surface - md.geometry.thickness) - md.geometry.bed) < 1e-3);
md.geometry.base(indexes) = md.geometry.bed(indexes);
md.geometry.thickness(md.geometry.thickness < 10) = 10;
md.geometry.surface = md.geometry.base + md.geometry.thickness;
md.masstransport.min_thickness = 10;

% Ocean levelset 
ocean_indexes = find(md.geometry.base > md.geometry.bed);
md.mask.ocean_levelset = ones(md.mesh.numberofvertices,1);
md.mask.ocean_levelset(ocean_indexes) = -1;


% Ice velocity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

md.inversion.vx_obs  = InterpFromGridToMesh(x, y, vx', md.mesh.x, md.mesh.y, 0);
md.inversion.vy_obs  = InterpFromGridToMesh(x, y, vy', md.mesh.x, md.mesh.y, 0);
md.inversion.vz_obs = zeros(md.mesh.numberofvertices, 1);
pos = find(md.inversion.vx_obs < -8e3 | md.inversion.vy_obs < -8e3 | md.geometry.thickness < 10);
md.inversion.vx_obs(pos) = 0;
md.inversion.vy_obs(pos) = 0;
md.inversion.vel_obs = sqrt(md.inversion.vx_obs.^2 + md.inversion.vy_obs.^2);

md.initialization.vx = md.inversion.vx_obs;
md.initialization.vy = md.inversion.vy_obs;
md.initialization.vz = zeros(md.mesh.numberofvertices,1);
md.initialization.vel = md.inversion.vel_obs;

md.stressbalance.spcvx = NaN * ones(md.mesh.numberofvertices, 1);
md.stressbalance.spcvy = NaN * ones(md.mesh.numberofvertices, 1);
md.stressbalance.spcvz = NaN * ones(md.mesh.numberofvertices, 1);

pos = find(md.mesh.vertexonboundary & md.geometry.thickness > 10);
md.stressbalance.spcvx(pos) = md.inversion.vx_obs(pos);
md.stressbalance.spcvy(pos) = md.inversion.vy_obs(pos);
md.stressbalance.spcvz = zeros(md.mesh.numberofvertices,1);
plotmodel(md, 'data', md.inversion.vel_obs);


% SMB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_smb = zeros(md.mesh.numberofvertices + 1, 13);
for i = 2007:2009
   i
   smb = double(h5read(file_name3, strcat('/adot_', int2str(i))));
   smb = InterpFromGridToMesh(x, y, smb', md.mesh.x, md.mesh.y, 0);
   all_smb(1:(end-1), i - 2006) = smb;
   all_smb(end, i - 2006) = i;
   
end

for i = 2010:2019
   i
   smb = double(h5read(file_name4, strcat('/adot_', int2str(i))));
   smb = InterpFromGridToMesh(x, y, smb', md.mesh.x, md.mesh.y, 0);
   all_smb(1:(end-1), i - 2006) = smb;
   all_smb(end, i - 2006) = i;
end 

md.smb.mass_balance = all_smb;


% Effective pressure regions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

md.friction.pressure_regions = (1./(1 + exp(-0.0125*(md.geometry.bed + 150))));
pos = find(md.geometry.surface > 1200);
md.friction.pressure_regions(pos) = 1;


% Time Stepping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

md.timestepping.start_time = 2007;
md.timestepping.final_time = 2020;
md.timestepping.time_step = 0.005;
md.timestepping.interp_forcings = 1;


% Transient settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

md.transient.issmb = 1;
md.transient.ismasstransport = 1;
md.transient.isstressbalance = 1;
md.transient.isgroundingline = 1;
md.transient.ismovingfront = 1;
md.settings.output_frequency = 20;
md.transient.requested_outputs={'default','GroundedArea','FloatingArea','TotalFloatingBmb','TotalGroundedBmb','TotalSmb', 'IceVolume', 'IceVolumeAboveFloatation', 'CalvingCalvingrate'};


% Miscelaneous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Levelset
md.levelset.spclevelset = NaN(md.mesh.numberofvertices,1);

% Mass transport
md.masstransport.spcthickness = NaN*ones(md.mesh.numberofvertices,1);

% Thermal
md.thermal.spctemperature = NaN*ones(md.mesh.numberofvertices,1);

% Ice rheology
md.materials.rheology_n = 3*ones(md.mesh.numberofelements,1);
md.materials.rheology_B = paterson(268.15)*ones(md.mesh.numberofelements,1);

% Stress balance stuff
md.stressbalance.referential = NaN * ones(md.mesh.numberofvertices,6);
md.stressbalance.loadingforce = 0 * ones(md.mesh.numberofvertices,3);

% Initial temp
md.initialization.temperature = 268.15 * ones(md.mesh.numberofvertices,1);;

% Sliding params.
md.friction.coefficient = 50 * ones(md.mesh.numberofvertices,1);
md.friction.p = ones(md.mesh.numberofelements,1);
md.friction.q = ones(md.mesh.numberofelements,1);

% Pressure
md.initialization.pressure = md.materials.rho_ice*md.constants.g*md.geometry.thickness;

% Calving
md.calving = calvingvonmises();
md.calving.stress_threshold_groundedice = 1000e3;
md.calving.stress_threshold_floatingice = 200e3;
md.calving.min_thickness = 10;

% Basal forcings
md.basalforcings.groundedice_melting_rate = zeros(md.mesh.numberofvertices,1);
md.basalforcings.floatingice_melting_rate = zeros(md.mesh.numberofvertices,1);
md.basalforcings.geothermalflux = 0.0707*ones(md.mesh.numberofvertices,1);

% Frontal forcings
md.frontalforcings.meltingrate = zeros(md.mesh.numberofvertices,1);

% Flow equation
md = setflowequation(md,'SSA','all');

% Water fraction
md.initialization.waterfraction = zeros(md.mesh.numberofvertices,1);

% Save model
save model1 md;
