addpath /home/jake/trunk-jpl/bin/ /home/jake/trunk-jpl/lib/

% Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_name1 = '/media/jake/drive1/greenland_data/h5/geometry/geometry_data.h5';
file_name2 = '/media/jake/drive1/greenland_data/h5/velocity/velocity_data.h5';

x = double(h5read(file_name1, '/x'));
y = double(h5read(file_name1, '/y'));
x = x(: ,1);
y = y(1 ,:)';
velx = double(h5read(file_name2, '/vx'));
vely = double(h5read(file_name2, '/vy'));


% Create mesh
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

md=triangle(model,'data/helheim_domain.exp',1250);
md.miscellaneous.name = 'Helheim';
vx  = InterpFromGridToMesh(x, y, velx', md.mesh.x, md.mesh.y, 0);
vy  = InterpFromGridToMesh(x, y, vely', md.mesh.x, md.mesh.y, 0);
v = sqrt(vx.^2 + vy.^2);
md=bamg(md, 'hmax', 5000, 'hmin', 200, 'gradation', 1.2, 'field', v, 'err', 2.5, 'anisomax', 20);
save model0 md;
