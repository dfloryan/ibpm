clear

% Run ibpm from matlab
motionfile = 'testmotion.txt';
nameNonlinear = 'h0.2_f0.5_';
nameAdjoint = 'h0.2_f0.5_adj_';
outdir = '.';
tecplot = 0;
restartNonlinear = 1;
restartAdjoint = 0;
forceNonlinear = 1;
forceAdjoint = 0;
nx = 300;
ny = 300;
ngrid = 1;
length = 6;
xoffset = -1;
yoffset = -3;
geom = 'plate4.geom';
ubfNonlinear = 1;
ubfAdjoint = 0;
Re = 100;
modelNonlinear = 'nonlinear';
modelAdjoint = 'adjoint2';
icNonlinear = 'h0.2_f0.5_00000.bin';
dt = 0.01;
nsteps = 200;
period = 200;
pbaseflowname = 'h0.2_f0.5_%05d.bin';

command = ['./simple_withoptions',...
    ' -nameNonlinear ',nameNonlinear,...
    ' -nameAdjoint ',nameAdjoint,...
    ' -outdir ',outdir,...
    ' -tecplot ',num2str(tecplot),...
    ' -restartNonlinear ',num2str(restartNonlinear),...
    ' -restartAdjoint ',num2str(restartAdjoint),...
    ' -forceNonlinear ',num2str(forceNonlinear),...
    ' -forceAdjoint ',num2str(forceAdjoint),...
    ' -nx ',num2str(nx),...
    ' -ny ',num2str(ny),...
    ' -ngrid ',num2str(ngrid),...
    ' -length ',num2str(length),...
    ' -xoffset ',num2str(xoffset),...
    ' -yoffset ',num2str(yoffset),...
    ' -geom ',num2str(geom),...
    ' -motionfile ',motionfile,...
    ' -ubfNonlinear ',num2str(ubfNonlinear),...
    ' -ubfAdjoint ',num2str(ubfAdjoint),...
    ' -Re ',num2str(Re),...
    ' -modelNonlinear ',num2str(modelNonlinear),...
    ' -modelAdjoint ',num2str(modelAdjoint),...
    ' -icNonlinear ',icNonlinear,...
    ' -dt ',num2str(dt),...
    ' -nsteps ',num2str(nsteps),...
    ' -period ',num2str(period),...
    ' -pbaseflowname ',pbaseflowname];

system(command)
