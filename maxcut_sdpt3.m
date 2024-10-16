tic;
fileID = fopen("sdpt3_path.txt", 'r');
if fileID == -1
    error('Cannot open the file: sdpt3_path.txt');
end
SDPT3_PATH = strtrim(fscanf(fileID, '%c'));
fclose(fileID);
addpath(genpath(SDPT3_PATH)); 
OPTIONS.gaptol = 1.e-9;

current_directory = pwd;
M = readmatrix(current_directory+"/matrix.txt");

nb_nodes = size(M,1);
C = spdiags(M*ones(nb_nodes,1),0,nb_nodes,nb_nodes) - M;
clearvars M;
C = -0.125*(C+C');
b = ones(nb_nodes,1);

blk{1,1} = 's';
blk{1,2} = nb_nodes;
A = cell(1,nb_nodes);
for k = 1:nb_nodes
    A{k} = sparse(k,k,1,nb_nodes,nb_nodes);
end
isspx = ones(size(blk,1),1);
Avec = svec(blk,A,isspx);
clearvars A;

seed = 42;
rng(seed,'twister');
[~,X,~,~,~,~] = sdpt3(blk,Avec,C,b,OPTIONS);
writematrix(full(cell2mat(X)),current_directory+"/sdpt3_result.txt")
time = toc;
fileID = fopen(current_directory+"/sdpt3_time.txt",'w');
fprintf(fileID,'%6.2f',time);
fclose(fileID);