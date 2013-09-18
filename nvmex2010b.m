function nvmex2011a(cuFileName)
%NVMEX Compiles and links a CUDA file for MATLAB usage
%   NVMEX(FILENAME) will create a MEX-File (also with the name FILENAME) by
%   invoking the CUDA compiler, nvcc, and then linking with the MEX
%   function in MATLAB.

% Copyright 2009 The MathWorks, Inc.

% !!! Modify the paths below to fit your own installation !!!
if ispc % Windows
    CUDA_LIB_Location = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\lib\x64';
    Host_Compiler_Location = '-ccbin "C:\VisualStudio9\VC\bin"';
    PIC_Option = '';
else % Mac and Linux (assuming gcc is on the path)
    CUDA_LIB_Location = '/usr/local/cuda/lib64';
    Host_Compiler_Location = '';
    PIC_Option = ' --compiler-options -fPIC';
end
% !!! End of things to modify !!!

[dummy, filename] = fileparts(cuFileName);

nvccCommandLine = [ ...
    'nvcc --compile ' cuFileName ' ' Host_Compiler_Location ' ' ...
    '-m64 ' ' -o ' filename '.o '  ...
    PIC_Option ...
    ' -I' '"/opt/matlabR2010b/extern/include"' ...
    ];

mexCommandLine = ['mex (''' filename '.o'', ''-L' CUDA_LIB_Location ''', ''-lcudart'', ''-lcurand'')'];

disp(nvccCommandLine);
status = system(nvccCommandLine);
if status < 0
    error 'Error invoking nvcc';
end

disp(mexCommandLine);
eval(mexCommandLine);

end