cls

nvcc -c -o temp.obj oceanFFT_kernel.cu

cl.exe /c /EHsc /I"C:\glew\include" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include" OGLTemplate.cpp 

link.exe /LIBPATH:"C:\glew\lib\Release\x64" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" OGLTemplate.obj temp.obj user32.lib kernel32.lib gdi32.lib

del temp.obj OGLTemplate.obj 

OGLTemplate.exe
