#include "OGLTemplate.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math_constants.h>
#include "helper_timer.h"

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cufft.lib")

extern "C" void cudaGenerateSpectrumKernel(float2* d_h0, float2* d_ht,
	unsigned int in_width,
	unsigned int out_width,
	unsigned int out_height,
	float animTime, float patchSize);

extern "C" void cudaUpdateHeightmapKernel(float* d_heightMap, float2* d_ht,
	unsigned int width,
	unsigned int height, bool autoTest);

extern "C" void cudaCalculateSlopeKernel(float* h, float2* slopeOut,
	unsigned int width,
	unsigned int height);

const unsigned int meshSize = 256;
const unsigned int spectrumW = meshSize + 4;
const unsigned int spectrumH = meshSize + 1;

// FFT data
cufftHandle fftPlan;
float2* d_h0 = 0;  // heightfield at time 0
float2* h_h0 = 0;
float2* d_ht = 0;  // heightfield at time t
float2* d_slope = 0;

// pointers to device object
float* g_hptr = NULL;
float2* g_sptr = NULL;

// simulation parameters
const float g = 9.81f;        // gravitational constant
const float A = 1e-7f;        // wave scale factor
const float patchSize = 100;  // patch size
float windSpeed = 100.0f;
float windDir = CUDART_PI_F / 3.0f;
float dirDepend = 0.07f;

GLuint heightVertexBuffer, slopeVertexBuffer;
GLuint posVertexBuffer;
GLuint indexBuffer;
cudaError_t error;

StopWatchInterface* timer = NULL;
float animTime = 0.0f;
float prevTime = 0.0f;
float animationRate = -0.001f;
bool animate = true;

struct cudaGraphicsResource* cuda_posVB_resource, * cuda_heightVB_resource, * cuda_slopeVB_resource;  // handles OpenGL-CUDA exchange

// Set default uniform variables parameters for the vertex shader
GLuint uniHeightScale, uniChopiness, uniSize;

// Beginning of GPU Architecture definitions
static  int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{0x80,  64},
		{0x86, 128},
		{0x87, 128},
		{0x89, 128},
		{0x90, 128},
		{-1, -1} };

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

static const char* _ConvertSMVer2ArchName(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the GPU Arch name)
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		const char* name;
	} sSMtoArchName;

	sSMtoArchName nGpuArchNameSM[] = {
		{0x30, "Kepler"},
		{0x32, "Kepler"},
		{0x35, "Kepler"},
		{0x37, "Kepler"},
		{0x50, "Maxwell"},
		{0x52, "Maxwell"},
		{0x53, "Maxwell"},
		{0x60, "Pascal"},
		{0x61, "Pascal"},
		{0x62, "Pascal"},
		{0x70, "Volta"},
		{0x72, "Xavier"},
		{0x75, "Turing"},
		{0x80, "Ampere"},
		{0x86, "Ampere"},
		{0x87, "Ampere"},
		{0x89, "Ada"},
		{0x90, "Hopper"},
		{-1, "Graphics Device"} };

	int index = 0;

	while (nGpuArchNameSM[index].SM != -1) {
		if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchNameSM[index].name;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoArchName for SM %d.%d is undefined."
		"  Default to use %s\n",
		major, minor, nGpuArchNameSM[index - 1].name);
	return nGpuArchNameSM[index - 1].name;
}
// end of GPU Architecture definitions


// This function returns the best GPU (with maximum GFLOPS)
static int gpuGetMaxGflopsDeviceId() {
	int current_device = 0, sm_per_multiproc = 0;
	int max_perf_device = 0;
	int device_count = 0;
	int devices_prohibited = 0;

	uint64_t max_compute_perf = 0;
	cudaGetDeviceCount(&device_count);

	if (device_count == 0) {
		fprintf(stderr,
			"gpuGetMaxGflopsDeviceId() CUDA error:"
			" no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	// Find the best CUDA capable GPU device
	current_device = 0;

	while (current_device < device_count) {
		int computeMode = -1, major = 0, minor = 0;
		cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device);
		cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device);
		cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device);

		// If this GPU is not running on Compute Mode prohibited,
		// then we can add it to the list
		if (computeMode != cudaComputeModeProhibited) {
			if (major == 9999 && minor == 9999) {
				sm_per_multiproc = 1;
			}
			else {
				sm_per_multiproc =
					_ConvertSMVer2Cores(major, minor);
			}
			int multiProcessorCount = 0, clockRate = 0;
			cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device);
			cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
			if (result != cudaSuccess) {
				// If cudaDevAttrClockRate attribute is not supported we
				// set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
				if (result == cudaErrorInvalidValue) {
					clockRate = 1;
				}
				else {
					//fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,static_cast<unsigned int>(result), _cudaGetErrorEn(result));
					exit(EXIT_FAILURE);
				}
			}
			uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

			if (compute_perf > max_compute_perf) {
				max_compute_perf = compute_perf;
				max_perf_device = current_device;
			}
		}
		else {
			devices_prohibited++;
		}

		++current_device;
	}

	if (devices_prohibited == device_count) {
		fprintf(stderr,
			"gpuGetMaxGflopsDeviceId() CUDA error:"
			" all devices have compute mode prohibited.\n");
		exit(EXIT_FAILURE);
	}

	return max_perf_device;
}

void SelectCUDADevice()
{
	int devID = gpuGetMaxGflopsDeviceId();
	cudaSetDevice(devID);
	int major = 0, minor = 0;
	cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
	cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
	fprintf_s(gpFile, "GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, _ConvertSMVer2ArchName(major, minor), major, minor);
}

float urand() { return rand() / (float)RAND_MAX; }

// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss() {
	float u1 = urand();
	float u2 = urand();

	if (u1 < 1e-6f) {
		u1 = 1e-6f;
	}

	return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
}

// Phillips spectrum
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// A - constant
float phillips(float Kx, float Ky, float Vdir, float V, float A,
	float dir_depend) {
	float k_squared = Kx * Kx + Ky * Ky;

	if (k_squared == 0.0f) {
		return 0.0f;
	}

	// largest possible wave from constant wind of velocity v
	float L = V * V / g;

	float k_x = Kx / sqrtf(k_squared);
	float k_y = Ky / sqrtf(k_squared);
	float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

	float phillips = A * expf(-1.0f / (k_squared * L * L)) /
		(k_squared * k_squared) * w_dot_k * w_dot_k;

	// filter out waves moving opposite to wind
	if (w_dot_k < 0.0f) {
		phillips *= dir_depend;
	}

	// damp out waves with very small length w << l
	// float w = L / 10000;
	// phillips *= expf(-k_squared * w * w);

	return phillips;
}

// Generate base heightfield in frequency space
void generate_h0(float2* h0) {
	for (unsigned int y = 0; y <= meshSize; y++) {
		for (unsigned int x = 0; x <= meshSize; x++) {
			float kx = (-(int)meshSize / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
			float ky = (-(int)meshSize / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

			float P = sqrtf(phillips(kx, ky, windDir, windSpeed, A, dirDepend));

			if (kx == 0.0f && ky == 0.0f) {
				P = 0.0f;
			}

			// float Er = urand()*2.0f-1.0f;
			// float Ei = urand()*2.0f-1.0f;
			float Er = gauss();
			float Ei = gauss();

			float h0_re = Er * P * CUDART_SQRT_HALF_F;
			float h0_im = Ei * P * CUDART_SQRT_HALF_F;

			int i = y * spectrumW + x;
			h0[i].x = h0_re;
			h0[i].y = h0_im;
		}
	}
}

// create fixed vertex buffer to store mesh vertices
void createMeshPositionVBO()
{
	//createVBO(id, w * h * 4 * sizeof(float));

	// create vbo for position
	glGenBuffers(1, &posVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, posVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, meshSize * meshSize * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	//glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	//glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, posVertexBuffer);
	float* pos = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (!pos) {
		return;
	}

	for (int y = 0; y < meshSize; y++) {
		for (int x = 0; x < meshSize; x++) {
			float u = x / (float)(meshSize - 1);
			float v = y / (float)(meshSize - 1);
			*pos++ = u * 2.0f - 1.0f;
			*pos++ = 0.0f;
			*pos++ = v * 2.0f - 1.0f;
			*pos++ = 1.0f;
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// create index buffer for rendering quad mesh
void createMeshIndexBuffer(GLuint* id, int w, int h) {
	int size = ((meshSize * 2) + 2) * (meshSize - 1) * sizeof(GLuint);

	// create index buffer
	glGenBuffers(1, id);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	// fill with indices for rendering mesh as triangle strips
	GLuint* indices =
		(GLuint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (!indices) {
		return;
	}

	for (int y = 0; y < meshSize - 1; y++) {
		for (int x = 0; x < meshSize; x++) {
			*indices++ = y * meshSize + x;
			*indices++ = (y + 1) * meshSize + x;
		}

		// start new strip with degenerate triangle
		*indices++ = (y + 1) * meshSize + (meshSize - 1);
		*indices++ = (y + 1) * meshSize;
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda kernels
////////////////////////////////////////////////////////////////////////////////
void runCuda() {
	size_t num_bytes;

	// generate wave spectrum in frequency domain
	cudaGenerateSpectrumKernel(d_h0, d_ht, spectrumW, meshSize, meshSize, animTime, patchSize);

	// execute inverse FFT to convert to spatial domain
	cufftExecC2C(fftPlan, d_ht, d_ht, CUFFT_INVERSE);

	// update heightmap values in vertex buffer
	cudaGraphicsMapResources(1, &cuda_heightVB_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&g_hptr, &num_bytes, cuda_heightVB_resource);

	cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize, false);

	// calculate slope for shading
	cudaGraphicsMapResources(1, &cuda_slopeVB_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&g_sptr, &num_bytes, cuda_slopeVB_resource);

	cudaCalculateSlopeKernel(g_hptr, g_sptr, meshSize, meshSize);

	cudaGraphicsUnmapResources(1, &cuda_heightVB_resource, 0);
	cudaGraphicsUnmapResources(1, &cuda_slopeVB_resource, 0);
}

// WinMain()
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	// function declarations
	void initialize(void);
	void display(void);
	void update(void);

	// variable declarations
	bool bDone = false;
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MyApp");

	// code
	// open file for logging
	if (fopen_s(&gpFile, "AMCLog.txt", "w") != 0)
	{
		MessageBox(NULL, TEXT("Cannot open AMCLog.txt file.."), TEXT("Error"), MB_OK | MB_ICONERROR);
		exit(0);
	}
	fprintf(gpFile, "==== Application Started : Ganesha ====\n");

	// initialization of WNDCLASSEX
	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(hInstance,IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;
	wndclass.hIconSm = LoadIcon(hInstance, IDI_APPLICATION);

	// register above class
	RegisterClassEx(&wndclass);

	// get the screen size
	int width = GetSystemMetrics(SM_CXSCREEN);
	int height = GetSystemMetrics(SM_CYSCREEN);

	// create window
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName,
		TEXT("OpenGL Application : River"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		(width / 2) - 400,
		(height / 2) - 300,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	initialize();

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	// Game Loop!
	while (bDone == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
				bDone = true;
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}
		else
		{
			if (gbActiveWindow == true)
			{
				// call update() here for OpenGL rendering
				update();
				// call display() here for OpenGL rendering
				display();
			}
		}
	}

	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	// function declaration
	void display(void);
	void resize(int, int);
	void uninitialize();
	void ToggleFullscreen(void);

	// code
	switch (iMsg)
	{

	case WM_SETFOCUS:
		gbActiveWindow = true;
		break;

	case WM_KILLFOCUS:
		gbActiveWindow = false;
		break;

	case WM_SIZE:
		resize(LOWORD(lParam), HIWORD(lParam));
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			DestroyWindow(hwnd);
			break;

		case 0x46:
		case 0x66:
			ToggleFullscreen();
			break;

		default:
			break;
		}
		break;

	case WM_ERASEBKGND:
		return(0);

	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		uninitialize();
		PostQuitMessage(0);
		break;

	default:
		break;
	}

	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}

void initialize(void)
{
	// function declarations
	void resize(int, int);
	void uninitialize(void);

	// variable declarations
	PIXELFORMATDESCRIPTOR pfd;
	int iPixelFormatIndex;

	// code
	ghdc = GetDC(ghwnd);

	ZeroMemory((void *)&pfd, sizeof(PIXELFORMATDESCRIPTOR));
	pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL| PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 32;
	pfd.cRedBits = 8;
	pfd.cGreenBits = 8;
	pfd.cBlueBits = 8;
	pfd.cAlphaBits = 8;
	pfd.cDepthBits = 32;

	iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
	if (iPixelFormatIndex == 0)
	{
		fprintf(gpFile, "ChoosePixelFormat() failed..\n");
		DestroyWindow(ghwnd);
	}

	if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
	{
		fprintf(gpFile, "SetPixelFormat() failed..\n");
		DestroyWindow(ghwnd);
	}

	ghrc = wglCreateContext(ghdc);
	if (ghrc == NULL)
	{
		fprintf(gpFile, "wglCreateContext() failed..\n");
		DestroyWindow(ghwnd);
	}

	if (wglMakeCurrent(ghdc, ghrc) == FALSE)
	{
		fprintf(gpFile, "wglMakeCurrent() failed..\n");
		DestroyWindow(ghwnd);
	}

	// glew initialization for programmable pipeline
	GLenum glew_error = glewInit();
	if (glew_error != GLEW_OK)
	{
		fprintf(gpFile, "glewInit() failed..\n");
		DestroyWindow(ghwnd);
	}

	// fetch OpenGL related details
	fprintf(gpFile, "OpenGL Vendor:   %s\n", glGetString(GL_VENDOR));
	fprintf(gpFile, "OpenGL Renderer: %s\n", glGetString(GL_RENDERER));
	fprintf(gpFile, "OpenGL Version:  %s\n", glGetString(GL_VERSION));
	fprintf(gpFile, "GLSL Version:    %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	// fetch OpenGL enabled extensions
	GLint numExtensions;
	glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);

	fprintf(gpFile, "==== OpenGL Extensions ====\n");
	for (int i = 0; i < numExtensions; i++)
	{
		fprintf(gpFile, "  %s\n", glGetStringi(GL_EXTENSIONS, i));
	}
	fprintf(gpFile, "===========================\n\n");

	//// vertex shader
	// create shader
	GLuint gVertexShaderObject = glCreateShader(GL_VERTEX_SHADER);

	// provide source code to shader
	GLchar* vertexShaderSourceCode = NULL;
	GLchar* VS = (GLchar*)malloc((strlen(SHADER_PATH) + 1) * 1000 * sizeof(GLchar));
	if (VS == NULL)
	{
		exit(-1);
	}
	else
	{
		strcpy_s(VS, ((strlen(SHADER_PATH) + 1) * 1000 * sizeof(GLchar)), SHADER_PATH);
		strcat_s(VS, 1000, VERTEX_SHADER);
		PassShaderinString(VS, &vertexShaderSourceCode);
		glShaderSource(gVertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);
	}

	// compile shader
	glCompileShader(gVertexShaderObject);

	// compilation errors 
	GLint iShaderCompileStatus = 0;
	GLint iInfoLogLength = 0;
	GLchar *szInfoLog = NULL;

	glGetShaderiv(gVertexShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gVertexShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gVertexShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, "Vertex Shader Compiler Info Log: \n%s\n", szInfoLog);
				free(szInfoLog);
				DestroyWindow(ghwnd);
			}
		}
	}

	//// fragment shader
	// create shader
	GLuint gFragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

	GLchar* fragmentShaderSourceCode = NULL;
	GLchar* FS = (GLchar*)malloc((strlen(SHADER_PATH) + 1) * 1000 * sizeof(GLchar));
	if (FS == NULL)
	{
		exit(-1);
	}
	else
	{
		strcpy_s(FS, ((strlen(SHADER_PATH) + 1) * 1000 * sizeof(GLchar)), SHADER_PATH);
		strcat_s(FS, 1000, FRAGMENT_SHADER);
		PassShaderinString(FS, &fragmentShaderSourceCode);
		glShaderSource(gFragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);
	}

	// compile shader
	glCompileShader(gFragmentShaderObject);

	// compile errors
	iShaderCompileStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetShaderiv(gFragmentShaderObject, GL_COMPILE_STATUS, &iShaderCompileStatus);
	if (iShaderCompileStatus == GL_FALSE)
	{
		glGetShaderiv(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetShaderInfoLog(gFragmentShaderObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, "Fragment Shader Compiler Info Log: \n%s\n", szInfoLog);
				free(szInfoLog);
				DestroyWindow(ghwnd);
			}
		}
	}

	//// shader program
	// create
	gShaderProgramObject = glCreateProgram();

	// attach shaders
	glAttachShader(gShaderProgramObject, gVertexShaderObject);
	glAttachShader(gShaderProgramObject, gFragmentShaderObject);

	// pre-linking binding to vertex attribute
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_POSITION, "vPosition");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_HEIGHT, "heightAttribute");
	glBindAttribLocation(gShaderProgramObject, AMC_ATTRIBUTE_SLOPE, "slopeAttribute");

	// link shader
	glLinkProgram(gShaderProgramObject);

	// linking errors
	GLint iProgramLinkStatus = 0;
	iInfoLogLength = 0;
	szInfoLog = NULL;

	glGetProgramiv(gShaderProgramObject, GL_LINK_STATUS, &iProgramLinkStatus);
	if (iProgramLinkStatus == GL_FALSE)
	{
		glGetProgramiv(gShaderProgramObject, GL_INFO_LOG_LENGTH, &iInfoLogLength);
		if (iInfoLogLength > 0)
		{
			szInfoLog = (GLchar *)malloc(iInfoLogLength);
			if (szInfoLog != NULL)
			{
				GLsizei written;
				glGetProgramInfoLog(gShaderProgramObject, GL_INFO_LOG_LENGTH, &written, szInfoLog);

				fprintf(gpFile, ("Shader Program Linking Info Log: \n%s\n"), szInfoLog);
				free(szInfoLog);
				DestroyWindow(ghwnd);
			}
		}
	}

	// post-linking retrieving uniform locations
	mvpMatrixUniform = glGetUniformLocation(gShaderProgramObject, "u_mvpMatrix");

	SelectCUDADevice();
	cufftPlan2d(&fftPlan, meshSize, meshSize, CUFFT_C2C);

	// allocate memory
	int spectrumSize = spectrumW * spectrumH * sizeof(float2);
	cudaMalloc((void**)&d_h0, spectrumSize);
	h_h0 = (float2*)malloc(spectrumSize);
	generate_h0(h_h0);
	cudaMemcpy(d_h0, h_h0, spectrumSize, cudaMemcpyHostToDevice);

	int outputSize = meshSize * meshSize * sizeof(float2);
	cudaMalloc((void**)&d_ht, outputSize);

	// create vao for square
	glGenVertexArrays(1, &vaoSquare);
	glBindVertexArray(vaoSquare);

	/*
	// create vbo for position
	glGenBuffers(1, &vboPositionSquare);
	glBindBuffer(GL_ARRAY_BUFFER, vboPositionSquare);
	glBufferData(GL_ARRAY_BUFFER, sizeof(squareVertices), squareVertices, GL_STATIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	*/

	// Generate and bind the buffer
	glGenBuffers(1, &heightVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, heightVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, meshSize * meshSize * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_HEIGHT, 1, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_HEIGHT);

	// register our vbo with cuda graphics resource
	error = cudaGraphicsGLRegisterBuffer(&cuda_heightVB_resource, heightVertexBuffer,cudaGraphicsMapFlagsWriteDiscard);
	if (error != cudaSuccess) {
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed : %s..\n", cudaGetErrorString(error));
		uninitialize();
		DestroyWindow(ghwnd);
	}

	// create vbo for slope
	glGenBuffers(1, &slopeVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, slopeVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, outputSize, 0, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(AMC_ATTRIBUTE_SLOPE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_SLOPE);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register our vbo with cuda graphics resource
	error = cudaGraphicsGLRegisterBuffer(&cuda_slopeVB_resource, slopeVertexBuffer,
		cudaGraphicsMapFlagsWriteDiscard);
	if (error != cudaSuccess) {
		fprintf(gpFile, "cudaGraphicsGLRegisterBuffer failed : %s..\n", cudaGetErrorString(error));
		uninitialize();
		DestroyWindow(ghwnd);
	}

	// create vertex and index buffer for mesh
	createMeshPositionVBO();
	createMeshIndexBuffer(&indexBuffer, meshSize, meshSize);

	// Unbind the buffer
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	// set clear color
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// set clear depth
	glClearDepth(1.0f);

	// depth test
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	perspectiveProjectionMatrix = mat4::identity();

	// warm-up resize call
	resize(WIN_WIDTH, WIN_HEIGHT);
}

void resize(int width, int height)
{
	// code
	if (height == 0)
		height = 1;

	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	perspectiveProjectionMatrix = vmath::perspective(60.0f, (float)width/(float)height, 0.1f, 10.0f);
}

void display(void)
{
	// code
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// start using OpenGL program object
	glUseProgram(gShaderProgramObject);

	if (animate) {
		runCuda();
	}

	uniHeightScale = glGetUniformLocation(gShaderProgramObject, "heightScale");
	glUniform1f(uniHeightScale, 0.09f);

	uniChopiness = glGetUniformLocation(gShaderProgramObject, "chopiness");
	glUniform1f(uniChopiness, 1.0f); //1.0f

	uniSize = glGetUniformLocation(gShaderProgramObject, "size");
	glUniform2f(uniSize, (float)meshSize, (float)meshSize);

	// Set default uniform variables parameters for the pixel shader
	GLuint uniDeepColor, uniShallowColor, uniSkyColor, uniLightDir;

	uniDeepColor = glGetUniformLocation(gShaderProgramObject, "deepColor");
	glUniform4f(uniDeepColor, 0.0f, 0.5f, 0.4f, 1.0f);

	uniShallowColor = glGetUniformLocation(gShaderProgramObject, "shallowColor");
	glUniform4f(uniShallowColor, 0.1f, 0.3f, 0.3f, 1.0f);

	uniSkyColor = glGetUniformLocation(gShaderProgramObject, "skyColor");
	glUniform4f(uniSkyColor, 1.0f, 1.0f, 1.0f, 1.0f);

	uniLightDir = glGetUniformLocation(gShaderProgramObject, "lightDir");
	glUniform3f(uniLightDir, 0.0f, 1.0f, 0.0f);
	// end of uniform settings

	//declaration of matrices
	mat4 translateMatrix;
	mat4 scaleMatrix;
	mat4 modelViewMatrix;
	mat4 modelViewProjectionMatrix;

	//// square ////////////////////////

	// intialize above matrices to identity
	translateMatrix = mat4::identity();
	modelViewMatrix = mat4::identity();
	scaleMatrix = mat4::identity();
	modelViewProjectionMatrix = mat4::identity();

	// transformations
	translateMatrix = translate(0.0f, -0.5f, -3.0f);
	//scaleMatrix = scale(1.5f,1.5f,1.5f);
	modelViewMatrix = translateMatrix * scaleMatrix;

	// do necessary matrix multiplication
	modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

	// send necessary matrices to shader in respective uniforms
	glUniformMatrix4fv(mvpMatrixUniform, 1, GL_FALSE, modelViewProjectionMatrix);

	// bind with vaoTriangle (this will avoid many binding to vbo)
	glBindVertexArray(vaoSquare);  

	// draw necessary scene

	glBindBuffer(GL_ARRAY_BUFFER, posVertexBuffer);
	glVertexAttribPointer(AMC_ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_POSITION);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, heightVertexBuffer);
	glVertexAttribPointer(AMC_ATTRIBUTE_HEIGHT, 1, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_HEIGHT);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, slopeVertexBuffer);
	glVertexAttribPointer(AMC_ATTRIBUTE_SLOPE, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(AMC_ATTRIBUTE_SLOPE);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	//glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	glDrawElements(GL_TRIANGLE_STRIP, ((meshSize * 2) + 2) * (meshSize - 1),GL_UNSIGNED_INT, 0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// unbind vaoTriangle
	glBindVertexArray(0);

	// stop using OpenGL program object
	glUseProgram(0);

	SwapBuffers(ghdc);
}

void update(void)
{
	// code
	animTime = animTime + 0.1f;
}

void uninitialize(void)
{
	// code
	if (gbFullscreen == true)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

		ShowCursor(TRUE);
	}

	if (vaoSquare)
	{
		glDeleteVertexArrays(1, &vaoSquare);
		vaoSquare = 0;
	}

	if (vboPositionSquare)
	{
		glDeleteBuffers(1, &vboPositionSquare);
		vboPositionSquare = 0;
	}

	// destroy shader programs
	if (gShaderProgramObject)
	{
		GLsizei shaderCount;
		GLsizei i;

		glUseProgram(gShaderProgramObject);
		glGetProgramiv(gShaderProgramObject, GL_ATTACHED_SHADERS, &shaderCount);
		
		GLuint *pShaders = (GLuint*) malloc(shaderCount * sizeof(GLuint));
		if (pShaders)
		{
			glGetAttachedShaders(gShaderProgramObject, shaderCount, &shaderCount, pShaders);

			for (i = 0; i < shaderCount; i++)
			{
				// detach shader
				glDetachShader(gShaderProgramObject, pShaders[i]);

				// delete shader
				glDeleteShader(pShaders[i]);
				pShaders[i] = 0;
			}

			free(pShaders);
		}

		glDeleteProgram(gShaderProgramObject);
		gShaderProgramObject = 0;
		glUseProgram(0);
	}

	if (wglGetCurrentContext() == ghrc)
	{
		wglMakeCurrent(NULL, NULL);
	}

	if (ghrc)
	{
		wglDeleteContext(ghrc);
		ghrc = NULL;
	}

	if (ghdc)
	{
		ReleaseDC(ghwnd, ghdc);
		ghdc = NULL;
	}

	if (gpFile)
	{
		fprintf(gpFile, "==== Application Terminated ====\n");
		fclose(gpFile);
		gpFile = NULL;
	}
}


