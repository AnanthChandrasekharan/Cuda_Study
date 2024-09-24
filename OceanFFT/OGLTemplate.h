#pragma once
// headers
#include <windows.h>
#include <stdio.h>
#include <string.h>

#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

#include <gl/glew.h> 
#include <gl/GL.h>

#include "vmath.h"

#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")

// macros
#define WIN_WIDTH  800
#define WIN_HEIGHT 600

using namespace vmath;

enum {
	AMC_ATTRIBUTE_POSITION = 0,
	AMC_ATTRIBUTE_HEIGHT,
	AMC_ATTRIBUTE_SLOPE
};

// global function declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// global variables
HDC   ghdc = NULL;
HGLRC ghrc = NULL;

bool gbFullscreen = false;
bool gbActiveWindow = false;

HWND  ghwnd = NULL;
FILE* gpFile = NULL;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

GLuint gShaderProgramObject;

GLuint vaoSquare;
GLuint vboPositionSquare;
GLuint vboColorSquare;

GLuint mvpMatrixUniform;

mat4 perspectiveProjectionMatrix;

// position array of square
const GLfloat squareVertices[] = {
	 1.0f,  1.0f, 0.0f,
	-1.0f,  1.0f, 0.0f,
	-1.0f, -1.0f, 0.0f,
	 1.0f, -1.0f, 0.0f
};

const GLfloat squareColor[] = {
	 1.0f,  0.0f, 0.0f,
	 0.0f,  1.0f, 0.0f,
	 0.0f,  0.0f, 1.0f,
	 1.0f,  1.0f, 1.0f
};

#define RMC_ICON 101

#define SHADER_PATH (GLchar*)"C:\\Users\\anant\\OneDrive\\Documents\\Anjaneya\\Shaders\\"
#define VERTEX_SHADER (GLchar*)"VS.txt"
#define FRAGMENT_SHADER (GLchar*)"FS.txt"

inline void PassShaderinString(const char* fileName, GLchar** buffer)
{
	FILE* f = NULL;
	if (fopen_s(&f, fileName, "rb") == 0)
	{
		fseek(f, (long)0, SEEK_END);
		long fsize = ftell(f);
		fseek(f, 0, SEEK_SET);

		*buffer = (char*)malloc((fsize + 1) * sizeof(char));
		if (*buffer != NULL)
		{
			fread_s((void*)*buffer, (fsize + 1) * sizeof(char), 1, (fsize + 1), f);
			fclose(f);
			(*buffer)[fsize] = 0;
		}
	}
}

inline void ToggleFullscreen(void)
{
	// local variables
	MONITORINFO mi = { sizeof(MONITORINFO) };

	// code
	if (gbFullscreen == false)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			if (GetWindowPlacement(ghwnd, &wpPrev) &&
				GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd, HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		gbFullscreen = true;
	}
	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd, HWND_TOP,
			0, 0, 0, 0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);
		ShowCursor(TRUE);
		gbFullscreen = false;
	}
}
