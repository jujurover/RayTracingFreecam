#pragma once

void initPBO(int width, int height);
void runCuda(int frame, int width, int height);
void display(int width, int height);
void initTexture(int width, int height);
void cleanupPBO();
void cleanupGraphics();
void initScreenQuad();
void clearHighResBuffer();

void initPBOLowRes(int width, int height);
void runCudaLowRes(int frame, int width, int height);
void displayLowRes(int width, int height);
void initTextureLowRes(int width, int height);
void cleanupPBOLowRes();
void initScreenQuadLowRes();

void initScreenShader();
void setupScene(int width, int height);
void stageCameraYawPitch(float pitch, float yaw);
void stageCameraFOV(float fov);
void stageCameraMovement(float right, float up, float forward);
void stageResetCameraWorldUp();
void stageCameraWorldUpRotation(float angle);
void updateCamera();
void cleanupScene();