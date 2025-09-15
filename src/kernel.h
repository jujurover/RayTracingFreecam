#pragma once

void initPBO(int width, int height);
void runCuda(int frame, int width, int height);
void display(int width, int height);
void initTexture(int width, int height);
void cleanupPBO();
void cleanupGraphics();
void initScreenQuad();
void initScreenShader();
void setupScene(int width, int height);
void updateCamera(float pitch, float yaw);