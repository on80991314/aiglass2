#pragma once

// -------- IMU (MPU6050) over I2C --------
#define IMU_SDA_PIN       5
#define IMU_SCL_PIN       6
#define IMU_I2C_ADDR      0x68
#define IMU_SAMPLE_HZ     100

// -------- I2S mic (INMP441, 16-bit mono) --------
#define MIC_I2S_PORT      I2S_NUM_0
#define MIC_CLK_PIN       42     // ���س��J���� Clock
#define MIC_DATA_PIN      41     // ���س��J���� Data
#define MIC_SAMPLE_RATE   16000
#define MIC_FRAME_MS      40     // send a chunk every 40 ms

// -------- I2S speaker (MAX98357A, 16-bit mono) --------
#define SPK_I2S_PORT      I2S_NUM_1
#define SPK_LRC_PIN       1
#define SPK_BCLK_PIN      2
#define SPK_DIN_PIN       42
#define SPK_SAMPLE_RATE   16000
