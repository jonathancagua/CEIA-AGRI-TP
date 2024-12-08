#include <stdio.h>
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_dl.h"
#include "esp_nn.h"

// Configuración del modelo
#define MODEL_PATH "/spiffs/best.onnx"
#define INPUT_WIDTH 320
#define INPUT_HEIGHT 320
#define INPUT_CHANNELS 3

void init_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_320X320;  // Tamaño de entrada para YOLO
    config.jpeg_quality = 10;
    config.fb_count = 1;

    // Inicializar la cámara
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE("CAMERA", "Error inicializando la cámara: %s", esp_err_to_name(err));
    } else {
        ESP_LOGI("CAMERA", "Cámara inicializada");
    }
}

// Preprocesar la imagen
void preprocess_image(camera_fb_t *fb, float *input_buffer) {
    uint8_t *image_data = fb->buf;
    for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS; i++) {
        input_buffer[i] = (float)image_data[i] / 255.0f;  // Normalizar entre 0 y 1
    }
}

// Ejecutar inferencia
void run_inference(dl_matrix3d_t *input, dl_matrix3d_t *output) {
    dl_matrix3d_t *model_output = esp_dl_forward(input);
    if (model_output) {
        memcpy(output->item, model_output->item, output->n * sizeof(float));
        ESP_LOGI("INFERENCE", "Inferencia completada");
    } else {
        ESP_LOGE("INFERENCE", "Error en la inferencia");
    }
}

void app_main(void) {
    // Inicializar cámara
    init_camera();

    // Inicializar esp-dl
    esp_dl_model_config_t config = {
        .model_path = MODEL_PATH,
    };
    esp_dl_model_t *model = esp_dl_model_load(&config);
    if (!model) {
        ESP_LOGE("MODEL", "Error cargando el modelo");
        return;
    }

    // Capturar una imagen
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE("CAMERA", "Error capturando la imagen");
        return;
    }

    // Preparar entrada y salida
    dl_matrix3d_t *input = dl_matrix3d_alloc(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
    preprocess_image(fb, input->item);
    dl_matrix3d_t *output = dl_matrix3d_alloc(1, 1, 3);  // Ajustar según las clases

    // Ejecutar inferencia
    run_inference(input, output);

    // Interpretar resultados
    int class_id = 0;
    float max_score = 0.0f;
    for (int i = 0; i < output->n; i++) {
        if (output->item[i] > max_score) {
            max_score = output->item[i];
            class_id = i;
        }
    }

    ESP_LOGI("RESULT", "Clase predicha: %d con probabilidad %.2f", class_id, max_score);

    // Liberar recursos
    dl_matrix3d_free(input);
    dl_matrix3d_free(output);
    esp_camera_fb_return(fb);
    esp_dl_model_unload(model);
}
