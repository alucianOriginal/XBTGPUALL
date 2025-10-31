#pragma once

#include <CL/cl.h>
#include <array>
#include <cstdint>
#include <string>

#define INPUT_SIZE 512
#define HASH_SIZE 32
#define NONCES_PER_THREAD 1
#define BUCKET_COUNT 32
#define HASH_ROUNDS_OUTPUT_SIZE 32

//--Hier werden die Ressourcen der Grafikkarte im Detail eingeteilt.--

struct GpuResources {
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_device_id device = nullptr;
  cl_mem input_buffer = nullptr;
  cl_mem output_buffer = nullptr;
  cl_mem output_hashes_buffer = nullptr;
  cl_mem pool_target_buffer = nullptr;
};

//--Externe Werte mit eingetragen.--

extern int next_request_id;
extern std::string current_job_id;
extern std::string worker_name;
extern bool abort_mining;
extern bool socket_valid;
extern std::array<uint8_t, 32> current_target;
