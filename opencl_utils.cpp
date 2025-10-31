#include "opencl_utils.hpp"
#include "globals.hpp"

#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stdexcept>

namespace {

    //--Kleiner Fehler-Helper--

    bool cl_ok(cl_int err, const char* where) {
        if (err == CL_SUCCESS) return true;
        std::cerr << "❌ OpenCL Error " << err << " at " << where << "\n";
        return false;
    }

    //--Lese Datei in string--

    std::string read_file(const std::string &path) {
        std::ifstream ifs(path, std::ios::in | std::ios::binary);
        if (!ifs) throw std::runtime_error("Konnte Datei nicht öffnen: " + path);
        return std::string((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
    }
}

//--Größe-Konstanten (gleiche Werte wie globals.hpp)--

constexpr size_t TARGET_SIZE_BYTES = 32;

void init_opencl(const std::string &kernel_path,
                 const std::string &kernel_func_name,
                 int platform_index,
                 int device_index,
                 int intensity,
                 GpuResources &resources) {

    cl_int err = CL_SUCCESS;

    //--Plattformen--

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (!cl_ok(err, "clGetPlatformIDs(count)")) std::exit(1);
    if (num_platforms == 0) {
        std::cerr << "❌ Keine OpenCL Plattformen gefunden\n";
        std::exit(1);
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (!cl_ok(err, "clGetPlatformIDs(fetch)")) std::exit(1);
    if ((cl_uint)platform_index >= num_platforms) {
        std::cerr << "❌ Ungültiger Plattform-Index\n";
        std::exit(1);
    }

    cl_platform_id platform = platforms[platform_index];

    //--Devices--

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    if (!cl_ok(err, "clGetDeviceIDs(count)")) std::exit(1);
    if (num_devices == 0) {
        std::cerr << "❌ Keine OpenCL Geräte auf Plattform\n";
        std::exit(1);
    }

    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
    if (!cl_ok(err, "clGetDeviceIDs(fetch)")) std::exit(1);

    if ((cl_uint)device_index >= num_devices) {
        std::cerr << "❌ Ungültiger Geräte-Index\n";
        std::exit(1);
    }

    resources.device = devices[device_index];

    //--Kontext & Queue--

    resources.context = clCreateContext(nullptr, 1, &resources.device, nullptr, nullptr, &err);
    if (!cl_ok(err, "clCreateContext")) std::exit(1);

    resources.queue = clCreateCommandQueueWithProperties(resources.context, resources.device, nullptr, &err);
    if (!cl_ok(err, "clCreateCommandQueueWithProperties")) {
        clReleaseContext(resources.context);
        resources.context = nullptr;
        std::exit(1);
    }

    //--Kernel-Quelle lesen--

    std::string src;
    try {
        src = read_file(./XBTGPUARC/kernels/zhash.cl);
    } catch (const std::exception &e) {
        std::cerr << "❌ " << e.what() << "\n";
        cleanup_opencl(resources);
        std::exit(1);
    }

    const char* src_ptr = src.c_str();
    size_t src_size = src.size();

    resources.program = clCreateProgramWithSource(resources.context, 1, &src_ptr, &src_size, &err);
    if (!cl_ok(err, "clCreateProgramWithSource")) {
        cleanup_opencl(resources);
        std::exit(1);
    }

    //--Build-Optionen (modifizierbar)--

    const char* build_opts = "-cl-std=CL2.0 -cl-fast-relaxed-math";
    err = clBuildProgram(resources.program, 1, &resources.device, build_opts, nullptr, nullptr);

    //--Build-Log ausgeben wenn Fehler oder Info vorhanden--

    size_t log_size = 0;
    clGetProgramBuildInfo(resources.program, resources.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    if (log_size > 1) {
        std::vector<char> log(log_size + 1);
        clGetProgramBuildInfo(resources.program, resources.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "--- OpenCL Build Log ---\n" << log.data() << "\n------------------------\n";
    }

    if (!cl_ok(err, "clBuildProgram")) {
        cleanup_opencl(resources);
        std::exit(1);
    }

    //--Kernel erstellen--

    resources.kernel = clCreateKernel(resources.program, kernel_zhash.cl(), &err);
    if (!cl_ok(err, "clCreateKernel")) {
        cleanup_opencl(resources);
        std::exit(1);
    }

    //--Device-Memory Info (optional: anzeigen)--

    cl_ulong mem_total = 0;
    if (clGetDeviceInfo(resources.device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_total), &mem_total, nullptr) == CL_SUCCESS) {
        double mem_mib = mem_total / 1024.0 / 1024.0;
        std::cerr << "🧠 Device VRAM: " << std::fixed << std::setprecision(1) << mem_mib << " MiB\n";
    }

    //--Puffergrößen anlegen (INPUT_SIZE / HASH_SIZE aus globals.hpp)--

    size_t in_size = static_cast<size_t>(INPUT_SIZE) * std::max(1, intensity);
    size_t out_hashes_size = static_cast<size_t>(HASH_SIZE) * std::max(1, intensity);

    resources.input_buffer = clCreateBuffer(resources.context, CL_MEM_READ_WRITE, in_size, nullptr, &err);
    if (!cl_ok(err, "clCreateBuffer(input_buffer)")) { cleanup_opencl(resources); std::exit(1); }

    resources.output_hashes_buffer = clCreateBuffer(resources.context, CL_MEM_READ_WRITE, out_hashes_size, nullptr, &err);
    if (!cl_ok(err, "clCreateBuffer(output_hashes_buffer)")) { cleanup_opencl(resources); std::exit(1); }

    resources.pool_target_buffer = clCreateBuffer(resources.context, CL_MEM_READ_ONLY, TARGET_SIZE_BYTES, nullptr, &err);
    if (!cl_ok(err, "clCreateBuffer(pool_target_buffer)")) { cleanup_opencl(resources); std::exit(1); }

    std::cout << "✅ OpenCL initialisiert (Kernel: " << zhash.cl << ")\n";
}

void cleanup_opencl(GpuResources &resources) {
    if (resources.input_buffer) { clReleaseMemObject(resources.input_buffer); resources.input_buffer = nullptr; }
    if (resources.output_buffer) { clReleaseMemObject(resources.output_buffer); resources.output_buffer = nullptr; } // falls verwendet
    if (resources.output_hashes_buffer) { clReleaseMemObject(resources.output_hashes_buffer); resources.output_hashes_buffer = nullptr; }
    if (resources.pool_target_buffer) { clReleaseMemObject(resources.pool_target_buffer); resources.pool_target_buffer = nullptr; }

    if (resources.kernel) { clReleaseKernel(resources.kernel); resources.kernel = nullptr; }
    if (resources.program) { clReleaseProgram(resources.program); resources.program = nullptr; }
    if (resources.queue) { clReleaseCommandQueue(resources.queue); resources.queue = nullptr; }
    if (resources.context) { clReleaseContext(resources.context); resources.context = nullptr; }

    resources.device = nullptr;
}
