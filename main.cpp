#include "globals.hpp"
#include "miner_loop.hpp"
#include "mining_job.hpp"
#include "notify_parser.hpp"
#include "opencl_utils.hpp"
#include "stratum_notify_listener.hpp"

#include <CL/cl.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

//--Alle OpenCL-Geräte auf dem Computer auflisten--

void list_opencl_devices() {
  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS) {
    std::cerr << "❌ Fehler bei clGetPlatformIDs: " << err << "\n";
    return;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

  std::cout << "🌍 Gefundene OpenCL-Plattformen: " << num_platforms << "\n";

  for (cl_uint i = 0; i < num_platforms; ++i) {
    char name[128], vendor[128], version[128];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name,
                      nullptr);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor,
                      nullptr);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version),
                      version, nullptr);

    std::cout << "\n[Plattform " << i << "]\n";
    std::cout << "  Name:    " << name << "\n";
    std::cout << "  Vendor:  " << vendor << "\n";
    std::cout << "  Version: " << version << "\n";

    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr,
                         &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
      std::cout << "  ⚠️  Keine Geräte gefunden.\n";
      continue;
    }

    std::vector<cl_device_id> devices(num_devices);
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices,
                   devices.data(), nullptr);

    for (cl_uint j = 0; j < num_devices; ++j) {
      char devname[128];
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(devname), devname,
                      nullptr);
      std::cout << "    [Device " << j << "] " << devname << "\n";
    }
  }
}

int main(int argc, char **argv) {

  //--Default-Werte--

  int platform_index = 0;
  int device_index = 0;
  int intensity = 256;
  std::string algo = "zhash_144_5";
  std::string wallet = "Gb4V4a9Jk3p8aH6jkW3Aq3sq8rQCuJQ6S8";
  std::string worker = "A730m";
  std::string password = "x";
  std::string pool_host = "solo-btg.2miners.com";
  int pool_port = 4040;

  //--🧾 Argumente parsen--

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--platform" && i + 1 < argc)
      platform_index = std::atoi(argv[++i]);
    else if (arg == "--device" && i + 1 < argc)
      device_index = std::atoi(argv[++i]);
    else if (arg == "--intensity" && i + 1 < argc)
      intensity = std::atoi(argv[++i]);
    else if (arg == "--algo" && i + 1 < argc)
      algo = argv[++i];
    else if (arg == "--wallet" && i + 1 < argc)
      wallet = argv[++i];
    else if (arg == "--worker" && i + 1 < argc)
      worker = argv[++i];
    else if (arg == "--password" && i + 1 < argc)
      password = argv[++i];
    else if (arg == "--pool" && i + 1 < argc)
      pool_host = argv[++i];
    else if (arg == "--port" && i + 1 < argc)
      pool_port = std::atoi(argv[++i]);
    else if (arg == "--help") {
      std::cout
          << "Usage: ./xbtgpuarc [options]\n"
          << "Options:\n"
          << "  --platform N         OpenCL Plattform-Index (default 0)\n"
          << "  --device N           OpenCL Geräte-Index (default 0)\n"
          << "  --intensity N        Threads pro Gerät (default 256)\n"
          << "  --algo NAME          Kernel/Algo-Name (default zhash_144_5)\n"
          << "  --wallet ADDR        Wallet-Adresse\n"
          << "  --worker NAME        Worker-Name\n"
          << "  --password PASS      Passwort für Pool (default 'x')\n"
          << "  --pool HOST          Pool-Adresse (default 2miners)\n"
          << "  --port PORT          Port (default 4040)\n";
      return 0;
    }
  }

  std::cout << "🚀 Starte XBTGPUARC mit Algo: " << algo << "\n";
  std::cout << "👤 Worker: " << wallet << "." << worker << "\n";
  std::cout << "🎛️  Platform: " << platform_index
            << " | Device: " << device_index << " | Intensity: " << intensity
            << "\n";
  std::cout << "🌐 Pool: " << pool_host << ":" << pool_port << "\n";

  list_opencl_devices();

  //--Initialisiere OpenCL--

  GpuResources resources;
  init_opencl("kernels/zhash.cl",algo, platform_index, device_index, intensity,
              resources);

  //--Starte Stratum-Listener + Mining-Thread--

  run_stratum_listener(pool_host, pool_port, wallet, worker, password,
                       intensity, resources);

  cleanup_opencl(resources);
  return 0;
}
