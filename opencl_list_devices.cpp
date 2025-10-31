#include <CL/cl.h>
#include <iostream>
#include <string>
#include <vector>

void check_error(cl_int err, const std::string &msg) {
  if (err != CL_SUCCESS) {
    std::cerr << "❌ Fehler: " << msg << " (" << err << ")\n";
    exit(1);
  }
}

int main() {
  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
  check_error(err, "clGetPlatformIDs (count)");

  std::vector<cl_platform_id> platforms(num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  check_error(err, "clGetPlatformIDs (fetch)");

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

  return 0;
}
