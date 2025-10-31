#pragma once

#include <string>
#include <CL/cl.h>
#include "globals.hpp"

//--Initialisiert OpenCL (einheitliche Signatur)--
//--kernel_path: Pfad zur .cl Datei--
//--kernel_func_name: Name der Kernel-Funktion inside the .cl (z.B. "zhash.cl")--
//--platform_index / device_index: Auswahl--
//--intensity: wieviele Einheiten (wird für Puffergrößen verwendet)--

void init_opencl(const std::string& kernel_path,
                 const std::string& kernel_func_name,
                 int platform_index,
                 int device_index,
                 int intensity,
                 GpuResources& resources);

//--Gibt alle OpenCL-Ressourcen frei (sicher mehrfach aufrufbar)--

void cleanup_opencl(GpuResources& resources);

//--Schreibt das hex-Target in GPU-Puffer (bestehende Signatur)--

void update_opencl_target(const GpuResources& resources, const std::string& hex_target);

//--Optional: kleines Helfer-Interface zum Setzen der Kernel-Args (wenn benötigt)--

void set_kernel_args(const GpuResources& resources,
                     cl_mem solution_indexes_buffer,
                     uint32_t start_nonce);
