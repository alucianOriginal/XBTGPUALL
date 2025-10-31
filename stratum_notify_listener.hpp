#pragma once
#include "opencl_utils.hpp"
#include <string>

//--Startet Stratum-Connection, empfängt Jobs, startet Miner, horcht auf Arbeit vom Pool.--

void run_stratum_listener(const std::string &pool_host, int pool_port,
                          const std::string &wallet, const std::string &worker,
                          const std::string &password, int intensity,
                          GpuResources &gpu_resources); //--← NICHT const--
