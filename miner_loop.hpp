#pragma once

#include "mining_job.hpp"
#include "opencl_utils.hpp"
#include "optional"
#include <array>
#include <functional>
#include <string>

//--Externe Globals--
//--Mining-Steuerung--

void stop_mining();
void miner_loop(
const MiningJob &job,
const std::function<void(uint32_t, const std::array<uint8_t, 32> &,
const MiningJob &)> &on_valid_share,
 const GpuResources &resources, int intensity);

//--Hex-Helfer--

std::string sanitize_hex_string(const std::string &input);
std::optional<uint32_t> safe_stoul_hex(const std::string &hex_str);
