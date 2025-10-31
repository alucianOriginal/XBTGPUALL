#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

//--Enthält einen vollständigen MiningJob basierend auf einem Benachrichtigungs Ereignis--

struct MiningJob {
  std::string job_id;
  std::string prevhash;
  std::string coinb1;
  std::string coinb2;
  std::vector<std::string> merkle_branch;
  std::string version;
  std::string nbits;
  std::string ntime;
  std::string extranonce1;
  std::string extranonce2;
  std::string bits;
};

//--Vergleich: Hash < Target--

inline bool is_valid_hash(const std::array<uint8_t, 32> &hash,
                          const std::vector<uint8_t> &target) {
  for (size_t i = 0; i < 32; ++i) {
    if (hash[i] < target[i])
      return true;
    if (hash[i] > target[i])
      return false;
  }
  return true;
}

//--Wandelt Compact-Format aus "bits" (z. B. 0x1d00ffff) in 256-bit Target!--

inline std::vector<uint8_t> bits_to_target(uint32_t bits) {
  uint32_t exponent = bits >> 24;
  uint32_t mantissa = bits & 0x007fffff;

  std::vector<uint8_t> target(32, 0);

  if (exponent <= 3) {

    //--Mantisse nach rechts schieben--

    mantissa >>= 8 * (3 - exponent);
    target[31] = mantissa & 0xFF;
    target[30] = (mantissa >> 8) & 0xFF;
    target[29] = (mantissa >> 16) & 0xFF;
  } else if (exponent <= 32) {

    //--Platzierung der Mantisse ab dem richtigen Byte--

    int idx = 32 - exponent;
    target[idx] = (mantissa >> 16) & 0xFF;
    target[idx + 1] = (mantissa >> 8) & 0xFF;
    target[idx + 2] = mantissa & 0xFF;
  } else {

    //--Exponent außerhalb gültigem Rahmens--
    //-→Zurück für leeres Ziel (niemals gültig)--

    return std::vector<uint8_t>(32, 0xFF);
  }

  return target;
}
