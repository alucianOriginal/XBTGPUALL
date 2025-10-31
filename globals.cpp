#include "globals.hpp"
#include "miner_loop.hpp"

//--Globale Variablen definieren--

bool abort_mining = false;
bool socket_valid = false;

int next_request_id = 1;
std::string current_job_id = "";
std::string worker_name = "";

std::array<uint8_t, 32> current_target = {};

//--Funktion implementieren--

void stop_mining() { abort_mining = true; }
