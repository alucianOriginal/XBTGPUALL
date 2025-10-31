#pragma once

#include "mining_job.hpp"
#include <boost/json.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

//--Diese Funktion parst eine JSON-Zeile vom Pool und wandelt sie in ein--
//-MiningJob-Objekt um.--

inline std::optional<MiningJob> parse_notify(const std::string &line) {
  using namespace boost::json;
  boost::system::error_code ec;
  value json_value = parse(line, ec);

  if (ec || !json_value.is_object()) {
    return std::nullopt; //--Keine gültige JSON-Nachricht--
  }

  const object &obj = json_value.as_object();

  //--Sicherstellen, dass es eine "mining.notify"-Nachricht ist--

  if (!obj.contains("method") ||
      value_to<std::string>(obj.at("method")) != "mining.notify") {
    return std::nullopt;
  }

  if (!obj.contains("params") || !obj.at("params").is_array()) {
    std::cerr << "❌ Fehler: 'mining.notify' hat keine gültigen Parameter.\n";
    return std::nullopt;
  }
  const array &params = obj.at("params").as_array();

  //--Die Parameter-Anzahl für BTG auf 2miners.com ist typischerweise 8 oder mehr--
  //--Das Programm wird am Ende auf Solo und Poolmining zur Auswahl aufgestockt, bis diese Zeilen Ersetzt wurden.--

  if (params.size() < 8) {
    std::cerr << "❌ Fehler: 'mining.notify' hat zu wenige Parameter ("
              << params.size() << "). Erwartet >= 8.\n";
    return std::nullopt;
  }

  MiningJob job;

  //--Parameterzuweisung LOG
  //--Stratum-Protokoll für solo-btg.2miners.com:--
  //--params[0]: job_id--
  //--params[1]: version--
  //--params[2]: prevhash--
  //--params[3]: coinb1--
  //--params[4]: coinb2--
  //--params[5]: nbits--
  //--params[6]: ntime--
  //--params[7]: clean_job (boolean)--

  //-WICHTIG: Dieser Pool sendet KEINEN separaten 'merkle_branch'.--
  //-Der Merkle-Root muss vom Miner selbst berechnet werden, indem--
  //-die Coinbase-Transaktion gehasht wird. Die 'merkle_branch'--
  //-Liste bleibt also absichtlich leer.--

  job.job_id = value_to<std::string>(params.at(0));
  job.version = value_to<std::string>(params.at(1));
  job.prevhash = value_to<std::string>(params.at(2));
  job.coinb1 = value_to<std::string>(params.at(3));
  job.coinb2 = value_to<std::string>(params.at(4));
  job.nbits = value_to<std::string>(params.at(5));
  job.ntime = value_to<std::string>(params.at(6));
  job.clean_job = params.at(7).as_bool();

  //--Die Merkle Branch Liste wird explizit geleert, da sie nicht vom Pool kommt.--

  job.merkle_branch.clear();
  std::cout << "🌿 Job korrekt geparst. Merkle Branch ist leer, wie vom Pool "
               "erwartet.\n";

  //--Alte Felder für Kompatibilität füllen--

  job.bits = job.nbits;
  job.extranonce1 = "";         //--Wird später vom Unterschreiber gesetzt--
  job.extranonce2 = "00000000"; //--Platzhalter--

  //--Debug-Ausgabe--


  std::cout << "🔍 Debug Notify: bits = '" << job.nbits << "', ntime = '"
            << job.ntime << "'\n";

  return job;
}
