
#include "stratum_notify_listener.hpp"
#include "globals.hpp"
#include "miner_loop.hpp"
#include "notify_parser.hpp" //--Wir nutzen jetzt den externen Parser!--
#include "opencl_utils.hpp"

#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

using boost::asio::ip::tcp;

 //--Globale Variablen, die von anderen Teilen des Programms gesetzt werden--

extern int next_request_id;
extern std::string current_job_id;
extern std::string worker_name;

 //--Funktion zum Senden eines gefundenen Shares an den Pool--

void submit_share(tcp::socket &socket, const std::string &nonce_hex,
                  const std::string &ntime_hex, const std::string &job_id) {
  using namespace boost::json;

  //--Erstellt die Parameter für die "mining.submit" Methode--

  array params;
  params.emplace_back(worker_name);
  params.emplace_back(job_id);
  params.emplace_back(
      "00000000"); //--extranonce2 Platzhalter, oft nicht benötigt--
  params.emplace_back(ntime_hex);
  params.emplace_back(nonce_hex);

  //--Baut die komplette JSON-RPC-Anfrage zusammen--

  object request;
  request["id"] = next_request_id++;
  request["method"] = "mining.submit";
  request["params"] = params;

  //--Sendet die Nachricht an den Pool--

  std::string message = serialize(request) + "\n";
  boost::asio::write(socket, boost::asio::buffer(message));
  std::cout << "📤 Share für Job " << job_id << " gesendet:\n" << message;
}

//--Hauptfunktion, die die Verbindung zum Stratum-Pool hält und auf Nachrichten lauscht--

void run_stratum_listener(const std::string &pool_host, int pool_port,
                          const std::string &wallet, const std::string &worker,
                          const std::string &password, int intensity,
                          GpuResources &gpu_resources) {
  const std::string port_str = std::to_string(pool_port);
  worker_name = wallet + "." + worker; //--Setzt den globalen Worker-Namen--

  try {
    boost::asio::io_context io_context;
    tcp::resolver resolver(io_context);
    auto endpoints = resolver.resolve(pool_host, port_str);
    tcp::socket socket(io_context);
    boost::asio::connect(socket, endpoints);

    std::cout << "📡 Verbunden mit " << pool_host << ":" << port_str << "\n";

    //--Standard-Nachrichten zur Anmeldung am Pool--

    std::string subscribe =
        R"({"id": 1, "method": "mining.subscribe", "params": []})"
        "\n";
    std::string authorize =
        R"({"id": 2, "method": "mining.authorize", "params": [")" +
        worker_name + R"(", ")" + password +
        R"("]})"
        "\n";

    boost::asio::write(socket, boost::asio::buffer(subscribe));
    boost::asio::write(socket, boost::asio::buffer(authorize));

    std::string buffer; //--Puffer für eingehende Daten vom Socket--
    static std::thread mining_thread;

    //--Endlosschleife zum Lesen von Nachrichten--

    for (;;) {
      char reply[4096];
      boost::system::error_code error;
      size_t len = socket.read_some(boost::asio::buffer(reply), error);
      if (len == 0 && error)
        break; //--Verbindung geschlossen oder Fehler--

      buffer.append(reply, len);
      size_t pos = 0;

      //--Verarbeite jede vollständige Zeile (getrennt durch '\n') im Puffer--

      while ((pos = buffer.find('\n')) != std::string::npos) {
        std::string line = buffer.substr(0, pos);
        buffer.erase(0, pos + 1);
        std::cout << "🌐 Nachricht:\n" << line << "\n";

        //--Versuch, die Nachricht als Mining-Job zu parsen--

        auto job_opt = parse_notify(line);

        if (job_opt) { //--Wenn ein gültiger Job empfangen wurde--
          auto &job = *job_opt;
          current_job_id = job.job_id; //--Update der globalen Job-ID--

          std::cout << "🎯 Job ID: " << job.job_id << "\n";
          std::cout << "🧱 PrevHash: " << job.prevhash << "\n";

          //--Stoppe den alten Mining-Prozess, falls er noch läuft--

          if (mining_thread.joinable()) {
            stop_mining();
            mining_thread.join();
          }

          //--Definiere eine Lambda-Funktion, die aufgerufen wird, wenn eine--
          //-Lösung gefunden wird!!!--

          auto share_submitter = [&](uint32_t nonce,
                                     const std::array<uint8_t, 32> &hash,
                                     const MiningJob &job) {
            std::stringstream ss_nonce;
            ss_nonce << std::hex << std::setw(8) << std::setfill('0') << nonce;

            //--Keine Umwandlung von job.ntime nötig, ist schon hex-String--

            submit_share(socket, ss_nonce.str(), job.ntime, job.job_id);
          };

          //--Starte den neuen Mining-Prozess in einem eigenen Thread--

          mining_thread = std::thread([&, job]() {
            miner_loop(job, share_submitter, gpu_resources, intensity);
          });
        }
      }
      if (error == boost::asio::error::eof)
        break;
      else if (error)
        throw boost::system::system_error(error);
    }

    if (mining_thread.joinable()) {
      stop_mining();
      mining_thread.join();
    }

  } catch (const std::exception &e) {
    std::cerr << "❌ Stratum-Fehler: " << e.what() << "\n";
  }
}
