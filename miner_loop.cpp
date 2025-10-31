#include "miner_loop.hpp"
#include "mining_job.hpp"
#include "opencl_utils.hpp"

#include <CL/cl.h>
#include <array>
#include <atomic> //--Atomic Nutzen Ja Nein Vielleicht erstmal Ja--
#include <cctype> //--Für std::isxdigit--
#include <chrono> //--Für Uhrzeit der Netzwekoperation--
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random> //--Für std::random_device und std::mt19937--
#include <string>
#include <thread>
#include <vector> //--FÜr Intel ARC GPUs DG2 Alchmemist--

//--Externe Status-Variablen--
//--Diese Variablen sind nicht in dieser Datei definiert, sondern werden von außen bereitgestellt.--
//--Sie dienen dazu, den Abbruch des Minings oder den Status der Socket-Verbindung zu signalisieren.--
//--Es sind einfache bool-Werte, die direkt gelesen werden.--

extern std::atomic<bool> abort_mining;
extern std::atomic<bool> socket_valid;
extern std::atomic<bool> job_wurde_übernommen;

//--Globale OpenCL-Objekte--

cl_context context = nullptr;
cl_command_queue queue = nullptr;
cl_kernel kernel = nullptr;
cl_program program = nullptr;
cl_device_id device = nullptr;

//--🧱 Erstellt den Eingabepuffer aus dem MiningJob--

namespace {

    //--Prüft, ob ein Zeichen eine Hexadezimalziffer ist--
    //--Eine Hexadezimalziffer ist 0-9 oder A-F (Groß- oder Kleinbuchstaben).--


    inline bool is_hex_char(unsigned char c) {
        return std::isxdigit(c) != 0;
    }

    //--Prüft, ob ein String ein gültiger Hexadezimal-String ist--
    //--Ein String ist gültig, wenn er leer ist oder nur Hexadezimalziffern enthält--
    //--und eine gerade Länge hat (da ein Byte aus zwei Hex-Ziffern besteht).--
    //--Optional kann ein "0x"-Präfix erlaubt sein.--

    bool is_valid_hex(const std::string& s, bool allow_0x_prefix = true) {
        if (s.empty()) return false;

        std::string clean = s;
        if (allow_0x_prefix && s.size() >= 2 &&
            s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            clean = s.substr(2); //--Präfix entfernen, wenn erlaubt--
            }

            if (clean.empty() || (clean.size() % 2) != 0) return false; //--Muss eine gerade Länge haben--

            for (unsigned char c : clean) {
                if (!is_hex_char(c)) return false; //--Alle Zeichen müssen Hex-Ziffern sein--
            }
            return true;
    }

    //--Entfernt ein optionales "0x"-Präfix von einem Hex-String--
    //--Wenn der String mit "0x" oder "0X" beginnt, wird dieser Teil entfernt.--

    std::string remove_0x_prefix(const std::string& s) {
        if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            return s.substr(2);
        }
        return s;
    }

    //--Konvertiert einen Hex-String in Bytes und hängt sie an einen Puffer an--
    //--Diese Funktion nimmt einen Hex-String und wandelt jedes Paar von Hex-Ziffern--
    //--in ein einzelnes Byte um, das dann einem cl_uchar-Vektor hinzugefügt wird.--
    //--Sie beinhaltet eine verbesserte Fehlerbehandlung für ungültige Eingaben.--

    void append_hex_to_buffer(const std::string& hex, std::vector<cl_uchar>& buffer,
                              const std::string& field_name = "") {
        if (hex.empty()) return;

        std::string clean_hex = remove_0x_prefix(hex);
        if (!is_valid_hex(clean_hex, false)) { //--Prüfen ohne Präfix--
            throw std::invalid_argument("Ungültiger Hex-String für Feld '" +
            field_name + "': " + hex);
        }

        buffer.reserve(buffer.size() + clean_hex.size() / 2); //--Speicher im Voraus reservieren--
        for (size_t i = 0; i < clean_hex.size(); i += 2) {
            try {
                unsigned long byte_val = std::stoul(clean_hex.substr(i, 2), nullptr, 16);
                if (byte_val > 0xFF) { //--Ein Byte ist max. 255 (0xFF)--
                    throw std::out_of_range("Byte-Wert außerhalb des Bereichs");
                }
                buffer.push_back(static_cast<cl_uchar>(byte_val));
            } catch (const std::exception& e) {
                throw std::invalid_argument("Konvertierungsfehler in Feld '" +
                field_name + "' bei Position " +
                std::to_string(i) + ": " + e.what());
            }
        }
                              }

                              //--Erstellt den Eingabepuffer für den OpenCL-Kernel aus einem MiningJob--
                              //--Diese Funktion sammelt alle relevanten Hex-Strings aus dem MiningJob-Objekt--
                              //--(Version, Prevhash, Ntime, Coinb1, Extranonce1, Extranonce2, Coinb2 und Merkle-Branch)--
                              //--und konvertiert sie in einen Vektor von Bytes, der als Eingabe für die GPU dient.--

                              void build_input_from_job(const MiningJob& job, std::vector<cl_uchar>& input_buffer) {
                                  input_buffer.clear();

                                  try {
                                      append_hex_to_buffer(job.version, input_buffer, "version");
                                      append_hex_to_buffer(job.prevhash, input_buffer, "prevhash");
                                      append_hex_to_buffer(job.ntime, input_buffer, "ntime");
                                      append_hex_to_buffer(job.coinb1, input_buffer, "coinb1");
                                      append_hex_to_buffer(job.extranonce1, input_buffer, "extranonce1");
                                      append_hex_to_buffer(job.extranonce2, input_buffer, "extranonce2");
                                      append_hex_to_buffer(job.coinb2, input_buffer, "coinb2");

                                      for (size_t i = 0; i < job.merkle_branch.size(); ++i) {
                                          append_hex_to_buffer(job.merkle_branch[i], input_buffer,
                                                               "merkle_branch[" + std::to_string(i) + "]");
                                      }
                                  } catch (const std::exception& e) {


                                      input_buffer.clear(); //--Puffer im Fehlerfall leeren--
                                      throw; //--Fehler weitergeben--
                                  }
                              }

                              //--Sichere Konvertierung eines Hex-Strings in einen 32-Bit-Integer (uint32_t)--
                              //--Diese Funktion wandelt einen Hex-String in eine vorzeichenlose 32-Bit-Ganzzahl um.--
                              //--Sie prüft auf Gültigkeit des Hex-Strings und stellt sicher, dass der Wert nicht--
                              //--über den maximalen Wert von uint32_t hinausgeht, um Überläufe zu vermeiden.--

                              std::optional<uint32_t> safe_stoul_hex_u32(const std::string& hex) {
                                  std::string clean_hex = remove_0x_prefix(hex);
                                  if (!is_valid_hex(clean_hex, false)) return std::nullopt; //--Prüfen ohne Präfix--

                                  try {
                                      size_t idx = 0;
                                      unsigned long v = std::stoul(clean_hex, &idx, 16);

                                      if (idx != clean_hex.size()) return std::nullopt; //--Nicht alle Zeichen gelesen--
                                      if (v > std::numeric_limits<uint32_t>::max()) return std::nullopt; //--Wert zu groß--

                                      return static_cast<uint32_t>(v);
                                  } catch (...) {
                                      return std::nullopt; //--Konvertierungsfehler--
                                  }
                              }

                              //--Verbesserte OpenCL-Fehlerbehandlung-
                              //--Diese Funktion prüft den Rückgabewert eines OpenCL-Aufrufs. Wenn ein Fehler auftritt,--
                              //--wird eine Fehlermeldung ausgegeben und optional das Build-Log des Kernels,--
                              //--falls die GpuResources verfügbar sind und ein Fehler im Build-Prozess vorlag.--

                              bool check_cl(cl_int err, const char* where, const GpuResources* resources = nullptr) {
                                  if (err == CL_SUCCESS) return true; //--Alles in Ordnung--

                                  std::cerr << "❌ OpenCL-Fehler (" << err << ") bei: " << where << "\n";

                                  //--Build-Log ausgeben, falls Programm und Gerät bekannt sind--

                                  if (resources && resources->program && resources->device) {
                                      size_t log_size = 0;
                                      clGetProgramBuildInfo(resources->program, resources->device,
                                                            CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

                                      if (log_size > 0) { //--Wenn ein Log vorhanden ist--
                                          std::vector<char> build_log(log_size + 1); //--Dynamisch Puffer allozieren--

                                          clGetProgramBuildInfo(resources->program, resources->device,
                                                                CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
                                          std::cerr << "Build-Log:\n" << build_log.data() << "\n";
                                      }
                                  }

                                  return false; //--Fehler aufgetreten--
                              }

                              //--RAII-Wrapper für OpenCL-Speicherobjekte (cl_mem)-
                              //--Dieser Wrapper sorgt dafür, dass OpenCL-Speicherobjekte automatisch freigegeben werden,-
                              //--wenn sie nicht mehr benötigt werden (z. B. wenn der Wrapper den Gültigkeitsbereich verlässt).-
                              //--Dies verhindert Speicherlecks und vereinfacht die Fehlerbehandlung.--

                              struct CLMemWrapper {
                                  cl_mem mem = nullptr; // Das OpenCL-Speicherobjekt

                                  explicit CLMemWrapper(cl_mem mem_obj = nullptr) : mem(mem_obj) {}

                                  //--Destruktor: Gibt das Speicherobjekt frei, wenn der Wrapper zerstört wird--

                                  ~CLMemWrapper() { if (mem) clReleaseMemObject(mem); }

                                  //--Kopierkonstruktor und Zuweisungsoperator sind gelöscht (nicht kopierbar)--

                                  CLMemWrapper(const CLMemWrapper&) = delete;
                                  CLMemWrapper& operator=(const CLMemWrapper&) = delete;

                                  //--Move-Konstruktor: Ermöglicht das Verschieben von Besitzrechten--

                                  CLMemWrapper(CLMemWrapper&& other) noexcept : mem(other.mem) {
                                      other.mem = nullptr; // Quelle leeren
                                  }

                                  //--Move-Zuweisungsoperator: Ermöglicht das Verschieben von Besitzrechten--

                                  CLMemWrapper& operator=(CLMemWrapper&& other) noexcept {
                                      if (this != &other) { //--Selbstzuweisung verhindern--
                                          if (mem) clReleaseMemObject(mem); //--Eigenes Objekt freigeben--
                                          mem = other.mem;
                                          other.mem = nullptr; //--Quelle leeren--
                                      }
                                      return *this;
                                  }

                                  //--Konvertierungsoperator: Ermöglicht die implizite Umwandlung in cl_mem--

                                  operator cl_mem() const { return mem; }

                                  //--Expliziter bool-Operator: Prüft, ob ein gültiges Speicherobjekt vorhanden ist--

                                  explicit operator bool() const { return mem != nullptr; }
                              };

                 //--Der Miner-Loop-
                 //--Dies ist die Hauptschleife des Miners. Sie wiederholt die folgenden Schritte,--
                 //--um nach gültigen Lösungen zu suchen:--
                 //--1. Hole den aktuellen Mining-Job.--
                 //--2. Erstelle eine Start-Nonce für diese Batch-Verarbeitung.--
                 //--3. Aktualisiere die Daten auf der Grafikkarte (GPU), falls sich der Job geändert hat.--
                 //--4. Setze die Argumente für den OpenCL-Kernel.--
                 //--5. Starte den Kernel (das ist das eigentliche Rechenprogramm auf der GPU).--
                 //--6. Warte, bis der Kernel fertig ist.--
                 //--7. Lies die Ergebnisse (die gefundenen Lösungen) von der GPU zurück.--
                 //--8. Überprüfe die gefundenen Lösungen.--

 void miner_loop(
    const std::function<MiningJob()> get_current_job,
    const std::function<void(uint32_t, const std::array<uint8_t, 32>&, const MiningJob&)> on_valid_share,
    const GpuResources &resources,
    int intensity) {

    //--1-- Hole den aktuellen Mining-Job
    MiningJob current_job;
    uint32_t start_nonce = 0;

    while (true) {
        //--1-- Hole den aktuellen Mining-Job
        MiningJob new_job = get_current_job();

        //-- Prüfe ob sich der Job geändert hat
        if (new_job.job_id != current_job.job_id) {
            current_job = new_job;
            start_nonce = 0; // Reset Nonce bei Job-Änderung
        } else {
            start_nonce += resources.batch_size; // Increment Nonce für nächsten Batch
        }

        //--2-- Erstelle eine Start-Nonce für diese Batch-Verarbeitung
        uint32_t batch_start_nonce = start_nonce;

        //--3-- Aktualisiere die Daten auf der GPU, falls sich der Job geändert hat
        if (current_job.job_id != new_job.job_id || resources.need_data_update) {
            update_gpu_data(resources, current_job);
        }

        //--4-- Setze die Argumente für den OpenCL-Kernel
        set_kernel_arguments(resources.kernel, resources, current_job, batch_start_nonce);

        //--5-- Starte den Kernel (das ist das eigentliche Rechenprogramm auf der GPU)
        cl_int error = clEnqueueNDRangeKernel(resources.command_queue, resources.kernel,
                                             1, NULL, &resources.global_work_size,
                                             &resources.local_work_size, 0, NULL, NULL);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("Fehler beim Starten des Kernels: " + std::to_string(error));
        }

        //--6-- Warte, bis der Kernel fertig ist
        clFinish(resources.command_queue);

        //--7-- Lies die Ergebnisse (die gefundenen Lösungen) von der GPU zurück
        std::vector<uint32_t> results(resources.results_buffer_size);
        error = clEnqueueReadBuffer(resources.command_queue, resources.results_buffer, CL_TRUE,
                                  0, results.size() * sizeof(uint32_t), results.data(),
                                  0, NULL, NULL);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("Fehler beim Lesen der Ergebnisse: " + std::to_string(error));
        }

        //--8-- Überprüfe die gefundenen Lösungen
        for (size_t i = 0; i < results.size(); i += 2) {
            if (results[i] != 0xFFFFFFFF) { // Gültige Lösung gefunden
                uint32_t found_nonce = results[i];
                uint32_t solution_hash = results[i + 1];

                // Überprüfe ob die Lösung wirklich gültig ist
                std::array<uint8_t, 32> hash = calculate_hash_with_nonce(current_job, found_nonce);

                if (is_valid_share(hash, current_job.target)) {
                    //-- Gültige Lösung gefunden - Benachrichtige über Callback
                    on_valid_share(found_nonce, hash, current_job);
                }
            }
        }

        //-- Kurze Pause um CPU zu entlasten (optional)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}


Ergänzung

// GPU-Daten aktualisieren
void update_gpu_data(const GpuResources& resources, const MiningJob& job) {
    // Kopiere Job-Daten (Blockheader, Target, etc.) zur GPU
    clEnqueueWriteBuffer(resources.command_queue, resources.job_data_buffer, CL_TRUE,
                        0, sizeof(MiningJob), &job, 0, NULL, NULL);
}

// Kernel-Argumente setzen
void set_kernel_arguments(cl_kernel kernel, const GpuResources& resources,
                         const MiningJob& job, uint32_t start_nonce) {
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &resources.job_data_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &resources.results_buffer);
    clSetKernelArg(kernel, 2, sizeof(uint32_t), &start_nonce);
}

// Hash mit Nonce berechnen (CPU-seitige Validierung)
std::array<uint8_t, 32> calculate_hash_with_nonce(const MiningJob& job, uint32_t nonce) {
    // Implementierung der Hash-Berechnung (z.B. SHA-256)
    // ...
}

// Überprüfen ob Share gültig ist
bool is_valid_share(const std::array<uint8_t, 32>& hash, const std::array<uint8_t, 32>& target) {
    return std::memcmp(hash.data(), target.data(), 32) < 0; // Hash muss kleiner als Target sein
}


) {

  //--Initialisierung des Zufallszahlengenerators--
  //--Ein guter Zufallszahlengenerator ist wichtig für die Nonce.--
  //--Ein guter Zufallszahlengenerator ist wichtig für die Nonce.--
  //--Ein guter Zufallszahlengenerator ist wichtig für die Nonce.--

                     std::random_device rd; //--Quelle für echte Zufallszahlen--
                     std::mt19937 rng(rd()); //--Mersenne Twister Engine--
                     std::uniform_int_distribution<uint32_t> dist; //--Für 32-Bit Zufallszahlen--

                     //--SYCL-Beispiel für ulong8-Vergleiche--
                     //--Holen des initialen Mining-Jobs--
                     //--Der erste Job, mit dem der Miner startet.--

                     MiningJob current_job = get_current_job();
                     std::string current_job_id = current_job.job_id;

                     //--Work-Size berechnen--
                     //--Dies sind die Größen, die der GPU sagen, wie viele Arbeitseinheiten--
                     //--sie gleichzeitig verarbeiten soll. Hardcoded für einfache Kontrolle.--

                      sycl::vec<ulong, 8> nonce = ...; //--512-bit Nonce-Berechnung--
                      sycl::vec<ulong, 8> hash = blake2b(nonce); //--SIMD-optimierte Hashfunktion--
                     const size_t local_work_size = 64; //--Größe einer lokalen Arbeitsgruppe--
                     const size_t min_intensity = 1;    //--Mindestintensität, kann extern kommen--
                     const size_t batch_size = min_intensity * 4096ULL; //--Gesamtmenge der Nonces pro Batch--

                      //--Globale Work-Size ist ein Vielfaches der lokalen, um volle Gruppen zu gewährleisten--

 size_t global_work_size =
      ((batch_size + local_work_size - 1) / local_work_size) * local_work_size;

      //--Host-Puffer für Ergebnisse--
      //--Hier werden die Ergebnisse von der GPU zwischengespeichert, bevor sie verarbeitet werden.--

  //--2-- Target berechnen--

  auto clean_bits = sanitize_hex_string(job.nbits);
  auto maybe_bits = safe_stoul_hex(clean_bits);
  if (!maybe_bits)
    return;
  std::vector<uint8_t> target = bits_to_target(*maybe_bits);

  //--3-- Puffer vorbereiten--

  std::vector<cl_uchar> host_input_buffer;
  build_input_from_job(job, host_input_buffer);
  host_input_buffer.resize(512, 0);


                     //--Globale Work-Size ist ein Vielfaches der lokalen, um volle Gruppen zu gewährleisten--
                     size_t global_work_size =
                     const size_t global_work_size = ((batch_size + local_work_size - 1) / local_work_size) * local_work_size;

                     //--Host-Puffer für Ergebnisse--
                     //--Hier werden die Ergebnisse von der GPU zwischengespeichert, bevor sie verarbeitet werden.--

 std::vector<cl_uchar> output_buffer(32 * batch_size); //--Für die finalen Hashes (32 Bytes pro Hash)--
  std::vector<cl_uint> index_buffer(2 * batch_size, 0); //--Für die Indizes der gefundenen Paare (idxA, idxB)--

       //--4--OpenCL-Puffer für die GPU anlegen--
       //--Diese Puffer werden nur einmal am Anfang erstellt und dann immer wiederverwendet.--
       //--Das ist effizienter, als sie in jeder Schleife neu zu erstellen.--

        cl_int err = CL_SUCCESS; //--Für die Fehlerprüfung der OpenCL-Aufrufe--

        //--Eingabepuffer für die Job-Daten--

        CLMemWrapper cl_input(clCreateBuffer(resources.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     512, host_input_buffer.data(), &err);
        if (!check_cl(err, "clCreateBuffer(cl_input)", &resources)) return;

        //--Ausgabepuffer für die Hashes, die der Kernel erzeugt--

        CLMemWrapper cl_output = clCreateBuffer(resources.context, CL_MEM_WRITE_ONLY,
                                    output_buffer.size(), nullptr, &err);
        if (!check_cl(err, "clCreateBuffer(cl_output)", &resources)) return;

        //--Puffer für die Indizes der Hash-Paare--

        CLMemWrapper cl_indexes =
        clCreateBuffer(resources.context, CL_MEM_WRITE_ONLY,
                     index_buffer.size() * sizeof(cl_uint), nullptr, &err);
        if (!check_cl(err, "clCreateBuffer(cl_indexes)", &resources)) return;


        //--Puffer für die Anzahl der gefundenen Lösungen--</pre>

        CLMemWrapper cl_solution_count.mem = clCreateBuffer(
        resources.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);

        if (!check_cl(err, "clCreateBuffer(cl_solution_count)", &resources)) return;
        if (err != CL_SUCCESS) { /* Fehlerbehandlung */
  }

  //--5--Haupt-Mining-Schleife--
  //--Diese Schleife läuft, solange das Mining nicht abgebrochen werden soll--
  //--und die Socket-Verbindung gültig ist.--

    while (!abort_mining && socket_valid) { //--Keine .load() mehr, direkte bool-Abfrage--
    uint32_t start_nonce = batch_size;
    ;
    //--Job-Wechsel-Logik--
    //--Wir fragen nach dem neuesten Job. Wenn sich die Job-ID geändert hat,--
    //--dann aktualisieren wir unseren aktuellen Job.--

    MiningJob new_job = get_current_job();
    if (new_job.job_id != current_job_id) {
      std::cout << "🔄 Neuer Job empfangen: " << new_job.job_id << "\n";
      current_job = new_job;
      current_job_id = new_job.job_id; //--Job-ID aktualisieren--


      //--Bei Job-Wechsel müssen die Eingabepuffer für die GPU neu gebaut werden.--
      //--Die Puffer selbst werden NICHT neu erstellt, nur ihr Inhalt aktualisiert.--

    }

    //--Target berechnen--
    //--Der "Target"-Wert bestimmt, wie schwierig es ist, einen gültigen Hash zu finden.--
    //--Er wird aus dem "nbits" des aktuellen Jobs berechnet.--

    auto maybe_bits = safe_stoul_hex_u32(current_job.nbits);
    if (!maybe_bits) {
      std::cerr << "❌ Ungültige nbits: " << current_job.nbits << "\n";

      //--Ein kurzer Schlaf, um die CPU nicht zu überlasten, wenn Fehler auftreten--

      continue; //--Nächste Schleifeniteration versuchen--
    }
    std::vector<uint8_t> target = bits_to_target(*maybe_bits);

    //--Eingabepuffer für den Kernel vorbereiten--
    //--Die Job-Daten werden in ein Format gebracht, das die GPU verarbeiten kann.--

    std::vector<cl_uchar> host_input;
    try {
      build_input_from_job(current_job, host_input);
      host_input.resize(512, 0); //--Sicherstellen, dass der Puffer eine Mindestgröße hat--
    } catch (const std::exception& e) {
      std::cerr << "❌ Fehler beim Erstellen des Eingabepuffers: " << e.what() << "\n";
      continue;
    }

    //--Start-Nonce für diese Arbeitsgruppe--
    //--Eine zufällige Start-Nonce, um verschiedene Teile des Nonce-Raums zu durchsuchen.--

     const uint32_t start_nonce = dist(rng);

    //--Solution-Count auf Null zurücksetzen--
    //--Vor jedem Kernel-Lauf setzen wir den Zähler für die gefundenen Lösungen zurück.--

   const cl_uint zero = 0;
    if (!check_cl(clEnqueueWriteBuffer(resources.queue, cl_solution_count.mem, CL_TRUE, 0,
                         sizeof(cl_uint), &zero, 0, nullptr, nullptr);
                        "clEnqueueWriteBuffer(solution_count=0)", &resources)) {
                        continue;
    }

    //--Daten zur GPU schicken (Inhalt der Puffer aktualisieren)--
    //--Hier werden die vorbereiteten Job-Daten auf die GPU kopiert.--
    //--Die Pufferobjekte bleiben dieselben, nur die Daten ändern sich.--

    if (!check_cl(clEnqueueWriteBuffer(resources.queue, cl_input.mem, CL_TRUE, 0,
      host_input.size(), host_input.data(),
                                       0, nullptr, nullptr),
                                        "clEnqueueWriteBuffer(input_data)", &resources)) {
                                        continue;
                }

    //--Vor der Kernel-Ausführung--
    //--Kernel-Argumente setzen (dem Kernel sagen, welche Puffer er nutzen soll)--
    //--Die Argumente werden jedes Mal neu gesetzt, da sich die Daten oder die Start-Nonce ändern können.--

    if (!check_cl(clSetKernelArg(resources.kernel, 0, sizeof(cl_mem), &cl_input.mem),
      "clSetKernelArg(0, cl_input)", &resources)) continue;
    if (!check_cl(clSetKernelArg(resources.kernel, 1, sizeof(cl_mem), &cl_output.mem),
      "clSetKernelArg(1, cl_output)", &resources)) continue;
    if (!check_cl(clSetKernelArg(resources.kernel, 2, sizeof(cl_mem), &cl_indexes.mem),
      "clSetKernelArg(2, cl_indexes)", &resources)) continue;
    if (!check_cl(clSetKernelArg(resources.kernel, 3, sizeof(cl_mem), &cl_solution_count.mem),
      "clSetKernelArg(3, cl_solution_count)", &resources)) continue;
    if (!check_cl(clSetKernelArg(resources.kernel, 4, sizeof(uint32_t), &start_nonce),
      "clSetKernelArg(4, start_nonce)", &resources)) continue;

    std::cout << "Starting mining loop with:\n";
    std::cout << "  Job ID: " << job.job_id << "\n";
    std::cout << "  PrevHash: " << job.prevhash << "\n";
    std::cout << "  Target: " << job.nbits << "\n";
    std::cout << "  Intensity: " << intensity << "\n";

    //--Den Kernel auf der GPU ausführen!!!--
    //--Die eigentliche Rechenarbeit beginnt.--

    cl_event evt = nullptr; //--Ereignisobjekt für die Kernel-Ausführung--

    err = clEnqueueNDRangeKernel(resources.queue, resources.kernel, 1, nullptr,
                                 &global_work_size, &local_work_size, 0,
                                 nullptr, &event);
    if (!check_cl(err, "clEnqueueNDRangeKernel", &resources)) {
      if (evt) clReleaseEvent(evt); //--Ereignis freigeben--
      continue;
    }
    clWaitForEvents(1, &event); //--Warten, bis der Kernel fertig ist--
    clReleaseEvent(evt);      //--Ereignis freigeben--

    //--Lösungen verarbeiten--
    //--Ergebnisse von der GPU zurücklesen--
    //--Die gefundenen Hashes, Indizes und die Lösungsanzahl werden von der GPU geholt.--

     if (!check_cl(clEnqueueReadBuffer(resources.queue, cl_output.mem, CL_TRUE, 0,
                        output_buffer.size(), output_buffer.data(), 0, nullptr, nullptr);
                         "clEnqueueReadBuffer(output)", &resources)) continue;

     if (!check_cl(clEnqueueReadBuffer(resources.queue, cl_indexes.mem, CL_TRUE, 0,
                        index_buffer.size() * sizeof(cl_uint),
                        index_buffer.data(), 0, nullptr, nullptr);
                        "clEnqueueReadBuffer(indexes)", &resources)) continue;

                        cl_uint solution_count = 0;
                        if (!check_cl(clEnqueueReadBuffer(resources.queue, cl_solution_count.mem, CL_TRUE, 0,
                        sizeof(cl_uint), &solution_count, 0, nullptr, nullptr),
                        "clEnqueueReadBuffer(solution_count)", &resources)) continue;

    //--Gefundene Lösungen verarbeiten (PLATZHALTER)--
    //--HIER kommt deine ECHTE PoW-Validierungslogik rein!--
    //--Aktuell wird ein Platzhalter-XOR-Check durchgeführt.--

    for (size_t i = 0; i < batch_size; ++i) {
      const uint32_t idxA = index_buffer[i * 2];
      const uint32_t idxB = index_buffer[i * 2 + 1];

      if (idxA == 0 && idxB == 0)
        continue;

      //--Gültigkeitsprüfung der Indizes--

      if (idxA >= batch_size || idxB >= batch_size) //--Sollte nicht passieren, aber zur Sicherheit!--
        continue;

      std::array<uint8_t, 32> final_hash; //--32 Bytes für den Hash--
      for (int j = 0; j < 32; ++j) {
        final_hash[j] =
            output_buffer[idxA * 32 + j] ^ output_buffer[idxB * 32 + j];
      }
      const size_t offsetA = static_cast<size_t>(idxA) * 32 + j;
      const size_t offsetB = static_cast<size_t>(idxB) * 32 + j;
      if (is_valid_hash(final_hash, target)) {
        on_valid_share(start_nonce + idxA, final_hash, job);
      }
    }
  }

  //--Aufräumen--

  clReleaseMemObject(cl_input);
  clReleaseMemObject(cl_output);
  clReleaseMemObject(cl_indexes);
  clReleaseMemObject(cl_solution_count);
  clReleaseKernel(resources.kernel);
  clReleaseProgram(resources.program);
  clReleaseCommandQueue(resources.queue);
  clReleaseContext(resources.context);
}

//--Aufräumen (nicht explizit nötig wegen RAII)--
//--Die CLMemWrapper-Objekte (cl_input, cl_output, etc.) geben ihren--
//--Speicher automatisch frei, wenn die Funktion endet.--
