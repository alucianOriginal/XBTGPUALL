#--Hier startet man den Bauvorgang für das Miningprogramm auf seinem Computer um es Einsatzbereit zu machen.--

CXXFLAGS := -std=c++17 -Wall -O2 -DCL_TARGET_OPENCL_VERSION=300 -MMD -MP
LDFLAGS  := -lOpenCL -lboost_system -lboost_json -lpthread

#Quellcode-Dateien--

SRC := main.cpp \
       miner_loop.cpp \
       opencl_utils.cpp \
       stratum_notify_listener.cpp \
       globals.cpp

OBJ  := $(SRC:.cpp=.o)
DEPS := $(OBJ:.o=.d)
OUT  := xbtgpuarc

//--Standard-Ziel/Target--

all: $(OUT)

#Bau des GPU-Miners--

$(OUT): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

#Generisches Compile-Ziel--

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

#Säubern--

clean:
	rm -f $(OUT) $(CPU_OUT) *.o *.d

-include $(DEPS)
