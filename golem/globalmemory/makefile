# Acquire compile environment from sst-config utility
CXX=$(shell sst-config --CXX)
CXXFLAGS=$(shell sst-config --ELEMENT_CXXFLAGS)
LDFLAGS=$(shell sst-config --ELEMENT_LDFLAGS)

# make all - compile libPingPong.so and register (install) it
all: libPingPong.so install

# Compile library
libPingPong.so: sender.o responder.o
	$(CXX) $(CXXFLAGS) sender.o responder.o -o $@ $(LDFLAGS)

# Install (register) library with SST-Core using sst-register utility
install:
	sst-register PingPong PingPong_LIBDIR=$(CURDIR)

clean:
	rm -rf *.o libPingPong.so