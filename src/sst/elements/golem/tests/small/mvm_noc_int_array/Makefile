CC=$(ARCH)-linux-musl-gcc
CXX=$(ARCH)-linux-musl-g++

LDFLAGS=-static -fopenmp
LDFLAGS_PTHREAD=-static -pthread
LDFLAGS_STATIC=-static

PROG1=mvm_noc_int_array
PROG3=mvm_single_router
PROG4=mvm_pipeline_mainmem
PROG5=test_atomic_16cores_pthread
PROG6=test_hbm2
PROG7=test_noc_dma
PROG8=test_pipeline_mailbox_pthread
PROG9=test_pipeline_mailbox_software
PROG10=test_pipeline_mailbox_enhanced

all: $(PROG7)

$(PROG1) : $(PROG1).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(ARCH)/$@ $<

$(PROG3) : $(PROG3).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(ARCH)/$@ $<

$(PROG4) : $(PROG4).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(ARCH)/$@ $<

$(PROG5) : $(PROG5).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $(ARCH)/$@ $<

$(PROG6) : $(PROG6).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS_STATIC) -o $(ARCH)/$@ $<

$(PROG7) : $(PROG7).cpp golem_matmul_runtime.cpp
	$(CXX) $(CFLAGS) $(LDFLAGS_STATIC) -o $(ARCH)/$@ $(PROG7).cpp golem_matmul_runtime.cpp

$(PROG8) : $(PROG8).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS_STATIC) -o $(ARCH)/$@ $<

$(PROG9) : $(PROG9).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) $(EXTRA_CFLAGS) -o $(ARCH)/$@ $<

$(PROG10) : $(PROG10).cpp
	$(CXX) $(CFLAGS) $(LDFLAGS_STATIC) $(EXTRA_CFLAGS) -o $(ARCH)/$@ $<

clean:
	rm -rf $(ARCH)/$(PROG7)
