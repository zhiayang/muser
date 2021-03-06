# Makefile for Orion-X3/Orion-X4/mx and derivatives
# Written in 2011
# This makefile is licensed under the WTFPL

WARNINGS        := -Wno-unused-parameter -Wno-sign-conversion -Wno-padded -Wno-conversion -Wno-shadow -Wno-missing-noreturn -Wno-unused-macros -Wno-switch-enum -Wno-deprecated -Wno-format-nonliteral -Wno-trigraphs -Wno-unused-const-variable -Wno-deprecated-declarations

GCCWARNINGS     := -Wno-init-list-lifetime


CLANGWARNINGS   := -Wno-undefined-func-template -Wno-comma -Wno-nullability-completeness -Wno-redundant-move -Wno-nested-anon-types -Wno-gnu-anonymous-struct -Wno-reserved-id-macro -Wno-extra-semi -Wno-gnu-zero-variadic-macro-arguments -Wno-shift-sign-overflow -Wno-exit-time-destructors -Wno-global-constructors -Wno-c++98-compat-pedantic -Wno-documentation-unknown-command -Wno-weak-vtables -Wno-c++98-compat -Wold-style-cast


OUTPUT          := build/muser

CC              ?= "clang"
CXX             ?= "clang++"

CXXSRC          := $(shell find source external -iname "*.cpp")
CXXOBJ          := $(CXXSRC:.cpp=.cpp.o)
CXXDEPS         := $(CXXSRC:.cpp=.cpp.d)

NUMFILES        := $$(($(words $(CXXSRC))))

DEFINES         := -D__USE_MINGW_ANSI_STDIO=1
SANITISE        :=

CXXFLAGS        += -std=c++17 -fvisibility=hidden -O3 -msse -c -Wall $(SANITISE) $(DEFINES)
LDFLAGS         += $(SANITISE) -fvisibility=hidden

PRECOMP_HDRS    := source/include/precompile.h
PRECOMP_GCH     := $(PRECOMP_HDRS:.h=.h.gch)

UNAME_IDENT     := $(shell uname)
COMPILER_IDENT  := $(shell $(CC) --version | head -n 1)

OPENSSL_CFLAGS  := $(shell pkg-config --cflags openssl)
OPENSSL_LDFLAGS := $(shell pkg-config --libs openssl)

LIBAV_CFLAGS    := $(shell pkg-config --cflags libavformat libavutil libavcodec libswresample)
LIBAV_LDFLAGS   := $(shell pkg-config --libs libavformat libavutil libavcodec libswresample)

LIBVORBIS_CFLAGS    := $(shell pkg-config --cflags vorbis)
LIBVORBIS_LDFLAGS   := $(shell pkg-config --libs vorbis vorbisenc)

CXXFLAGS += -DASIO_STANDALONE=1 $(OPENSSL_CFLAGS) $(LIBAV_CFLAGS) $(LIBVORBIS_CFLAGS)
LDFLAGS  += $(OPENSSL_LDFLAGS) $(LIBAV_LDFLAGS) $(LIBVORBIS_LDFLAGS)


.DEFAULT_GOAL = all
-include $(CXXDEPS)


.PHONY: clean all

all: $(OUTPUT)


$(OUTPUT): $(PRECOMP_GCH) $(CXXOBJ)
	@printf "# linking\n"
	@mkdir -p $(dir $(OUTPUT))
	@$(CXX) -o $@ $(CXXOBJ) $(LDFLAGS)

%.cpp.o: %.cpp
	@$(eval DONEFILES += "CPP")
	@printf "# compiling [$(words $(DONEFILES))/$(NUMFILES)] $<\n"
	@$(CXX) $(CXXFLAGS) $(WARNINGS) -include source/include/precompile.h -Isource/include -Iexternal -MMD -MP -o $@ $<

%.h.gch: %.h
	@printf "# precompiling header $<\n"
	@$(CXX) $(CXXFLAGS) $(WARNINGS) -o $@ $<



# haha
clena: clean
clean:
	@rm -f $(OUTPUT)
	@find source -name "*.o" | xargs rm -f
	@find source -name "*.gch*" | xargs rm -f
	@find source -name "*.pch*" | xargs rm -f

	@find source -name "*.c.m" | xargs rm -f
	@find source -name "*.c.d" | xargs rm -f
	@find source -name "*.cpp.m" | xargs rm -f
	@find source -name "*.cpp.d" | xargs rm -f









